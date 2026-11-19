# SPDX-License-Identifier: Apache-2.0
"""One serial owner per bus, with calibrated feedback and grouped arm control."""

import json
import math
import secrets
from pathlib import Path
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.task import Future
from rclpy.time import Time
from rcl_interfaces.msg import ParameterDescriptor
from std_srvs.srv import Trigger
from std_msgs.msg import String

from xyc_interfaces.msg import ArmState, MotorFeedback, MotorFeedbackArray
from xyc_interfaces.srv import ArmBusTransaction as ArmBusTransactionSrv, ProtectiveStop
from xyc.robot.calibration import FEEDBACK_ENDPOINT_TOLERANCE_TICKS, arm_names, normalize_tick
from xyc_hardware.arm_state import ArmStateConverter
from xyc.robot.feetech import FeetechReader, MODEL_NAMES, decode_feedback
from xyc_hardware.arm_joint_control import (
    GRIPPER_TORQUE_LIMIT_RAW_DEFAULT, ArmBusHold, ArmBusTransaction,
    BoundJointDriver, JointControl, JointDriver,
)
from xyc_hardware.chassis_control import ChassisDriver, ChassisRos
from xyc_hardware.motion_stop import stop_service
from xyc.transport.teleop import TARGET_COMMAND_EXPIRED


def motor_launch_parameters(bus, calibration_dir, *, calibrated, group_mode,
                            chassis_mode, gripper_torque_limit_raw):
    from xyc.robot.profile import robot_profile

    return {
        **robot_profile().motor_parameters(bus),
        'calibration_file': str(calibration_dir / f'{bus}.json') if calibrated else '',
        'group_mode': group_mode,
        'chassis_mode': chassis_mode and bus == 'left',
        'gripper_torque_limit_raw': gripper_torque_limit_raw,
        **({'poll_hz': 60.0, 'read_timeout_s': .014} if group_mode else {}),
    }


class MotorStateNode(Node):
    def __init__(self, *, teleop_link=None, **kwargs):
        super().__init__('motor_state', **kwargs)
        self.teleop_link = teleop_link
        self.reader = None
        self.state_converter = None
        self.state_publisher = None
        self.recording_publisher = self.create_publisher(String, 'recording_sample', qos_profile_sensor_data)
        self.joint_controls = {}
        self.group_mode = False
        self.arm_transaction = None
        self.group_session_id = ''
        self.chassis = None
        self.motion_fault = ''
        self.stop_confirmed = False
        self._sequence = 0
        self._next_warning = 0.0
        self._next_state_warning = 0.0
        try:
            self.declare_parameter('port', '')
            self.declare_parameter('bus', '')
            self.declare_parameter('baudrate', 1_000_000)
            self.declare_parameter('poll_hz', 30.0)
            self.declare_parameter('read_timeout_s', 0.02)
            self.declare_parameter('calibration_file', '')
            self.declare_parameter('group_mode', False, ParameterDescriptor(read_only=True))
            self.declare_parameter('chassis_mode', False, ParameterDescriptor(read_only=True))
            self.declare_parameter('gripper_torque_limit_raw', GRIPPER_TORQUE_LIMIT_RAW_DEFAULT,
                                   ParameterDescriptor(read_only=True))
            self.declare_parameter('motor_ids', Parameter.Type.INTEGER_ARRAY)
            self.declare_parameter('motor_names', Parameter.Type.STRING_ARRAY)
            self.declare_parameter('model_numbers', Parameter.Type.INTEGER_ARRAY)
            def value(name):
                return self.get_parameter(name).value

            port = value('port')
            self.bus = value('bus')
            self.ids = list(value('motor_ids'))
            self.names = list(value('motor_names'))
            expected_models = list(value('model_numbers'))
            poll_hz = value('poll_hz')
            timeout = value('read_timeout_s')
            group_mode = value('group_mode')
            chassis_mode = value('chassis_mode')
            gripper_torque_limit_raw = value('gripper_torque_limit_raw')
            if (type(gripper_torque_limit_raw) is not int
                    or not 1 <= gripper_torque_limit_raw <= 1000):
                raise ValueError('gripper_torque_limit_raw must be an integer in [1, 1000]')
            if chassis_mode and self.bus != 'left':
                raise ValueError('Chassis control belongs to the left bus')
            self.group_mode = group_mode
            if group_mode and not value('calibration_file'):
                raise ValueError('Group control requires an arm calibration file')
            if not port or self.bus not in ('left', 'right'):
                raise ValueError('Provide port and bus=left/right')
            if not (len(self.ids) == len(self.names) == len(expected_models)):
                raise ValueError('IDs, names and model_numbers must have equal lengths')
            if any(not name for name in self.names) or len(set(self.names)) != len(self.names):
                raise ValueError('Motor names must be nonempty and unique')
            if any(number not in MODEL_NAMES for number in expected_models):
                raise ValueError('Unsupported STS model number')
            if not math.isfinite(poll_hz) or poll_hz <= 0:
                raise ValueError('poll_hz must be finite and positive')
            if not math.isfinite(timeout) or not 0 < timeout < 1.0 / poll_hz:
                raise ValueError('read_timeout_s must be positive and below the polling period')

            reader_class = JointDriver if group_mode else FeetechReader
            control_options = {'bus': self.bus, 'joint_id': 6} if group_mode else {}
            if chassis_mode:
                reader_class = ChassisDriver
                control_options = {'bus': self.bus, 'joint_id': 6}
            self.reader = reader_class(
                port, self.ids, baudrate=value('baudrate'), timeout_s=timeout,
                clock_ns=lambda: self.get_clock().now().nanoseconds,
                **control_options,
            )
            if teleop_link is not None:
                self.reader.motion_gate = teleop_link.dispatch
            self.reader.open()
            self.models = self.reader.read_models()
            for motor_id, expected in zip(self.ids, expected_models):
                if self.models[motor_id] != expected:
                    raise ValueError(
                        f'ID {motor_id}: expected model {expected}, '
                        f'received {self.models[motor_id]}'
                    )
            if group_mode:
                self.reader.verify_speed_units()
                speed, acceleration = self.reader.motion_profile(1)
                gripper_speed, gripper_acceleration = self.reader.motion_profile(7)
                self.get_logger().info(
                    f'Continuous-control servo profile: joints {speed}/{acceleration}, '
                    f'gripper {gripper_speed}/{gripper_acceleration} (speed/acceleration)'
                )
            calibration_path = value('calibration_file')
            if calibration_path:
                document = json.loads(Path(calibration_path).expanduser().read_text())
                names_to_ids = dict(zip(self.names, self.ids))
                selected = {name: names_to_ids[name] for name in arm_names(self.bus)}
                settings = self.reader.read_coordinate_settings(list(selected.values()))
                hardware = {
                    name: {'id': motor_id, 'model_number': self.models[motor_id],
                           **settings[motor_id]}
                    for name, motor_id in selected.items()
                }
                self.state_converter = ArmStateConverter(
                    document, hardware, {'arm': self.bus, 'port': port},
                )
                self.state_publisher = self.create_publisher(
                    ArmState, 'state', qos_profile_sensor_data,
                )
                self.get_logger().info(
                    f'Calibration verified against hardware; publishing '
                    f'{self.state_publisher.topic_name}; '
                    f'calibration_id={self.state_converter.calibration_id}'
                )
                if group_mode:
                    for name in arm_names(self.bus):
                        entry = document['motors'][name]
                        self.joint_controls[name] = JointControl(
                            BoundJointDriver(self.reader, entry['id']), entry,
                            gripper_torque_limit_raw=gripper_torque_limit_raw,
                            report_temperature_recheck=(
                                lambda detail, name=name: self.get_logger().warning(
                                    f'{name}: {detail}', throttle_duration_sec=1.0)
                            ),
                        )
                    group = ArmBusHold(
                        self.bus, self.joint_controls, max_prepare_age_s=5.0,
                        write_group_goals=self.reader.write_arm_goals,
                        read_group_snapshot=self.reader.read_arm_control_snapshot,
                    )
                    self.get_logger().info('Grouped goal writes enabled for arm IDs 1..7')
                    self.get_logger().info('Absolute arm targets use the full calibrated range without shortening')
                    self.get_logger().info(
                        f'Continuous grippers use shared contact hold; torque limit '
                        f'{gripper_torque_limit_raw}/1000'
                    )
                    self.get_logger().info('Grouped preflight feedback enabled for arm IDs 1..7')
                    self.arm_transaction = ArmBusTransaction(group, timeout_s=5.0)
                    self.group_session_id = secrets.token_hex(16)
                    self.group_service = self.create_service(
                        ArmBusTransactionSrv, 'group/transaction', self.group_transaction,
                    )
                    self.joint_status_service = self.create_service(Trigger, 'joint/status', self.joint_status)
                    self.torque_status_service = self.create_service(Trigger, 'joint/torque_status', self.read_torque_status)
                    self.get_logger().info('Seven-joint group ready; waiting for an explicit hold request')
            if chassis_mode:
                self.chassis = ChassisRos(self, self.reader, teleop_link=teleop_link)
                self.get_logger().info('底盘/升降接口就绪；等待电脑遥操连接后自动准备')
            if group_mode or chassis_mode:
                self.install_protective_stop()
            self.publisher = self.create_publisher(
                MotorFeedbackArray, 'feedback', qos_profile_sensor_data,
            )
            self.timer = self.create_timer(1.0 / poll_hz, self.publish_feedback)
            self.get_logger().info(
                f'{self.bus} bus: verified {len(self.ids)} motors; '
                f'publishing {self.publisher.topic_name} at target {poll_hz:g} Hz'
            )
        except BaseException:
            self.destroy_node()
            raise

    def publish_feedback(self):
        started = time.monotonic()
        transport_error = ''
        try:
            replies = self.reader.read_feedback()
        except (OSError, ValueError) as exc:
            replies = {}
            transport_error = str(exc)
        message = MotorFeedbackArray()
        message.header.stamp = self.get_clock().now().to_msg()
        message.sequence = self._sequence
        self._sequence = (self._sequence + 1) % (2**32)
        message.read_duration_s = time.monotonic() - started
        failures = []
        for motor_id, name in zip(self.ids, self.names):
            item = MotorFeedback()
            item.name = name
            item.bus = self.bus
            item.motor_id = motor_id
            item.model_number = self.models[motor_id]
            item.header.stamp = message.header.stamp
            item.voltage_v = float('nan')
            item.temperature_c = float('nan')
            item.current_a = float('nan')
            reply = replies.get(motor_id)
            if reply is None:
                item.error = transport_error or 'No fresh reply within read_timeout_s'
            else:
                item.header.stamp = Time(nanoseconds=reply.stamp_ns).to_msg()
                item.servo_error = reply.servo_error
                if reply.servo_error:
                    item.error = f'Servo reports error 0x{reply.servo_error:02x}'
                else:
                    for field, field_value in decode_feedback(reply.data).items():
                        setattr(item, field, field_value)
                    item.valid = True
            if not item.valid:
                failures.append(f'{motor_id}: {item.error}')
            message.motors.append(item)
        self.publisher.publish(message)
        now = time.monotonic()
        if self.chassis is not None:
            self.chassis.observe(message)
        self.observe_joint_feedback(message)
        if self.state_converter is not None:
            state = self.state_converter.convert(message)
            self.state_publisher.publish(state)
            if self.recording_publisher.get_subscription_count():
                self.publish_recording_sample(state, message)
            if not state.complete and now >= self._next_state_warning:
                problems = '; '.join(
                    f'{name}: {error}' for name, error in zip(state.names, state.errors) if error
                )
                self.get_logger().warning(f'{self.bus} calibrated state invalid: {problems}')
                self._next_state_warning = now + 2.0
        if failures and now >= self._next_warning:
            self.get_logger().warning(f'{self.bus} feedback invalid: ' + '; '.join(failures))
            self._next_warning = now + 2.0
        self.last_feedback_cycle_duration_s = time.monotonic() - started

    def publish_recording_sample(self, state, feedback):
        """Expose measured state/ownership; actions have their own command events."""
        sample = {'stamp_ns': self.get_clock().now().nanoseconds, 'names': list(state.names),
                  'calibration_id': state.calibration_id, 'session': self.group_session_id,
                  'valid': False, 'reason': '等待完整反馈和全部关节的控制所有权'}
        if state.complete and all(
                name in self.joint_controls and self.joint_controls[name].owned
                and not self.joint_controls[name].fault for name in state.names):
            sample.update(valid=True, reason='', state=list(state.position_normalized),
                          state_stamps_ns=[s.sec * 1_000_000_000 + s.nanosec for s in state.sample_stamps],
                          driver_target_ticks=[self.joint_controls[name].target_ticks for name in state.names],
                          driver_targets=[float(normalize_tick(self.joint_controls[name].entry,
                                                               self.joint_controls[name].target_ticks,
                                                               feedback_tolerance_ticks=FEEDBACK_ENDPOINT_TOLERANCE_TICKS))
                                          for name in state.names],
                          gripper_contact_latched=[
                              self.joint_controls[name].gripper_contact_latched for name in state.names
                          ])
        elif state.complete:
            group = getattr(getattr(self, 'arm_transaction', None), 'group', None)
            problems = [f'{name}: {control.fault or "未持有上力控制权"}'
                        for name in state.names if (control := self.joint_controls.get(name)) is not None
                        and (control.fault or not control.owned)]
            sample['reason'] = (f'总线控制已停止：{group.reason}' if group is not None and group.state == 'fault'
                                else '; '.join(problems) or sample['reason'])
        if self.chassis is not None:
            sample['mobile'] = self.chassis.recording_snapshot(feedback)
        self.recording_publisher.publish(String(data=json.dumps(sample, separators=(',', ':'), allow_nan=False)))

    def observe_joint_feedback(self, message):
        controls = self.joint_controls.items()
        for joint_name, controller in list(controls):
            if not (controller.owned or controller.busy or controller.holding):
                continue
            item = next((item for item in message.motors if item.name == joint_name), None)
            sample = None
            if item is not None and item.valid and not item.servo_error and not item.error:
                sample = {key: getattr(item, key) for key in (
                    'position_ticks', 'velocity_raw', 'load_raw', 'current_a', 'temperature_c', 'moving',
                )}
                sample['voltage_v'] = getattr(item, 'voltage_v', float('nan'))
            event = controller.observe(sample)
            if event is not None:
                level, detail = event
                if level in ('error', 'warning'):
                    bus_snapshot = [{
                        'name': motor.name,
                        'valid': motor.valid,
                        'position_ticks': motor.position_ticks if motor.valid else None,
                        'velocity_raw': motor.velocity_raw if motor.valid else None,
                        'current_a': motor.current_a if motor.valid else None,
                        'voltage_v': motor.voltage_v if motor.valid else None,
                        'error': motor.error,
                    } for motor in message.motors if motor.motor_id <= 7]
                    detail += '; bus_snapshot=' + json.dumps(bus_snapshot, separators=(',', ':'), allow_nan=True)
                self.log_joint_event(level, detail, joint_name)

    def log_joint_event(self, level, detail, joint_name):
        # rclpy caches severity by call site; do not change it on the same line.
        message = f'{joint_name}: {detail}'
        if level == 'info':
            self.get_logger().info(message)
        elif level == 'error':
            self.get_logger().error(message)
        else:
            self.get_logger().warning(message)

    def joint_status(self, request, response):
        # Software ownership is not a direct hardware torque reading.
        response.success = True
        response.message = json.dumps({
            name: {
                'phase': control.phase,
                'owned': control.owned,
                'fault': control.fault,
                'last_message': control.last_message,
                'gripper_contact_latched': control.gripper_contact_latched,
                'gripper_torque_limit_raw': control.gripper_torque_limit_raw,
            }
            for name, control in self.joint_controls.items()
        }, ensure_ascii=False, sort_keys=True)
        return response

    def read_torque_status(self, request, response):
        """Read the actual RAM torque register for each arm motor; never write."""
        torques = {}
        failures = []
        for motor_id, name in enumerate(arm_names(self.bus), 1):
            try:
                torques[name] = BoundJointDriver(self.reader, motor_id).read_torque()
            except (OSError, ValueError, RuntimeError) as exc:
                torques[name] = None
                failures.append(f'{name}: {exc}')
        response.success = not failures
        response.message = json.dumps({'torques': torques, 'errors': failures}, ensure_ascii=False)
        return response

    def group_transaction(self, request, response):
        """Internal endpoint; startup ownership and streaming writes are explicit."""
        started = time.monotonic()
        self.last_transaction_kind = request.operation
        try:
            transaction = self.arm_transaction
            token = str(request.token)
            response.session_id = self.group_session_id
            if self.teleop_link is not None and request.operation in (
                    'prepare_hold', 'commit_hold', 'stream_targets'):
                self.teleop_link.require_active()
            if request.operation == 'inspect':
                pass
            elif self.motion_fault:
                raise ValueError(self.motion_fault)
            elif request.session_id != self.group_session_id:
                raise ValueError('Bus session changed; transaction rejected')
            elif request.operation == 'prepare_hold':
                transaction.prepare_hold(token)
            elif request.operation == 'commit_hold':
                transaction.commit_hold(token)
            elif request.operation == 'stream_targets':
                deadline = self.command_deadline(request.expires_at)
                transaction.apply_stream_targets(token, request.targets, deadline=deadline)
            elif request.operation == 'abort':
                transaction.abort(token)
            else:
                raise ValueError('Unknown group transaction operation')
            response.success = True
            response.message = 'completed'
        except (OSError, ValueError, RuntimeError) as exc:
            response.success = False
            response.message = str(exc)
        response.state = self.arm_transaction.group.state
        self.last_transaction_duration_s = time.monotonic() - started
        return response

    def command_deadline(self, stamp):
        expiry_ns = stamp.sec * 1_000_000_000 + stamp.nanosec
        started = time.monotonic()
        remaining = (expiry_ns - self.get_clock().now().nanoseconds) / 1e9
        if remaining <= 0:
            raise ValueError(TARGET_COMMAND_EXPIRED)
        if remaining > 2.0:
            raise ValueError('Target command has an invalid lifetime')
        return started + remaining

    def install_protective_stop(self):
        # A stop can latch during serial preflight. Only the default callback
        # group executes serial I/O, including the verified stop below.
        self.stop_guard = self.create_guard_condition(self.complete_protective_stop)
        self.stop_callbacks = MutuallyExclusiveCallbackGroup()
        self.create_service(ProtectiveStop, stop_service(self.bus), self.request_protective_stop,
                            callback_group=self.stop_callbacks)

    async def request_protective_stop(self, request, response):
        self.motion_fault = self.motion_fault or request.reason or 'Motion control fault'
        self.reader.motion_fault = self.motion_fault
        if self.chassis is not None:
            self.chassis.motion_fault = self.motion_fault
        completion = Future(executor=self.executor)
        self.stop_work = response, completion
        self.stop_guard.trigger()
        return await completion

    def complete_protective_stop(self):
        response, completion = self.stop_work
        completion.set_result(self.protective_stop(response))

    def protective_stop(self, response):
        if self.stop_confirmed:
            response.success, response.message = True, self.motion_fault
            return response
        failures = []
        if self.chassis is not None:
            try:
                self.chassis.protective_stop(self.motion_fault)
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append(f'base/lift: {exc}')
        arm_error = self.stop_arm(self.motion_fault)
        if arm_error:
            failures.append(arm_error)
        self.stop_confirmed = response.success = not failures
        response.message = '; '.join(failures) if failures else self.motion_fault
        if failures:
            self.get_logger().error(f'{self.bus} 保护停车未确认：{response.message}')
        return response

    def stop_arm(self, reason):
        if self.arm_transaction is None:
            return
        try:
            self.arm_transaction.protective_release(reason)
        except (OSError, ValueError, RuntimeError) as exc:
            return f'arm: {exc}'

    def destroy_node(self):
        try:
            try:
                if self.chassis is not None:
                    self.chassis.close()
            finally:
                if arm_error := self.stop_arm('Motor node shutdown'):
                    self.get_logger().error(f'{self.bus} 手臂保护停车未确认：{arm_error}')
        finally:
            try:
                if self.reader is not None:
                    self.reader.close()
            finally:
                destroyed = super().destroy_node()
        return destroyed


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        node = MotorStateNode()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            executor.shutdown()
            if node is not None:
                node.destroy_node()
        finally:
            rclpy.try_shutdown()


if __name__ == '__main__':
    main()
