# SPDX-License-Identifier: Apache-2.0
"""Execute a PC dataset stream through existing ROS arm and mobile drivers."""

import json
import math
import socket
import threading
import time

import rclpy
from geometry_msgs.msg import TwistStamped
from rcl_interfaces.srv import GetParameters
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger
from xyc_interfaces.msg import ArmState
from xyc_interfaces.srv import SetArmTargets

from xyc_hardware.chassis_control import COMMAND_QOS
from xyc_hardware.replay_transport import (
    ARM_NAMES, PORT, PROTOCOL, ReplaySession, physical_mobile_action, receive, send,
)


class MotionReplay(Node):
    def __init__(self, **kwargs):
        super().__init__('motion_replay', **kwargs)
        self.port = self.declare_parameter('port', PORT).value
        self.command_source = self.declare_parameter('command_source', 'replay').value
        if self.command_source not in ('replay', 'policy'):
            raise ValueError('command_source must be replay or policy')
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.arms, self.mobile = {}, None
        self.arm_control = {}
        self.accepted_mobile = {}
        self.last_mobile = [0., 0., 0., 0.]
        self.command_sequence = 0
        self.armed = False
        self.closing = threading.Event()
        self.connection = None
        self.targets = self.create_client(SetArmTargets, '/robot/arms/set_targets')
        self.start_arms = self.create_client(Trigger, '/robot/arms/start')
        self.owner = self.create_client(GetParameters, '/arm_group_coordinator/get_parameters')
        self.mobile_services = {name: self.create_client(Trigger, '/robot/chassis/' + name)
                                for name in ('enable', 'stop', 'home_lift')}
        self.base_pub = self.create_publisher(TwistStamped, '/robot/base/cmd_vel', COMMAND_QOS)
        self.lift_pub = self.create_publisher(TwistStamped, '/robot/lift/cmd_vel', COMMAND_QOS)
        self.state_subscriptions = [self.create_subscription(
            ArmState, f'/motors/{side}/state', lambda msg, side=side: self._arm(side, msg),
            qos_profile_sensor_data) for side in ('left', 'right')]
        self.state_subscriptions += [
            self.create_subscription(String, '/robot/chassis/status', self._mobile, 1),
            self.create_subscription(String, '/robot/chassis/recording_command', self._accepted, COMMAND_QOS)]
        self.state_subscriptions += [self.create_subscription(
            String, f'/motors/{side}/recording_sample', lambda msg, side=side: self._arm_control(side, msg),
            qos_profile_sensor_data) for side in ('left', 'right')]
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.listener.bind(('127.0.0.1', self.port))
            self.port = self.listener.getsockname()[1]
            self.listener.listen(1)
            self.listener.settimeout(.2)
        except BaseException:
            self.listener.close()
            raise
        self.worker = threading.Thread(target=self._serve, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f'Unified {self.command_source} control ready on loopback {self.port}; startup sends no commands')

    def _arm(self, side, message):
        with self.lock:
            self.arms[side] = time.monotonic(), message

    def _mobile(self, message):
        try:
            payload = json.loads(message.data)
        except ValueError:
            return
        with self.lock:
            self.mobile = time.monotonic(), payload

    def _arm_control(self, side, message):
        try:
            payload = json.loads(message.data)
        except ValueError:
            return
        with self.lock:
            self.arm_control[side] = time.monotonic(), payload

    def _accepted(self, message):
        try:
            event = json.loads(message.data)
            identity = event['sequence']
        except (ValueError, KeyError):
            return
        with self.condition:
            self.accepted_mobile[identity] = event
            while len(self.accepted_mobile) > 64:
                self.accepted_mobile.pop(next(iter(self.accepted_mobile)))
            self.condition.notify_all()

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            positions, ids, errors = [], [], []
            driver_targets, driver_target_ticks, driver_target_stamps = [], [], {}
            for index, side in enumerate(('left', 'right')):
                entry = self.arms.get(side)
                valid = (entry is not None and now - entry[0] <= .3 and entry[1].complete
                         and list(entry[1].names) == list(ARM_NAMES[index * 7:(index + 1) * 7])
                         and len(entry[1].position_normalized) == 7 and all(entry[1].valid)
                         and all(math.isfinite(v) for v in entry[1].position_normalized))
                if not valid:
                    errors.append(side + ' arm feedback unavailable')
                positions.extend(list(entry[1].position_normalized) if valid else [None] * 7)
                ids.append(entry[1].calibration_id if entry else '')
                control = self.arm_control.get(side)
                payload = control[1] if control else {}
                fresh_targets = control is not None and now - control[0] <= .3
                for output, key in ((driver_targets, 'driver_targets'),
                                    (driver_target_ticks, 'driver_target_ticks')):
                    values = payload.get(key)
                    output.extend(list(values) if fresh_targets and isinstance(values, list)
                                  and len(values) == 7 else [None] * 7)
                driver_target_stamps[side] = payload.get('stamp_ns') if fresh_targets else None
                if self.armed and (control is None or now - control[0] > .3 or not control[1].get('valid')
                                   or control[1].get('calibration_id') != ids[-1]):
                    errors.append(side + ' arm ownership/fault: ' + str(control[1].get('reason') if control else 'waiting'))
            arms_ready = not errors
            mobile = self.mobile[1] if self.mobile else {}
            if (not self.mobile or now - self.mobile[0] > .4 or mobile.get('fault')
                    or mobile.get('feedback_paused') or len(mobile.get('base_measured') or []) != 3):
                errors.append('base/lift feedback unavailable: ' + str(
                    mobile.get('fault') or mobile.get('reason', 'waiting')))
            height = mobile.get('height_m')
            velocity = mobile.get('base_measured') or [None] * 3
            return {'ready': not errors, 'arms_ready': arms_ready, 'reason': '; '.join(errors), 'calibration_ids': ids,
                    'state': [*positions, height, *velocity], 'chassis': mobile,
                    'driver_targets': driver_targets, 'driver_target_ticks': driver_target_ticks,
                    'driver_target_stamps_ns': driver_target_stamps,
                    'sample_received_monotonic_s': {
                        side: self.arms[side][0] if side in self.arms else None for side in ('left', 'right')},
                    'chassis_received_monotonic_s': self.mobile[0] if self.mobile else None,
                    'feedback_monotonic_ns': time.monotonic_ns()}

    def _call(self, client, request, timeout=3.):
        if not client.wait_for_service(timeout_sec=.5):
            raise RuntimeError('Required replay ROS service is unavailable')
        future = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not future.done():
            if self.closing.is_set() or time.monotonic() >= deadline:
                client.remove_pending_request(future)
                raise TimeoutError('ROS reply timed out; command outcome unknown, not retried')
            time.sleep(.001)
        result = future.result()
        if result is None:
            raise RuntimeError('Empty ROS reply')
        return result

    def _trigger(self, client):
        result = self._call(client, Trigger.Request())
        if not result.success:
            raise RuntimeError(result.message)

    def start(self, ids):
        parameters = GetParameters.Request()
        parameters.names = ['command_source']
        result = self._call(self.owner, parameters)
        if len(result.values) != 1 or result.values[0].string_value != self.command_source:
            raise ValueError(
                f'Start the {self.command_source} backend; the arm controller has a different owner')
        current = self.snapshot()
        if not current['ready'] or ids != current['calibration_ids']:
            raise ValueError('Robot feedback/calibration changed before starting')
        deadline = time.monotonic() + 1.5
        while (not current['chassis'].get('enabled')
               and current['chassis'].get('enable_ready') is False):
            if time.monotonic() >= deadline:
                raise ValueError(current['chassis'].get('enable_wait_reason') or 'Waiting for stationary feedback')
            time.sleep(.01)
            current = self.snapshot()
            if not current['ready']:
                raise ValueError(current['reason'])
        if self.armed:
            self._set_arms(current['state'][:14], ids)
        else:
            self._trigger(self.start_arms)
        self.armed = True
        self._trigger(self.mobile_services['enable'])
        self.last_mobile = [0., 0., 0., 0.]
        deadline = time.monotonic() + .8
        while not self.snapshot()['ready']:
            if time.monotonic() >= deadline:
                raise RuntimeError('Arm ownership/feedback was not confirmed after startup')
            time.sleep(.005)

    def home(self):
        self._trigger(self.mobile_services['home_lift'])
        self.last_mobile = [0., 0., 0., 0.]
        self.renew()

    def _set_arms(self, positions, ids):
        self.command_sequence += 1
        request = SetArmTargets.Request()
        request.header.stamp = self.get_clock().now().to_msg()
        request.source_id = self.command_source
        request.sequence = self.command_sequence
        request.valid_for.nanosec = 250_000_000
        request.arm_calibration_ids = list(ids)
        request.position_normalized = list(positions)
        reply = self._call(self.targets, request, timeout=.6)
        if not reply.accepted:
            raise RuntimeError('Arm replay target rejected: ' + reply.message)

    def _publish_mobile(self, values):
        stamp = self.get_clock().now().to_msg()
        lift, x, y, yaw = values
        base, vertical = TwistStamped(), TwistStamped()
        for message in (base, vertical):
            message.header.stamp = stamp
            message.header.frame_id = 'base_link'
        base.twist.linear.x, base.twist.linear.y, base.twist.angular.z = x, y, yaw
        vertical.twist.linear.z = lift
        self.base_pub.publish(base)
        self.lift_pub.publish(vertical)
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    def target(self, action, ids):
        status = self.snapshot()
        if (not status['ready'] or not status['chassis'].get('enabled')
                or status['chassis'].get('homing') or status['state'][14] is None):
            raise ValueError('Replay needs enabled base/lift, completed homing and fresh feedback')
        self._set_arms(action[:14], ids)
        mobile = physical_mobile_action(action[14:])
        stamp = self._publish_mobile(mobile)
        deadline = time.monotonic() + .2
        with self.condition:
            while stamp not in self.accepted_mobile:
                if self.closing.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError('Base/lift acceptance not confirmed; stopping replay')
                self.condition.wait(.002)
            event = self.accepted_mobile[stamp]
            if list(event['targets']) != list(mobile):
                raise RuntimeError('Base/lift accepted a different target')
        self.last_mobile = list(mobile)

    def renew(self):
        self._publish_mobile(self.last_mobile)

    def stop(self, *, freeze_arms=True):
        self.last_mobile = [0., 0., 0., 0.]
        self._publish_mobile(self.last_mobile)
        failures = []
        try:
            self._trigger(self.mobile_services['stop'])
        except Exception as exc:
            failures.append(str(exc))
        if freeze_arms and self.armed:
            state = self.snapshot()
            if state['arms_ready']:
                try:
                    self._set_arms(state['state'][:14], state['calibration_ids'])
                except Exception as exc:
                    failures.append(str(exc))
            else:
                failures.append('Arm freeze could not be confirmed: ' + state['reason'])
        if failures:
            raise RuntimeError('Replay stop could not be fully confirmed: ' + '; '.join(failures))

    def _serve(self):
        while not self.closing.is_set():
            try:
                connection, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            self.connection = connection
            session = ReplaySession(self)
            try:
                connection.settimeout(.35)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with connection, connection.makefile('rwb') as channel:
                    while not self.closing.is_set():
                        message = receive(channel)
                        try:
                            result = session.process(message)
                        except Exception as exc:
                            stopped, stop_error = True, ''
                            try:
                                session.disconnect()
                            except Exception as cleanup:
                                stopped, stop_error = False, str(cleanup)
                            send(channel, {'sequence': message['sequence'], 'protocol': PROTOCOL,
                                           'accepted': False, 'message': str(exc), 'snapshot': self.snapshot(),
                                           'stop_confirmed': stopped, 'stop_error': stop_error})
                            raise
                        send(channel, {'sequence': message['sequence'], 'protocol': PROTOCOL,
                                       'accepted': True, 'snapshot': result})
            except Exception as exc:
                self.get_logger().warning('Replay connection ended: ' + str(exc))
            finally:
                try:
                    session.disconnect()
                except Exception as exc:
                    self.get_logger().error(str(exc))
                connection.close()
                self.connection = None

    def close(self):
        self.closing.set()
        self.listener.close()
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.worker.join(timeout=5.)


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = SingleThreadedExecutor()
    try:
        node = MotionReplay()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()
