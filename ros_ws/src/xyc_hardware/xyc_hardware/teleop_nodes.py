# SPDX-License-Identifier: Apache-2.0
"""Relay leader feedback and coordinate its motion session."""

import json
from pathlib import Path
import select
import socket
import sys
import threading
import time
from uuid import uuid4

import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_srvs.srv import Trigger
from std_msgs.msg import String
from xyc_interfaces.msg import ArmState, LeaderState
from xyc_interfaces.srv import SetArmTargets
from xyc_hardware.motion_stop import MotionFaultStop

from xyc.robot.calibration import arm_names
from xyc.robot.profile import robot_profile
from xyc.teleop.mapping import FollowTargets, follower_snapshot, leader_snapshot, stamp_ns
from xyc.transport.teleop import (
    INPUT_LOSS_DEADLINE, MAX_SAMPLE_AGE, PERIOD, PORT, REPLY_DEADLINE, SIDES,
    MotionSessionEnded, TELEOP_TARGET_DROPPED,
    cadence, checked_positions, read_frame, send_frame,
)


LEADER_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
TARGET_ACK_WARN_S = .400
TARGET_ACK_STOP_S = 1.200
FOLLOWER_FEEDBACK_GRACE_S = .50


class LeaderLink:
    """Share TCP connection validity with producers of ROS motion commands."""

    def __init__(self):
        self.lock = threading.Lock()
        self.session = self.peer_events = None
        self.active = self.operator_exit = False
        self.stop_reason = self.stop_errors = self.on_fault = None

    def connect(self, session, connection):
        events = select.poll()
        events.register(connection, select.POLLRDHUP | select.POLLHUP | select.POLLERR)
        with self.lock:
            self.session, self.peer_events = session, events

    def disconnect(self, session, reason):
        with self.lock:
            if self.session == session:
                self.session = self.peer_events = None
                if self.active:
                    self._fail(reason)
                self.active = False

    def _fail(self, reason, *, operator_exit=False):
        if self.stop_reason is None:
            self.stop_reason, self.operator_exit = reason, operator_exit
            if self.on_fault is not None:
                self.on_fault()

    def fail(self, reason):
        with self.lock:
            self._fail(reason)

    def quit(self, session):
        with self.lock:
            if session == self.session:
                self._fail('操作员请求退出遥操', operator_exit=True)

    def _connected(self):
        if self.stop_reason is not None or self.peer_events is None:
            return False
        if self.peer_events.poll(0):
            if self.active:
                self._fail('主臂连接已断开')
            return False
        return True

    def claim(self, session):
        with self.lock:
            if self.session != session or not self._connected():
                return False
            self.active = True
            return True

    def accepts(self, session):
        with self.lock:
            return self.active and self.session == session and self._connected()

    def _require_active(self):
        if not self.active or not self._connected():
            raise MotionSessionEnded(self.stop_reason or '没有有效的主臂连接')

    def require_active(self):
        with self.lock:
            self._require_active()

    def dispatch(self, send, *args):
        with self.lock:
            self._require_active()
            # The ROS enqueue or serial write starts under the same gate as invalidation.
            return send(*args)

    def finish_stop(self, errors):
        with self.lock:
            self.stop_errors = errors

    def status(self):
        with self.lock:
            if self.active:
                self._connected()
            if self.stop_reason is not None:
                return {'reason': self.stop_reason, 'errors': self.stop_errors,
                        'operator_exit': self.operator_exit}


class LeaderStateBridge(Node):
    def __init__(self, link=None):
        super().__init__('leader_state_bridge')
        self.link = link
        self.period = PERIOD
        chassis = self.declare_parameter(
            'chassis_mode', False, ParameterDescriptor(read_only=True),
        ).value
        if chassis:
            from xyc_hardware.chassis_bridge import PCChassisBridge
            self.pc_chassis = PCChassisBridge(link=link)
        else:
            self.pc_chassis = None
        recording = self.declare_parameter(
            'recording', False, ParameterDescriptor(read_only=True),
        ).value
        if recording:
            from xyc_hardware.recording_bridge import RecordingBridge
            self.pc_recording = RecordingBridge()
        else:
            self.pc_recording = None
        self.outputs = {side: self.create_publisher(LeaderState, f'/teleop/{side}/leader_state',
                                                   LEADER_QOS) for side in SIDES}

    def publish_frame(self, session, sample_id, sample_ns, positions, calibration_ids):
        for side, values in positions.items():
            msg = LeaderState()
            msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(sample_ns, 1_000_000_000)
            msg.arm = side
            msg.session_id = session
            msg.sample_id = sample_id
            msg.calibration_id = calibration_ids[side]
            msg.names = list(arm_names(side))
            msg.position_normalized = [float(value) for value in values]
            self.outputs[side].publish(msg)

    def serve(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(('127.0.0.1', PORT))
            listener.listen(1)
            listener.settimeout(1.0)
            self.get_logger().info(f'等待 SSH 转发的主臂连接，端口 {PORT}，请求频率 {1 / self.period:g} Hz')
            while rclpy.ok():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                try:
                    with connection, connection.makefile('rwb') as channel:
                        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        connection.settimeout(REPLY_DEADLINE)
                        self.publish_session(channel, connection=connection)
                except (OSError, ValueError, EOFError) as exc:
                    self.get_logger().warning(f'主臂连接结束，继续监听端口 {PORT}：{exc}',
                                              throttle_duration_sec=1.0)

    def publish_session(self, channel, *, connection=None):
        session = uuid4().hex
        if self.link is not None:
            self.link.connect(session, connection)
        if self.pc_chassis is not None:
            self.pc_chassis.begin_session(session)
        if self.pc_recording is not None:
            self.pc_recording.begin_session(session)
        reason = '主臂连接已关闭'
        try:
            self._publish_session(channel, session)
        except (OSError, ValueError, EOFError) as exc:
            reason = f'主臂连接中断：{exc}'
            raise
        finally:
            if self.link is not None:
                self.link.disconnect(session, reason)
            if self.pc_chassis is not None:
                self.pc_chassis.disconnect(session)
            if self.pc_recording is not None:
                self.pc_recording.disconnect(session)

    def _publish_session(self, channel, session):
        last_sample = 0
        last_valid = time.monotonic()
        session_calibration = None
        for sequence, _ in enumerate(cadence(period=self.period), 1):
            if not rclpy.ok():
                return
            request = {'sequence': sequence, 'chassis_supported': self.pc_chassis is not None}
            if self.link is not None and (fault := self.link.status()) is not None:
                request['teleop_stop'] = fault
            if self.pc_chassis is not None:
                request.update(self.pc_chassis.snapshot())
            if self.pc_recording is not None:
                request.update(self.pc_recording.snapshot())
            # Snapshot locks can wait for ROS callbacks. Measure the reply
            # deadline from the actual send, not from local status preparation.
            started = time.monotonic()
            send_frame(channel, request)
            frame = read_frame(channel)
            received = time.monotonic()
            round_trip = received - started
            if frame['sequence'] != sequence:
                raise ValueError(f'主臂应答序号错误：期望 {sequence}，收到 {frame["sequence"]}')
            if round_trip > REPLY_DEADLINE:
                raise ValueError(f'主臂网络应答耗时 {round_trip:.3f}s，超过 {REPLY_DEADLINE:.3f}s')
            if self.link is not None:
                if frame.get('quit') is True:
                    self.link.quit(session)
                    if self.pc_recording is not None:
                        self.pc_recording.disconnect(session)
                if self.link.status() is not None:
                    continue  # Relay the latched diagnostic without taking any new commands.
            if self.pc_recording is not None:
                self.pc_recording.receive(session, frame.get('recording'))
            if 'chassis' in frame:
                if self.pc_chassis is None:
                    raise ValueError('请用 chassis_mode:=true 启动本地键盘桥接')
                if self.link is None or self.link.claim(session):
                    self.pc_chassis.receive_packet(session, frame['chassis'])
            if received - last_valid > INPUT_LOSS_DEADLINE:
                raise ValueError(f'主臂反馈连续异常：{frame.get("error", "反馈中断")}')
            if 'error' in frame:
                if not isinstance(frame['error'], str) or not frame['error']:
                    raise ValueError('主臂错误消息无效')
                self.get_logger().warning(f'主臂坏帧已跳过：{frame["error"]}', throttle_duration_sec=1.0)
                continue
            sample_id, age_ns = frame.get('sample_id'), frame.get('sample_age_ns')
            if type(sample_id) is not int or not max(1, last_sample) <= sample_id < 2**64:
                raise ValueError('主臂采样序号无效')
            if type(age_ns) is not int or age_ns < 0:
                raise ValueError('主臂采样时间无效')
            sample_age = age_ns + round(round_trip * 1e9)
            if sample_id == last_sample or sample_age > MAX_SAMPLE_AGE * 1e9:
                continue
            positions = checked_positions(frame)
            ids = frame.get('calibration_ids')
            if (not isinstance(ids, dict) or ids.keys() != positions.keys()
                    or any(not isinstance(value, str) or len(value) != 64
                           or any(c not in '0123456789abcdef' for c in value) for value in ids.values())):
                raise ValueError('主臂标定 ID 无效')
            if session_calibration is None:
                session_calibration = ids
                self.get_logger().info(f'主臂状态已上线：{", ".join(positions)}')
            elif ids != session_calibration:
                raise ValueError('主臂标定或侧别在连接期间变化')
            sample_ns = self.get_clock().now().nanoseconds - sample_age
            if sample_ns <= 0:
                continue
            if self.link is not None and not self.link.claim(session):
                continue
            self.publish_frame(session, sample_id, sample_ns, positions, ids)
            last_valid, last_sample = received, sample_id


class TeleopController(Node):
    def __init__(self, link=None):
        super().__init__('teleop_controller')
        fixed = ParameterDescriptor(read_only=True)
        side = self.declare_parameter('side', 'right', fixed).value
        self.live = self.declare_parameter('live', False, fixed).value
        self.with_motors = self.declare_parameter('with_motors', False, fixed).value
        self.link = link if self.live or self.with_motors else None
        self.fault_stop = MotionFaultStop(self) if self.live or self.with_motors else None
        self.stop_pending = False
        recording = self.declare_parameter('recording', False, fixed).value
        self.recording_active_at = None
        if recording:
            self.recording_subscription = self.create_subscription(
                String, '/robot/recording/status', self.on_recording_status, 1)
        follow_mode = self.declare_parameter('follow_mode', 'relative', fixed).value
        if follow_mode not in ('relative', 'absolute'):
            raise ValueError('follow_mode 应为 relative 或 absolute')
        self.absolute_targets = follow_mode == 'absolute'
        if side not in (*SIDES, 'both'):
            raise ValueError('side 应为 left、right 或 both')
        self.sides = SIDES if side == 'both' else (side,)
        self.stopped = False
        self.control_wake = self.create_guard_condition(self.tick) if self.live or self.link is not None else None
        self.leaders, self.followers = {}, {}
        for side in self.sides:
            self.create_subscription(LeaderState, f'/teleop/{side}/leader_state',
                                     lambda msg, side=side: self.on_leader(side, msg), LEADER_QOS)
        for side in SIDES:
            self.create_subscription(ArmState, f'/motors/{side}/state',
                                     lambda msg, side=side: self.followers.__setitem__(side, msg),
                                     qos_profile_sensor_data)
        self.client = self.create_client(SetArmTargets, '/robot/arms/set_targets') if self.live else None
        self.start_client = (
            self.create_client(Trigger, '/robot/arms/start')
            if self.live and self.with_motors else None
        )
        self.start_pending = None
        self.start_deadline = 0.0
        self.hold_confirmed_ns = 0
        self.source_identity = None
        self.last_seen = self.last_planned = 0
        self.last_input = self.last_report = time.monotonic()
        self.samples = self.acknowledged = 0
        self.planner = self.follower_ids = self.pending = self.last_sent = None
        self.last_sent_at = 0.0
        self.response_deadline = 0.0
        self.response_warn_at = 0.0
        self.follower_invalid_since = None
        # Live control wakes on data or a reply. This timer runs its watchdogs
        # and provides a bounded fallback if there are no wakeups.
        self.timer = self.create_timer(PERIOD, self.tick)
        if self.link is not None:
            self.link.on_fault = self.wake_control
        mode = '实机跟随' if self.live else '只读预览'
        self.get_logger().info(
            f'遥操控制就绪，模式 {mode}，跟随方式 {follow_mode}，等待 {", ".join(self.sides)} 主臂'
        )

    def wake_control(self, _reply=None):
        if self.control_wake is not None and not self.stopped:
            self.control_wake.trigger()

    def on_recording_status(self, message):
        try:
            active = json.loads(message.data).get('phase') in ('waiting', 'recording')
        except (ValueError, AttributeError):
            active = False
        self.recording_active_at = time.monotonic() if active else None

    def on_leader(self, side, msg):
        if self.stopped or (self.link is not None and not self.link.accepts(msg.session_id)):
            return
        self.leaders[side] = msg
        if self.pending is None:
            self.wake_control()

    def tick(self):
        if self.stopped:
            if self.stop_pending and (errors := self.fault_stop.poll()) is not None:
                self.stop_pending = False
                self.timer.cancel()
                if self.link is not None:
                    self.link.finish_stop(errors)
                if errors:
                    self.get_logger().error('遥操保护停车未确认：' + '; '.join(errors))
            return
        try:
            if self.link is not None and (fault := self.link.status()) is not None:
                raise ConnectionError(fault['reason'])
            self.update_targets()
        except (OSError, ValueError, RuntimeError) as exc:
            self.stopped = True
            self.leaders.clear()
            self.pending = self.start_pending = None
            if self.link is not None:
                self.link.fail(str(exc))
            reason = f'遥操已停止：{exc}'
            if self.fault_stop is not None:
                self.fault_stop.begin(reason)
                self.stop_pending = True
            else:
                self.timer.cancel()
                if self.link is not None:
                    self.link.finish_stop(())
            if self.link is not None and self.link.status()['operator_exit']:
                self.get_logger().info('操作员已退出遥操，后端已锁定本次控制会话。')
            else:
                self.get_logger().error(reason + '。请检查后重启遥操后端。')

    def start_managed_arms(self, now, sample_id):
        if self.start_pending is None:
            if not self.start_client.service_is_ready():
                self.get_logger().info('等待双臂上力服务', throttle_duration_sec=2.0)
                return
            self.start_pending = self.send(self.start_client, Trigger.Request())
            self.start_deadline = now + 12.0
            self.get_logger().info('主从臂反馈就绪，正在请求从臂原位上力')
            return
        if not self.start_pending.done():
            return
        response = self.start_pending.result()
        if response is None or not response.success:
            raise ValueError(response.message if response is not None else '双臂上力服务无响应')
        if not getattr(self.start_client, 'in_process', False):
            self.destroy_client(self.start_client)
        self.start_client = self.start_pending = None
        self.hold_confirmed_ns = self.get_clock().now().nanoseconds
        self.last_planned = sample_id
        self.get_logger().info('双臂已原位上力，等待新主从臂数据后开始跟随')

    def accept_leader_session(self, session, calibration_ids, sample_id, now):
        identity = (session, calibration_ids)
        if self.source_identity is None:
            self.source_identity = identity
        elif identity != self.source_identity or sample_id < self.last_seen:
            raise ValueError('主臂连接、标定或采样序号变化')
        if sample_id > self.last_seen:
            self.last_seen = sample_id
            self.last_input = now
            self.samples += 1

    def send(self, client, request):
        if self.link is not None:
            return self.link.dispatch(client.call_async, request)
        return client.call_async(request)

    def update_targets(self):
        now = time.monotonic()
        if (self.start_pending is not None and not self.start_pending.done()
                and now >= self.start_deadline):
            raise ValueError('双臂上力响应超时，实际扭矩状态待查')
        self.check_target_reply(now)

        if self.source_identity is not None and now - self.last_input > INPUT_LOSS_DEADLINE:
            raise TimeoutError('主臂持续断流')
        now_ns = self.get_clock().now().nanoseconds
        snapshot = leader_snapshot(self.leaders, self.sides, now_ns)
        if snapshot is None:
            return
        session, sample_id, calibration_ids, leader = snapshot
        self.accept_leader_session(session, calibration_ids, sample_id, now)
        if self.pending is not None or sample_id == self.last_planned:
            return

        try:
            names, feedback, follower_ids = follower_snapshot(self.followers, now_ns)
        except ValueError as exc:
            if self.planner is not None:
                if self.follower_invalid_since is None:
                    self.follower_invalid_since = now
                    self.get_logger().warning(
                        '从臂反馈暂时不完整，暂停下发新目标；已有目标继续由底层保持',
                        throttle_duration_sec=1.0,
                    )
                elif now - self.follower_invalid_since >= FOLLOWER_FEEDBACK_GRACE_S:
                    raise ValueError(
                        f'从臂反馈持续无效 {now - self.follower_invalid_since:.3f}s：{exc}'
                    ) from exc
                return
            self.follower_invalid_since = None
            self.get_logger().info('等待新鲜且完整的从臂反馈', throttle_duration_sec=2.0)
            return
        self.follower_invalid_since = None
        if self.client and not self.client.service_is_ready():
            if self.planner is not None:
                raise ValueError('14 维控制服务已断开')
            self.get_logger().info('等待 14 维控制服务', throttle_duration_sec=2.0)
            return
        if self.start_client is not None:
            self.start_managed_arms(now, sample_id)
            return
        if self.follower_ids is not None and self.follower_ids != follower_ids:
            raise ValueError('从臂标定在运行中发生变化')
        if self.planner is None:
            if any(stamp_ns(state) <= self.hold_confirmed_ns for state in self.followers.values()):
                return
            self.planner = FollowTargets(leader, feedback, absolute=self.absolute_targets)
            self.follower_ids = follower_ids
            self.get_logger().info(f'已记录两边起点，从臂当前位置 {[round(v, 2) for v in feedback]}')
        targets = self.planner.update(leader)
        self.last_planned = sample_id
        recording_at = getattr(self, 'recording_active_at', None)
        recording_cycle = recording_at is not None and now - recording_at <= 1.
        # A fresh operator frame still generates a command while stationary.
        # Keep normal no-record teleoperation's unchanged-target optimization.
        if self.client and (targets != self.last_sent or recording_cycle):
            request = SetArmTargets.Request()
            request.header.stamp = self.get_clock().now().to_msg()
            request.source_id = 'teleop'
            request.sequence = time.time_ns()
            request.valid_for.nanosec = 350_000_000
            request.arm_calibration_ids = follower_ids
            request.position_normalized = targets
            self.pending = self.send(self.client, request)
            sent_at = time.monotonic()
            self.response_warn_at = sent_at + TARGET_ACK_WARN_S
            self.response_deadline = sent_at + TARGET_ACK_STOP_S
            self.last_sent = targets
            self.last_sent_at = now
            if self.control_wake is not None:
                self.pending.add_done_callback(self.wake_control)
        if now - self.last_report >= 1.0:
            elapsed = now - self.last_report
            gaps = [abs(goal - actual) for goal, actual in zip(self.planner.reference, feedback)]
            worst = max(range(len(gaps)), key=gaps.__getitem__)
            self.get_logger().info(f'主臂新帧 {self.samples/elapsed:.1f} Hz，'
                                   f'控制确认 {self.acknowledged/elapsed:.1f} Hz，'
                                   f'最大跟随差 {names[worst]} {gaps[worst]:.1f}，'
                                   f'该关节本轮目标与反馈差 {abs(targets[worst] - feedback[worst]):.1f}，'
                                   f'目标 {[round(v, 2) for v in targets]}')
            self.last_report = now
            self.samples = self.acknowledged = 0

    def check_target_reply(self, now):
        if self.pending is None:
            return
        if self.pending.done():
            answer = self.pending.result()
            self.pending = None
            if answer is not None and answer.accepted:
                self.acknowledged += 1
            elif answer is not None and answer.message == TELEOP_TARGET_DROPPED:
                self.last_sent = None  # Replan from the latest inputs; never replay the old request.
                self.get_logger().warning('已丢弃过期且未下发的遥操目标，继续处理最新主臂帧',
                                          throttle_duration_sec=1.0)
            else:
                raise ValueError(answer.message if answer is not None else '控制服务无响应')
        elif now >= self.response_deadline:
            raise ValueError('控制回执持续超时，上一请求的执行状态待查')
        elif self.response_warn_at and now >= self.response_warn_at:
            self.response_warn_at = 0.0
            self.get_logger().warning('控制回执延迟，等待上一请求结果')


class LocalArmClient:
    """Use the same coordinator validation without a redundant local DDS hop."""
    in_process = True

    def __init__(self, coordinator, operation, response_type):
        self.coordinator, self.operation, self.response_type = coordinator, operation, response_type

    def service_is_ready(self):
        return True

    def call_async(self, request):
        return self.coordinator.transaction_workers.submit(
            self.operation, request, self.response_type())


def controller_components(link):
    controller = TeleopController(link)
    nodes = [controller]
    if controller.live:
        from xyc_hardware.arm_group_coordinator import ArmGroupCoordinator
        coordinator = ArmGroupCoordinator(teleop_link=link, use_global_arguments=False,
                                          parameter_overrides=[Parameter('command_source', value='teleop')])
        nodes.append(coordinator)
        controller.destroy_client(controller.client)
        controller.client = LocalArmClient(coordinator, coordinator.set_targets, SetArmTargets.Response)
        if controller.start_client is not None:
            controller.destroy_client(controller.start_client)
            controller.start_client = LocalArmClient(coordinator, coordinator.start, Trigger.Response)
    return nodes


def managed_motors(bridge, link):
    from xyc_hardware.motor_state_node import MotorStateNode, motor_launch_parameters

    calibration_dir = Path(bridge.declare_parameter(
        'calibration_dir', str(robot_profile().robot_root / 'config/calibration')).value).expanduser()
    gripper_limit = bridge.declare_parameter('gripper_torque_limit_raw', 100).value
    for bus in SIDES:
        parameters = motor_launch_parameters(
            bus, calibration_dir, calibrated=True, group_mode=True,
            chassis_mode=bridge.pc_chassis is not None, gripper_torque_limit_raw=gripper_limit,
        )
        yield MotorStateNode(teleop_link=link, namespace=f'/motors/{bus}', use_global_arguments=False,
                             parameter_overrides=[Parameter(name, value=value)
                                                  for name, value in parameters.items()])


def main(args=None):
    sys.setswitchinterval(.001)
    rclpy.init(args=args)
    nodes = []
    workers = []
    failures = []
    control_executor = SingleThreadedExecutor()

    def spin_ros(executor):
        try:
            executor.spin()
        except ExternalShutdownException:
            pass
        except Exception as exc:
            failures.append(exc)
            rclpy.try_shutdown()

    def start_executor(executor, name):
        worker = threading.Thread(target=spin_ros, args=(executor,), name=name, daemon=True)
        worker.start()
        workers.append((executor, worker))

    try:
        link = LeaderLink()
        nodes = controller_components(link)
        controller = nodes[0]
        bridge = LeaderStateBridge(controller.link)
        nodes.extend(node for node in (bridge, bridge.pc_chassis, bridge.pc_recording) if node is not None)
        if not controller.with_motors and (controller.live or bridge.pc_chassis is not None):
            raise ValueError('实机遥操需要 with_motors:=true，让电机节点共享断连检查')
        for node in nodes:
            control_executor.add_node(node)
        if controller.with_motors:
            for motor in managed_motors(bridge, link):
                nodes.append(motor)
                motor_executor = MultiThreadedExecutor(num_threads=2)
                motor_executor.add_node(motor)
                start_executor(motor_executor, f'teleop-{motor.bus}')
        start_executor(control_executor, 'teleop-ros')
        bridge.serve()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except (OSError, ValueError, EOFError) as exc:
        failures.append(exc)
    finally:
        for executor, _ in reversed(workers):
            executor.shutdown()
        for _, worker in workers:
            worker.join()
        for node in reversed(nodes):
            node.destroy_node()
        rclpy.try_shutdown()
    if failures:
        print(f'遥操后端已停止：{failures[0]}', file=sys.stderr, flush=True)
        return 1
    return 0
