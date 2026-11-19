# SPDX-License-Identifier: Apache-2.0
"""Real ROS/TCP replay wiring with fake buses, isolated from every physical driver."""

import json
import os
import socket
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger
from xyc_interfaces.msg import ArmState, MobileCommand

from xyc_hardware.arm_group_coordinator import ArmGroupCoordinator
from xyc.robot.calibration import denormalize_tick, normalize_tick
from xyc_hardware.chassis_control import CHASSIS_NOT_READY, COMMAND_QOS, COMMAND_RESULT_QOS
from xyc_hardware.motion_replay import MotionReplay
from xyc.contracts import ACTION_NAMES, ACTION_UNITS, STATE_NAMES, STATE_UNITS, physical_mobile_action
from xyc.transport.motion import MOTION_IDLE_S, PROTOCOL, receive, send
from test_arm_group_coordinator import FakeBus


class ReplayFakeBus(FakeBus):
    def __init__(self, side):
        self.feedback_available = True
        super().__init__(side)
        self.control_publisher = self.create_publisher(String, f'/motors/{side}/recording_sample', qos_profile_sensor_data)

    def publish_state(self):
        msg = ArmState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.arm = self.side
        msg.calibration_id = 'calibration' + self.side
        msg.names = list(self.arm_transaction.group.names)
        msg.position_normalized = [normalize_tick(control.entry, self.motors[name].position)
                                   for name, control in self.arm_transaction.group.controls.items()]
        msg.valid = [self.feedback_available] * 7
        msg.complete = self.feedback_available
        self.pub.publish(msg)
        valid = all(c.owned and not c.fault for c in self.arm_transaction.group.controls.values())
        self.control_publisher.publish(String(data=json.dumps({
            'calibration_id': 'calibration' + self.side, 'valid': valid, 'reason': '' if valid else 'not held',
            'stamp_ns': self.get_clock().now().nanoseconds,
            'driver_targets': [normalize_tick(c.entry, c.target_ticks) for c in self.arm_transaction.group.controls.values()]
                              if valid else [],
            'driver_target_ticks': [c.target_ticks for c in self.arm_transaction.group.controls.values()]
                                   if valid else []})))


class FakeChassis(Node):
    def __init__(self):
        super().__init__('fake_replay_chassis')
        self.enabled = False
        self.fault = ''
        self.stops = 0
        self.enable_wait_until = 0.
        self.refuse_enable_once = False
        self.enable_attempts = 0
        self.accepted = []
        self.lock = threading.Lock()
        self.status = self.create_publisher(String, '/robot/chassis/status', 1)
        self.commands = self.create_publisher(String, '/robot/chassis/recording_command', COMMAND_RESULT_QOS)
        self.subs = [self.create_subscription(MobileCommand, '/robot/mobile/command', self.on_velocity, COMMAND_QOS)]
        self.chassis_services = [self.create_service(Trigger, '/robot/chassis/' + name, callback)
                         for name, callback in [('enable', self.enable), ('stop', self.stop), ('home_lift', self.home)]]
        self.timer = self.create_timer(.03, self.publish_status)

    def publish_status(self):
        self.status.publish(String(data=json.dumps({'enabled': self.enabled, 'fault': self.fault,
                            'enable_ready': time.monotonic() >= self.enable_wait_until,
                            'enable_wait_reason': '等待连续0.15秒的完整底盘/升降静止反馈',
                            'feedback_paused': False, 'height_m': 0., 'base_measured': [0., 0., 0.], 'homing': ''})))

    def enable(self, _, response):
        self.enable_attempts += 1
        if self.refuse_enable_once:
            self.refuse_enable_once = False
            self.enable_wait_until = time.monotonic() + .18
            response.message = CHASSIS_NOT_READY + '等待连续0.15秒的完整底盘/升降静止反馈'
            return response
        self.enabled = response.success = True
        return response

    def stop(self, _, response):
        self.stops += 1
        response.success = True
        return response

    def protective_stop(self, reason):
        self.stops += 1
        self.accepted.append([0., 0., 0., 0.])
        self.fault = self.fault or reason

    def home(self, _, response):
        response.success = True
        return response

    def on_velocity(self, message):
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        with self.lock:
            if (self.fault or self.get_clock().now().nanoseconds >=
                    message.expires_at.sec * 10**9 + message.expires_at.nanosec):
                return
            values = physical_mobile_action(message.commands)
            self.accepted.append(values)
            targets = [int(value) for value in message.commands]
            self.commands.publish(String(data=json.dumps({'sequence': stamp, 'targets': targets, 'accepted': True})))


@unittest.skipUnless(os.environ.get('XYC_ISOLATED_TEST') == '1' and os.environ.get('ROS_DOMAIN_ID') == '183',
                     'requires isolated ROS domain 183; never run against robot nodes')
class TestReplayROS(unittest.TestCase):
    def test_source_owned_stream_acceptance_and_disconnect_stop(self):
        self.run_source('replay')

    def test_policy_uses_the_same_streaming_arm_and_mobile_controllers(self):
        self.run_source('policy')

    def test_policy_waits_for_post_hold_stationary_feedback(self):
        self.run_source('policy', wait_after_hold=True)

    def test_replay_disconnect_stops_both_arms_even_with_missing_feedback(self):
        self.run_source('replay', lose_arm_feedback=True)

    def test_policy_disconnect_stops_both_arms_even_with_missing_feedback(self):
        self.run_source('policy', lose_arm_feedback=True)

    def test_replay_disconnect_cancels_inflight_goal_before_second_bus_and_base(self):
        self.run_source('replay', failure='in-flight')

    def test_policy_disconnect_cancels_inflight_goal_before_second_bus_and_base(self):
        self.run_source('policy', failure='in-flight')

    def test_replay_stall_expires_even_with_readonly_status_traffic(self):
        self.run_source('replay', failure='stall')

    def test_policy_stall_expires_even_with_readonly_status_traffic(self):
        self.run_source('policy', failure='stall')

    def test_policy_failed_torque_off_reports_exact_joint_after_reconnect(self):
        self.run_source('policy', failure='stop-unconfirmed')

    def run_source(self, source, wait_after_hold=False, lose_arm_feedback=False, failure='chassis'):
        rclpy.init()
        nodes, thread, executor, connection = [], None, None, None
        replay = None
        try:
            left, right = ReplayFakeBus('left'), ReplayFakeBus('right')
            elbow_name = 'arm_left_elbow_flex'
            elbow = left.arm_transaction.group.controls[elbow_name]
            left.motors[elbow_name].position = denormalize_tick(elbow.entry, 97.80018)
            chassis = FakeChassis()
            left.chassis = chassis
            chassis.refuse_enable_once = wait_after_hold
            coordinator = ArmGroupCoordinator(parameter_overrides=[
                Parameter('command_source', value=source)])
            replay = MotionReplay(parameter_overrides=[Parameter('port', value=0),
                                                       Parameter('command_source', value=source)])
            nodes = [left, right, chassis, coordinator, replay]
            executor = MultiThreadedExecutor(num_threads=8)
            for node in nodes: executor.add_node(node)
            thread = threading.Thread(target=executor.spin, daemon=True)
            thread.start()
            deadline = time.monotonic() + 5.
            while not replay.snapshot()['ready'] and time.monotonic() < deadline: time.sleep(.02)
            self.assertTrue(replay.snapshot()['ready'])
            for client in replay.fault_request.clients.values():
                self.assertTrue(client.wait_for_service(timeout_sec=2.))
            connection = socket.create_connection(('127.0.0.1', replay.port), timeout=3.)
            connection.settimeout(4.)
            channel = connection.makefile('rwb')
            sequence = 0
            def request(operation, **values):
                nonlocal sequence
                if operation == 'target':
                    values.setdefault('valid_for_s', .25)
                sequence += 1
                send(channel, {'sequence': sequence, 'operation': operation, **values})

            def call(operation, **values):
                request(operation, **values)
                response = receive(channel)
                self.assertTrue(response['accepted'], response.get('message'))
                return response['snapshot']
            state = call('status')
            self.assertGreater(state['source_time_ns'], 0)
            self.assertEqual(set(state['observation_samples']), {'left', 'right'})
            self.assertGreater(state['observation_samples']['left']['stamp_ns'], 0)
            call('prepare', protocol=PROTOCOL, calibration_ids=state['calibration_ids'],
                 action_names=list(ACTION_NAMES), action_units=list(ACTION_UNITS),
                 state_names=list(STATE_NAMES), state_units=list(STATE_UNITS))
            call('start')
            self.assertEqual(chassis.enable_attempts, 2 if wait_after_hold else 1)
            deadline = time.monotonic() + 2.
            while not replay.snapshot()['chassis'].get('enabled') and time.monotonic() < deadline:
                call('heartbeat')
                time.sleep(.02)
            action = [0.] * 6 + [50.] + [0.] * 6 + [50.] + [1., 1., 0., 0.]
            action[2] = 99.35838
            action[5] = 80.
            action[6] = 100.
            call('target', action=action)
            self.assertIn('stream_targets', left.operations)
            self.assertIn('stream_targets', right.operations)
            self.assertIn([.015, .15, 0., 0.], chassis.accepted)
            self.assertIn(('goal', denormalize_tick(elbow.entry, 99.35838)), left.motors[elbow_name].writes)
            for name, target in (('arm_left_wrist_roll', 80.), ('arm_left_gripper', 100.)):
                control = left.arm_transaction.group.controls[name]
                self.assertIn(('goal', denormalize_tick(control.entry, target)), left.motors[name].writes)
            deadline = time.monotonic() + 2.
            while replay.snapshot()['driver_targets'][5:7] != [80., 100.] and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(replay.snapshot()['driver_targets'][5:7], [80., 100.])
            normal_stops = 0
            failed_motor = None
            if failure == 'in-flight':
                inspect = right.arm_transaction.group.read_group_snapshot
                in_preflight = threading.Event()
                writes = {name: len(motor.writes) for name, motor in right.motors.items()}

                def wait_for_stop():
                    in_preflight.set()
                    until = time.monotonic() + 1.
                    while not right.reader.motion_fault and time.monotonic() < until:
                        time.sleep(.002)
                    return inspect()

                right.arm_transaction.group.read_group_snapshot = wait_for_stop
                action[7], action[15] = 30., -1.
                request('target', action=action)
                self.assertTrue(in_preflight.wait(1.))
                connection.shutdown(socket.SHUT_RDWR)
            elif failure == 'stall':
                self.assertTrue(call('hold')['ready'])
                normal_stops = 1
                until = time.monotonic() + MOTION_IDLE_S + .08
                while time.monotonic() < until:
                    time.sleep(.06)
                    self.assertTrue(call('status')['ready'])
                call('target', action=action)
                until = time.monotonic() + MOTION_IDLE_S + 1.
                while True:
                    self.assertLess(time.monotonic(), until)
                    time.sleep(.06)
                    request('status')
                    rejected = receive(channel)
                    if not rejected['accepted']:
                        break
                self.assertTrue(rejected['stop_confirmed'], rejected['stop_error'])
                self.assertIn('loop timed out', rejected['message'])
            else:
                if failure == 'stop-unconfirmed':
                    failed_motor = right.motors['arm_right_shoulder_pan']
                    failed_motor.fail_release = True
                if lose_arm_feedback:
                    left.feedback_available = False
                else:
                    chassis.fault = 'ID 11 持续过流：5.23 A'
                deadline = time.monotonic() + 2.
                while replay.snapshot()['ready'] and time.monotonic() < deadline: time.sleep(.01)
                self.assertFalse(replay.snapshot()['ready'])
            if lose_arm_feedback:
                self.assertFalse(replay.snapshot()['arms_ready'])
            elif failure in ('chassis', 'stop-unconfirmed'):
                request('heartbeat')
                rejected = receive(channel)
                self.assertFalse(rejected['accepted'])
                self.assertEqual(rejected['stop_confirmed'], failed_motor is None)
                if failed_motor is not None:
                    self.assertIn('arm_right_shoulder_pan', rejected['stop_error'])
                self.assertIn('ID 11', rejected['message'])
                self.assertNotIn('calibration changed', rejected['message'])
            channel.close()
            connection.close()
            connection = None
            deadline = time.monotonic() + 3.
            while replay.connection is not None and time.monotonic() < deadline: time.sleep(.02)
            self.assertIsNone(replay.connection)
            self.assertEqual(chassis.stops, 1 + normal_stops)
            self.assertEqual(chassis.accepted[-1], [0., 0., 0., 0.])
            self.assertIsNone(replay.last_mobile)
            self.assertTrue(replay.motion_fault)
            for bus in (left, right):
                self.assertEqual(bus.stop_confirmed, failed_motor not in bus.motors.values())
                self.assertEqual(bus.arm_transaction.group.state, 'fault')
                self.assertTrue(all(motor.torque == 0 and motor.writes[-1] == ('torque', False)
                                    for motor in bus.motors.values() if motor is not failed_motor))
            if failure == 'in-flight':
                self.assertNotIn([.015, -.15, 0., 0.], chassis.accepted)
                for name, motor in right.motors.items():
                    self.assertEqual(motor.writes[writes[name]:], [('torque', False)])
            if failed_motor is not None:
                self.assertEqual(failed_motor.torque, 1)
                with socket.create_connection(('127.0.0.1', replay.port), timeout=1.) as probe:
                    with probe.makefile('rwb') as probe_channel:
                        send(probe_channel, {'sequence': 1, 'operation': 'status'})
                        observed = receive(probe_channel)['snapshot']
                        self.assertFalse(observed['stop_confirmed'])
                        self.assertIn('arm_right_shoulder_pan', observed['stop_error'])
            left.feedback_available = True
            self.assertFalse(replay.snapshot()['ready'])
        finally:
            if connection is not None: connection.close()
            if replay is not None: replay.close()
            if executor is not None: executor.shutdown()
            if thread is not None: thread.join(3.)
            for node in reversed(nodes): node.destroy_node()
            rclpy.shutdown()


class TestMobileEnableWait(unittest.TestCase):
    def test_only_prewrite_wait_rejections_are_retried(self):
        ready = {'ready': True, 'calibration_ids': ['left', 'right'], 'chassis': {'enable_ready': True}}
        trigger = Mock(side_effect=[RuntimeError(CHASSIS_NOT_READY + 'waiting'), None])
        node = SimpleNamespace(snapshot=lambda: ready, mobile_services={'enable': 'service'}, _trigger=trigger)
        MotionReplay._enable_mobile(node, ['left', 'right'])
        self.assertEqual(trigger.call_count, 2)

    def test_hardware_error_is_not_retried(self):
        ready = {'ready': True, 'calibration_ids': ['left', 'right'], 'chassis': {'enable_ready': True}}
        trigger = Mock(side_effect=RuntimeError('ID 8 register write outcome unknown'))
        node = SimpleNamespace(snapshot=lambda: ready, mobile_services={'enable': 'service'}, _trigger=trigger)
        with self.assertRaisesRegex(RuntimeError, 'ID 8'):
            MotionReplay._enable_mobile(node, ['left', 'right'])
        self.assertEqual(trigger.call_count, 1)


class TestTargetDeadline(unittest.TestCase):
    """Use the production backend methods without creating ROS nodes or motors."""

    def make_node(self, *, arm_delay=0., snapshot_delay=0., acceptance_delay=0., zero_verified=False):
        clock = [10.]
        generated = SimpleNamespace(sec=100, nanosec=0)
        node = SimpleNamespace(closing=threading.Event(), mobile_results={}, last_mobile=[0.] * 4)
        node.lock = threading.RLock()
        node.require_active_link = lambda: None
        node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: generated))
        def snapshot():
            clock[0] += snapshot_delay
            return {'ready': True, 'state': [0.] * 18,
                    'chassis': {'enabled': True, 'zero_targets_verified': zero_verified}}
        node.snapshot = snapshot
        def arms(*args, **kwargs): clock[0] += arm_delay
        node._set_arms = Mock(side_effect=arms)
        def publish(values, **kwargs):
            if not acceptance_delay:
                node.mobile_results[1] = {'targets': list(values), 'accepted': True}
            return 1
        node._publish_mobile = Mock(side_effect=publish)
        class Condition:
            def __enter__(self): pass
            def __exit__(self, *args): pass
            def wait(self, timeout):
                clock[0] += acceptance_delay
                node.mobile_results[1] = {'targets': [1, 1, 0, 0], 'accepted': True}
        node.condition = Condition()
        return node, clock, generated

    def test_expired_arm_confirmation_never_dispatches_mobile_targets(self):
        node, clock, _ = self.make_node(arm_delay=.4)
        with patch('xyc_hardware.motion_replay.time.monotonic', lambda: clock[0]):
            with self.assertRaisesRegex(TimeoutError, 'before base/lift dispatch'):
                MotionReplay.target(node, [0.] * 14 + [1., 1., 0., 0.], ['l', 'r'], valid_for_s=.25)
        node._publish_mobile.assert_not_called()
        self.assertEqual(node.last_mobile, [0.] * 4)
        self.assertEqual(node._set_arms.call_count, 1)

    def test_expiry_during_readiness_check_prevents_all_dispatch(self):
        node, clock, _ = self.make_node(snapshot_delay=.4)
        with patch('xyc_hardware.motion_replay.time.monotonic', lambda: clock[0]):
            with self.assertRaisesRegex(TimeoutError, 'before arm dispatch'):
                MotionReplay.target(node, [0.] * 18, ['l', 'r'], valid_for_s=.25)
        node._set_arms.assert_not_called()
        node._publish_mobile.assert_not_called()

    def test_live_target_preserves_generation_stamp_and_the_shared_deadline(self):
        node, clock, generated = self.make_node(arm_delay=.1)
        with patch('xyc_hardware.motion_replay.time.monotonic', lambda: clock[0]):
            MotionReplay.target(node, [0.] * 14 + [1., 1., 0., 0.], ['l', 'r'], valid_for_s=.25)
        self.assertEqual(node._set_arms.call_args.kwargs,
                         {'valid_for_s': .25, 'stamp': generated, 'deadline': 10.25})
        self.assertEqual(node._publish_mobile.call_args.kwargs, {'stamp': generated, 'expires_ns': 100_250_000_000})
        self.assertEqual(node.last_mobile, [1., 1., 0., 0.])

    def test_delayed_mobile_confirmation_cannot_restart_the_validity_window(self):
        node, clock, _ = self.make_node(arm_delay=.15, acceptance_delay=.15)
        with patch('xyc_hardware.motion_replay.time.monotonic', lambda: clock[0]):
            with self.assertRaisesRegex(TimeoutError, 'before base/lift confirmation'):
                MotionReplay.target(node, [0.] * 14 + [1., 1., 0., 0.], ['l', 'r'], valid_for_s=.25)
        self.assertIsNone(node.last_mobile)

    def test_verified_repeated_zero_skips_the_mobile_round_trip(self):
        node, clock, _ = self.make_node(zero_verified=True)
        node.last_mobile = None
        with patch('xyc_hardware.motion_replay.time.monotonic', lambda: clock[0]):
            for _ in range(2):
                MotionReplay.target(node, [0.] * 18, ['l', 'r'], valid_for_s=.25)
        self.assertEqual(node._set_arms.call_count, 2)
        node._publish_mobile.assert_called_once()
        self.assertEqual(node.last_mobile, [0.] * 4)

    def test_repeated_zero_with_unknown_driver_state_still_requires_confirmation(self):
        node, clock, _ = self.make_node(zero_verified=False)
        with patch('xyc_hardware.motion_replay.time.monotonic', lambda: clock[0]):
            MotionReplay.target(node, [0.] * 18, ['l', 'r'], valid_for_s=.25)
        node._publish_mobile.assert_called_once()

    def test_zero_after_nonzero_does_not_reuse_an_older_zero_status(self):
        node, clock, _ = self.make_node(zero_verified=True)
        node.last_mobile = [0., 1., 0., 0.]
        with patch('xyc_hardware.motion_replay.time.monotonic', lambda: clock[0]):
            MotionReplay.target(node, [0.] * 18, ['l', 'r'], valid_for_s=.25)
        node._publish_mobile.assert_called_once()
        self.assertEqual(node.last_mobile, [0.] * 4)

    def test_feedback_lost_during_arm_write_cannot_reuse_the_zero_result(self):
        node, clock, _ = self.make_node(zero_verified=True)
        node.snapshot = Mock(side_effect=[node.snapshot(),
                                          {'ready': False, 'reason': 'base feedback paused'}])
        with patch('xyc_hardware.motion_replay.time.monotonic', lambda: clock[0]):
            with self.assertRaisesRegex(RuntimeError, 'base feedback paused'):
                MotionReplay.target(node, [0.] * 18, ['l', 'r'], valid_for_s=.25)
        node._publish_mobile.assert_not_called()

    def test_explicit_mobile_rejection_reports_the_driver_reason(self):
        node, clock, _ = self.make_node()
        node._publish_mobile.side_effect = None
        node._publish_mobile.return_value = 1
        node.mobile_results[1] = {'targets': [0, 0, 0, 0], 'accepted': False,
                                  'error': 'Mobile command expired in the ROS queue'}
        with patch('xyc_hardware.motion_replay.time.monotonic', lambda: clock[0]):
            with self.assertRaisesRegex(RuntimeError, 'expired in the ROS queue'):
                MotionReplay.target(node, [0.] * 18, ['l', 'r'], valid_for_s=.25)
        self.assertEqual(node.mobile_results, {})

    def test_service_wait_cannot_dispatch_a_request_after_its_deadline(self):
        node, clock, _ = self.make_node()
        client = Mock()
        def wait(**kwargs):
            clock[0] += .4
            return True
        client.wait_for_service.side_effect = wait
        with patch('xyc_hardware.motion_replay.time.monotonic', lambda: clock[0]):
            with self.assertRaisesRegex(TimeoutError, 'before dispatch'):
                MotionReplay._call(node, client, object(), timeout=.6, deadline=10.25)
        client.call_async.assert_not_called()

    def test_late_ros_reply_is_not_treated_as_fresh_acceptance(self):
        node, clock, _ = self.make_node()
        client, future = Mock(), Mock()
        future.done.return_value = True
        def dispatch(request):
            clock[0] += .4
            return future
        client.call_async.side_effect = dispatch
        with patch('xyc_hardware.motion_replay.time.monotonic', lambda: clock[0]):
            with self.assertRaisesRegex(TimeoutError, 'outcome unknown'):
                MotionReplay._call(node, client, object(), timeout=.6, deadline=10.25)
        client.remove_pending_request.assert_called_once_with(future)
        future.result.assert_not_called()
