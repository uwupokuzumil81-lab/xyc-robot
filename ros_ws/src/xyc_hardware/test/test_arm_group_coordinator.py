# SPDX-License-Identifier: Apache-2.0
"""ROS integration tests with fake drivers, enabled only in a separate domain."""
import os
import socket
import threading
import time
import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

import rclpy
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Trigger
from std_msgs.msg import String
from xyc_interfaces.msg import ArmState
from xyc_interfaces.srv import ArmBusTransaction as ArmBusTransactionSrv, SetArmTargets
from xyc.robot.calibration import arm_names
from xyc_hardware.arm_group_coordinator import ArmGroupCoordinator
from xyc_hardware.arm_joint_control import ArmBusHold, ArmBusTransaction, JointControl
from xyc_hardware.motion_stop import stop_service
from xyc.contracts import CommandExpired
from xyc_hardware.motor_state_node import MotorStateNode
from xyc_hardware.teleop_nodes import LeaderLink, LeaderStateBridge, TeleopController, LocalArmClient, controller_components
from xyc.transport.teleop import INPUT_LOSS_DEADLINE, TELEOP_TARGET_DROPPED, read_frame, send_frame
from xyc.transport.lease import BackendLease, running_backend_pids
from test_arm_calibration import fixture
from test_joint_control import FakeJoint


class FakeBus(Node):
    group_transaction = MotorStateNode.group_transaction
    command_deadline = MotorStateNode.command_deadline
    protective_stop = MotorStateNode.protective_stop
    install_protective_stop = MotorStateNode.install_protective_stop
    request_protective_stop = MotorStateNode.request_protective_stop
    complete_protective_stop = MotorStateNode.complete_protective_stop
    stop_arm = MotorStateNode.stop_arm

    def __init__(self, side):
        super().__init__('fake_bus_' + side)
        self.side = side
        self.bus = side
        self.chassis = None
        self.motion_fault = ''
        self.teleop_link = None
        self.reader = SimpleNamespace(motion_fault='')
        self.stop_confirmed = False
        self.group_session_id = 'session' + side
        self.fail_target_commit = False
        self.operations = []
        self.fail_hold_commit = False
        self.delay_target_commit = 0.
        self.target_barrier = None
        doc = fixture(side)[2]
        self.motors = {}
        controls = {}
        for name in arm_names(side):
            entry = doc['motors'][name]
            motor = FakeJoint(entry)
            motor.bus = side
            self.motors[name] = motor
            controls[name] = JointControl(motor, entry)
        group = ArmBusHold(
            side, controls, max_prepare_age_s=5.,
            write_group_goals=self.write_arm_goals,
            read_group_snapshot=self.read_arm_control_snapshot,
        )
        self.arm_transaction = ArmBusTransaction(group, timeout_s=5.)
        self.service = self.create_service(ArmBusTransactionSrv, f'/motors/{side}/group/transaction', self.handle_transaction)
        self.install_protective_stop()
        self.pub = self.create_publisher(ArmState, f'/motors/{side}/state', qos_profile_sensor_data)
        self.timer = self.create_timer(.05, self.publish_state)

    def write_arm_goals(self, goals, *, deadline):
        if self.reader.motion_fault:
            raise CommandExpired(self.reader.motion_fault)
        for motor in self.motors.values():
            if motor.joint_id in goals:
                motor.write_goal(goals[motor.joint_id])

    def read_arm_control_snapshot(self):
        return {
            motor.joint_id: (motor.read_coordinate_settings([motor.joint_id])[motor.joint_id],
                             motor.read_torque(), motor.read_joint())
            for motor in self.motors.values()
        }

    def handle_transaction(self, request, response):
        self.operations.append(request.operation)
        if request.operation == 'stream_targets':
            if self.target_barrier is not None:
                self.target_barrier.wait(timeout=1.)
            if self.delay_target_commit:
                time.sleep(self.delay_target_commit)
        if ((request.operation == 'stream_targets' and self.fail_target_commit)
                or (request.operation == 'commit_hold' and self.fail_hold_commit)):
            response.success = False
            response.session_id = self.group_session_id
            response.state = self.arm_transaction.group.state
            response.message = 'injected failure'
            return response
        return self.group_transaction(request, response)

    def publish_state(self):
        msg = ArmState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.arm = self.side
        msg.calibration_id = 'calibration' + self.side
        msg.names = arm_names(self.side)
        msg.position_normalized = [0.] * 6 + [50.]
        msg.valid = [True] * 7
        msg.complete = True
        self.pub.publish(msg)


@unittest.skipUnless(os.environ.get('XYC_ISOLATED_TEST') == '1' and
                     os.environ.get('ROS_DOMAIN_ID') == '181', 'requires isolated ROS domain 181')
class TestArmGroupCoordinator(unittest.TestCase):
    def setUp(self):
        rclpy.init(args=['--ros-args', '-p', 'group_mode:=true', '-p', 'teleop_controller:live:=true'])
        self.left = FakeBus('left')
        self.right = FakeBus('right')
        self.coordinator = ArmGroupCoordinator()
        self.client = Node('fake_arm_test_client')
        self.start = self.client.create_client(Trigger, '/robot/arms/start')
        self.target = self.client.create_client(SetArmTargets, '/robot/arms/set_targets')
        self.executor = MultiThreadedExecutor(num_threads=6)
        for node in (self.left, self.right, self.coordinator, self.client):
            self.executor.add_node(node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        self.assertTrue(self.start.wait_for_service(timeout_sec=5.))
        self.assertTrue(self.target.wait_for_service(timeout_sec=5.))
        until = time.monotonic() + 5.
        while time.monotonic() < until and len(self.coordinator.states) < 2:
            time.sleep(.02)
        self.assertEqual(len(self.coordinator.states), 2)

    def tearDown(self):
        self.executor.shutdown()
        self.thread.join(timeout=3.)
        nodes = (self.client, self.coordinator, self.left, self.right)
        for node in nodes:
            node.destroy_node()
        rclpy.shutdown()

    def call(self, client, request):
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _: done.set())
        self.assertTrue(done.wait(12.), 'ROS service timed out')
        return future.result()

    def target_request(self):
        req = SetArmTargets.Request()
        req.header.stamp = self.client.get_clock().now().to_msg()
        req.source_id = 'teleop'
        req.sequence = 1
        req.valid_for.sec = 2
        req.arm_calibration_ids = ['calibrationleft', 'calibrationright']
        req.position_normalized = [1.] * 6 + [51.] + [1.] * 6 + [51.]
        return req

    def test_rejected_target_stops_both_arms_and_latches_their_old_session(self):
        buses = (self.left, self.right)
        controller = TeleopController()
        self.executor.add_node(controller)
        try:
            self.assertTrue(controller.live)
            for client in controller.fault_stop.clients.values():
                self.assertTrue(client.wait_for_service(timeout_sec=3.))
            self.assertTrue(self.call(self.start, Trigger.Request()).success)
            self.assertTrue(self.call(self.target, self.target_request()).accepted)
            joints = [joint for bus in buses for joint in bus.arm_transaction.group.controls.values()]
            self.assertTrue(all(joint.owned and joint.busy and joint.driver.torque == 1 for joint in joints))

            rejected = self.target_request()
            rejected.sequence = 2
            rejected.arm_calibration_ids[0] = 'changed'
            controller.response_deadline = time.monotonic() + 1.2
            controller.pending = controller.client.call_async(rejected)
            deadline = time.monotonic() + 3.
            while not all(bus.arm_transaction.group.state == 'fault' for bus in buses) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(controller.stopped)
            for bus in buses:
                self.assertEqual(bus.arm_transaction.group.state, 'fault')
                self.assertIn('calibration mismatch', bus.arm_transaction.group.reason)
            self.assertTrue(all(not joint.owned and not joint.busy and joint.driver.torque == 0 for joint in joints))
            self.assertTrue(all(joint.driver.writes[-1] == ('torque', False) for joint in joints))

            writes = [list(joint.driver.writes) for joint in joints]
            fresh = self.target_request()
            fresh.sequence = 3
            fresh.position_normalized = [10.] * 14
            self.assertFalse(self.call(self.target, fresh).accepted)
            for bus in buses:
                with self.assertRaisesRegex(ValueError, 'not idle'):
                    bus.arm_transaction.prepare_hold('100')
            self.assertEqual(writes, [joint.driver.writes for joint in joints])
        finally:
            self.executor.remove_node(controller)
            controller.destroy_node()

    def test_disconnect_rejects_queued_goals_and_relays_unconfirmed_joint_stop(self):
        link = LeaderLink()
        controller = TeleopController(link)
        bridge = LeaderStateBridge(link)
        controller.control_wake = None
        controller.timer.cancel()  # Leave the last DDS frame queued in the controller.
        self.coordinator.teleop_link = link
        self.executor.add_node(controller)
        finished = threading.Event()
        queued = threading.Event()
        release = threading.Event()
        entered = []
        worker_errors = []
        check_time = self.coordinator.check_command_time

        def pause_before_bus_dispatch(request):
            check_time(request)
            entered.append(request.sequence)
            if len(entered) == 2:
                queued.set()
                if not release.wait(2.):
                    raise TimeoutError('test did not release the arm transaction')

        def serve(connection):
            try:
                with connection, connection.makefile('rwb') as channel:
                    bridge.publish_session(channel, connection=connection)
            except EOFError:
                pass
            except Exception as exc:
                worker_errors.append(exc)
            finally:
                finished.set()

        def wait_for(predicate, timeout=3.):
            deadline = time.monotonic() + timeout
            while not predicate() and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertTrue(predicate())

        robot, pc = socket.socketpair()
        first = threading.Thread(target=serve, args=(robot,), daemon=True)
        try:
            wait_for(lambda: bridge.outputs['right'].get_subscription_count() > 0)
            first.start()
            with pc, pc.makefile('rwb') as channel:
                pc.settimeout(3.)
                request = read_frame(channel)
                send_frame(channel, {'sequence': request['sequence'], 'sample_id': 1,
                                     'sample_age_ns': 0, 'positions': {'right': [35.] * 7},
                                     'calibration_ids': {'right': 'a' * 64}})
                wait_for(lambda: 'right' in controller.leaders and len(controller.followers) == 2)
                self.assertTrue(self.call(self.start, Trigger.Request()).success)
                failed = self.right.motors['arm_right_shoulder_pan']
                failed.fail_release = True
                with patch.object(self.coordinator, 'check_command_time', side_effect=pause_before_bus_dispatch):
                    target = self.target.call_async(self.target_request())
                    self.assertTrue(queued.wait(2.))
                    pc.shutdown(socket.SHUT_WR)
                    self.assertTrue(finished.wait(2.))
                    release.set()
                    wait_for(target.done)
                    self.assertFalse(target.result().accepted)

            self.assertIn('right', controller.leaders)
            controller.tick()

            def stop_completed():
                controller.tick()
                return not controller.stop_pending

            wait_for(stop_completed)
            self.assertEqual(controller.leaders, {})
            self.assertTrue(controller.stopped)
            for bus in (self.left, self.right):
                self.assertNotIn('stream_targets', bus.operations)
                self.assertIn('主臂连接', bus.motion_fault)
            self.assertTrue(all(not motor.torque for bus in (self.left, self.right)
                                for motor in bus.motors.values() if motor is not failed))

            finished.clear()
            robot, pc = socket.socketpair()
            retry = threading.Thread(target=serve, args=(robot,), daemon=True)
            retry.start()
            with pc, pc.makefile('rwb') as channel:
                pc.settimeout(3.)
                setup = read_frame(channel)
                errors = setup['teleop_stop']['errors']
                self.assertTrue(any('arm_right_shoulder_pan' in error and 'torque-off not confirmed' in error
                                    for error in errors))
                send_frame(channel, {'sequence': setup['sequence'], 'sample_id': 1,
                                     'sample_age_ns': 0, 'positions': {'right': [40.] * 7},
                                     'calibration_ids': {'right': 'a' * 64}})
                self.assertEqual(read_frame(channel)['teleop_stop']['errors'], errors)
                pc.shutdown(socket.SHUT_WR)
            retry.join(3.)
            self.assertFalse(retry.is_alive())
            self.assertEqual(controller.leaders, {})
            self.assertEqual(worker_errors, [])
        finally:
            release.set()
            pc.close()
            if first.ident is None:
                robot.close()
            else:
                first.join(3.)
            self.executor.remove_node(controller)
            controller.destroy_node()
            bridge.destroy_node()

    def test_disconnect_drops_requests_already_in_both_motor_node_queues(self):
        link = LeaderLink()
        self.coordinator.teleop_link = self.left.teleop_link = self.right.teleop_link = link
        left_queued, right_queued, resume = (threading.Event() for _ in range(3))

        class BlockMotor:
            def __init__(self, queued):
                self.queued = queued

            def wait(self, timeout):
                self.queued.set()
                if not resume.wait(2.):
                    raise TimeoutError('Motor queue test timed out')

        robot, pc = socket.socketpair()
        with robot, pc:
            link.connect('operator', robot)
            self.assertTrue(link.claim('operator'))
            self.assertTrue(self.call(self.start, Trigger.Request()).success)
            self.left.target_barrier = BlockMotor(left_queued)
            self.right.target_barrier = BlockMotor(right_queued)
            target = self.target.call_async(self.target_request())
            try:
                self.assertTrue(left_queued.wait(2.) and right_queued.wait(2.))
                pc.shutdown(socket.SHUT_WR)
                # Motor owners deliberately have not gotten any stop RPC.
                self.assertFalse(self.left.motion_fault or self.right.motion_fault)
            finally:
                resume.set()
            deadline = time.monotonic() + 3.
            while not target.done() and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertTrue(target.done())
            self.assertFalse(target.result().accepted)
            self.assertIn('主臂连接', target.result().message)
            self.assertTrue(all(sum(kind == 'goal' for kind, _ in motor.writes) == 1
                                for bus in (self.left, self.right) for motor in bus.motors.values()))

    def test_quit_relays_confirmed_unified_stop_once_and_never_accepts_more_motion(self):
        self.assertTrue(self.call(self.start, Trigger.Request()).success)
        link = LeaderLink()
        controller = TeleopController(link)
        bridge = LeaderStateBridge(link)
        self.left.chassis = Mock()
        self.executor.add_node(controller)
        robot, pc = socket.socketpair()
        failure = []

        def serve():
            try:
                with robot, robot.makefile('rwb') as channel:
                    bridge.publish_session(channel, connection=robot)
            except EOFError:
                pass
            except Exception as exc:
                failure.append(exc)

        worker = threading.Thread(target=serve, daemon=True)
        worker.start()
        frame = {'sample_id': 1, 'sample_age_ns': 0, 'positions': {'right': [25.] * 7},
                 'calibration_ids': {'right': 'a' * 64}}
        try:
            with pc, pc.makefile('rwb') as channel:
                pc.settimeout(3.)
                request = read_frame(channel)
                send_frame(channel, {'sequence': request['sequence'], 'quit': True})
                for _ in range(200):
                    request = read_frame(channel)
                    status = request['teleop_stop']
                    self.assertTrue(status['operator_exit'])
                    if status['errors'] is not None:
                        self.assertEqual(status['errors'], [])
                        break
                    send_frame(channel, {'sequence': request['sequence'], 'quit': True, **frame})
                else:
                    self.fail('No response to operator quit')
                self.assertEqual(controller.leaders, {})
                self.assertTrue(self.left.stop_confirmed and self.right.stop_confirmed)
                self.assertTrue(all(not motor.torque for bus in (self.left, self.right)
                                    for motor in bus.motors.values()))
            worker.join(3.)
            self.assertFalse(worker.is_alive())
            self.assertEqual(failure, [])
            self.left.chassis.protective_stop.assert_called_once_with(
                f'遥操已停止：{controller.link.stop_reason}')
        finally:
            pc.close()
            worker.join(3.)
            self.executor.remove_node(controller)
            controller.destroy_node()
            bridge.destroy_node()

    def test_stalled_leader_latches_fault_and_rejects_next_arm_goal(self):
        controller = TeleopController()
        self.executor.add_node(controller)
        try:
            for client in controller.fault_stop.clients.values():
                self.assertTrue(client.wait_for_service(timeout_sec=3.))
            self.assertTrue(self.call(self.start, Trigger.Request()).success)
            self.assertTrue(self.call(self.target, self.target_request()).accepted)
            joints = [motor for bus in (self.left, self.right) for motor in bus.motors.values()]
            self.assertTrue(all(motor.torque == 1 for motor in joints))
            controller.last_input = time.monotonic() - INPUT_LOSS_DEADLINE - .01
            controller.source_identity = ('lost-leader', ('a', 'b'))
            deadline = time.monotonic() + 3.
            while not all(motor.torque == 0 for motor in joints) and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(controller.stopped)
            self.assertIn('主臂持续断流', self.left.motion_fault)
            self.assertTrue(all(motor.torque == 0 for motor in joints))
            next_target = self.target_request()
            next_target.sequence = 2
            self.assertFalse(self.call(self.target, next_target).accepted)
        finally:
            self.executor.remove_node(controller)
            controller.destroy_node()

    def test_recording_event_identifies_only_successfully_sent_commands(self):
        events = []
        subscriber = self.client.create_subscription(String, '/robot/arms/recording_command',
                                                      lambda msg: events.append(json.loads(msg.data)),
                                                      qos_profile_sensor_data)
        until = time.monotonic() + 3.
        while not self.coordinator.recording_publisher.get_subscription_count() and time.monotonic() < until:
            time.sleep(.01)
        self.assertTrue(self.call(self.start, Trigger.Request()).success)
        first = self.target_request()
        self.assertTrue(self.call(self.target, first).accepted)
        until = time.monotonic() + 2.
        while not events and time.monotonic() < until:
            time.sleep(.01)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event['targets'], list(first.position_normalized))
        self.assertEqual(event['sequence'], first.sequence)
        self.assertLessEqual(event['generated_ns'], event['sent_ns'])
        self.assertLessEqual(event['sent_ns'], event['accepted_ns'])
        # A second accepted request counts even if its numerical target is identical.
        second = self.target_request()
        second.sequence = 2
        self.assertTrue(self.call(self.target, second).accepted)
        until = time.monotonic() + 2.
        while len(events) < 2 and time.monotonic() < until:
            time.sleep(.01)
        self.assertEqual([e['sequence'] for e in events], [1, 2])
        self.right.fail_target_commit = True
        rejected = self.target_request()
        rejected.sequence = 3
        self.assertFalse(self.call(self.target, rejected).accepted)
        time.sleep(.05)
        self.assertEqual(len(events), 2)
        self.client.destroy_subscription(subscriber)

    def test_start_and_target_reach_both_fake_buses(self):
        self.assertTrue(all(not m.writes for m in (*self.left.motors.values(), *self.right.motors.values())))
        started = self.call(self.start, Trigger.Request())
        self.assertTrue(started.success, started.message)
        self.assertTrue(all(m.torque == 1 for m in (*self.left.motors.values(), *self.right.motors.values())))

        result = self.call(self.target, self.target_request())
        self.assertTrue(result.accepted, result.message)
        self.assertTrue(all(sum(k == 'goal' for k, _ in m.writes) == 2
                            for m in (*self.left.motors.values(), *self.right.motors.values())))

    def test_streaming_has_one_rpc_per_bus_and_retains_startup_handshake(self):
        self.coordinator.source = 'teleop'
        self.assertTrue(self.call(self.start, Trigger.Request()).success)
        for bus in (self.left, self.right):
            self.assertIn('prepare_hold', bus.operations)
            self.assertIn('commit_hold', bus.operations)
            bus.operations.clear()
        request = self.target_request()
        request.source_id = 'teleop'
        result = self.call(self.target, request)
        self.assertTrue(result.accepted, result.message)
        for bus in (self.left, self.right):
            self.assertEqual(bus.operations, ['stream_targets'])
            self.assertTrue(all(motor.torque == 1 for motor in bus.motors.values()))

    def test_streaming_partial_failure_releases_both_buses(self):
        self.coordinator.source = 'teleop'
        self.assertTrue(self.call(self.start, Trigger.Request()).success)
        self.right.fail_target_commit = True
        request = self.target_request()
        request.source_id = 'teleop'
        result = self.call(self.target, request)
        self.assertFalse(result.accepted)
        self.assertEqual(self.coordinator.phase, 'fault')
        self.assertTrue(all(motor.torque == 0 for bus in (self.left, self.right)
                            for motor in bus.motors.values()))

    def test_one_expired_stream_and_one_confirmed_success_replan_without_releasing(self):
        self.coordinator.source = 'teleop'
        self.assertTrue(self.call(self.start, Trigger.Request()).success)
        events = []
        sub = self.client.create_subscription(String, '/robot/arms/recording_command',
                    lambda msg: events.append(json.loads(msg.data)), qos_profile_sensor_data)
        end = time.monotonic() + 3.
        while not self.coordinator.recording_publisher.get_subscription_count() and time.monotonic() < end:
            time.sleep(.01)
        self.left.delay_target_commit = .28
        req = self.target_request()
        req.source_id = 'teleop'
        req.valid_for.sec, req.valid_for.nanosec = 0, 250_000_000
        result = self.call(self.target, req)
        self.assertEqual(result.message, TELEOP_TARGET_DROPPED)
        self.assertEqual(self.coordinator.phase, 'holding')
        self.assertTrue(all(motor.torque == 1 for bus in (self.left, self.right)
                            for motor in bus.motors.values()))
        self.assertFalse(events)
        self.left.delay_target_commit = 0.
        req = self.target_request()
        req.source_id, req.sequence = 'teleop', 2
        self.assertTrue(self.call(self.target, req).accepted)
        end = time.monotonic() + 2.
        while not events and time.monotonic() < end:time.sleep(.01)
        self.assertEqual([event['sequence'] for event in events], [2])
        self.client.destroy_subscription(sub)

    def test_local_client_uses_the_same_transaction_and_reports_failures(self):
        self.coordinator.source = 'teleop'
        starter = LocalArmClient(self.coordinator, self.coordinator.start, Trigger.Response)
        self.assertTrue(starter.call_async(Trigger.Request()).result(timeout=5.).success)
        client = LocalArmClient(self.coordinator, self.coordinator.set_targets, SetArmTargets.Response)
        request = self.target_request()
        request.source_id = 'teleop'
        self.assertTrue(client.call_async(request).result(timeout=5.).accepted)
        self.right.fail_target_commit = True
        request = self.target_request()
        request.source_id = 'teleop'
        request.sequence = 2
        self.assertFalse(client.call_async(request).result(timeout=5.).accepted)
        self.assertEqual(self.coordinator.phase, 'fault')
        self.assertTrue(all(motor.torque == 0 for bus in (self.left, self.right)
                            for motor in bus.motors.values()))



    def test_parallel_batches_release_both_buses_on_peer_inspection_failure(self):
        self.assertTrue(self.call(self.start, Trigger.Request()).success)
        self.left.target_barrier = self.right.target_barrier = threading.Barrier(2)
        self.left.motors['arm_left_shoulder_pan'].entry['hardware_homing_offset'] += 1
        rejected = self.call(self.target, self.target_request())
        self.assertFalse(rejected.accepted)
        self.assertIn('left: stream_targets rejected', rejected.message)
        self.assertEqual(self.coordinator.phase, 'fault')
        self.assertTrue(all(m.torque == 0 for bus in (self.left, self.right)
                            for m in bus.motors.values()))

    def test_wrong_source_and_calibration_do_not_write_goals(self):
        self.assertTrue(self.call(self.start, Trigger.Request()).success)
        req = self.target_request()
        req.source_id = 'other'
        self.assertFalse(self.call(self.target, req).accepted)
        req = self.target_request()
        req.arm_calibration_ids = ['wrong', 'calibrationright']
        self.assertFalse(self.call(self.target, req).accepted)
        self.assertTrue(all(sum(k == 'goal' for k, _ in m.writes) == 1
                            for m in (*self.left.motors.values(), *self.right.motors.values())))

    def test_repeated_sequence_is_rejected(self):
        self.assertTrue(self.call(self.start, Trigger.Request()).success)
        req = self.target_request()
        self.assertTrue(self.call(self.target, req).accepted)
        again = self.call(self.target, req)
        self.assertFalse(again.accepted)
        self.assertIn('sequence', again.message)

    def test_single_executor_keeps_60_hz_feedback_and_bus_replies_running(self):
        self.executor.remove_node(self.coordinator)
        coordinator_executor = SingleThreadedExecutor()
        coordinator_executor.add_node(self.coordinator)
        thread = threading.Thread(target=coordinator_executor.spin, daemon=True)
        thread.start()
        try:
            for bus in (self.left, self.right):
                bus.destroy_timer(bus.timer)
                bus.timer = bus.create_timer(1 / 60, bus.publish_state)
            self.coordinator.source = 'teleop'
            self.assertTrue(self.call(self.start, Trigger.Request()).success)
            self.left.delay_target_commit = self.right.delay_target_commit = .08
            began = time.monotonic()
            request = self.target_request()
            request.source_id = 'teleop'
            self.assertTrue(self.call(self.target, request).accepted)
            self.assertTrue(all(received > began + .02
                                for received, _ in self.coordinator.states.values()))
            self.left.delay_target_commit = self.right.delay_target_commit = 0.
            for sequence in range(2, 62):
                request = self.target_request()
                request.source_id = 'teleop'
                request.sequence = sequence
                request.valid_for.sec = 0
                request.valid_for.nanosec = 350_000_000
                response = self.call(self.target, request)
                self.assertTrue(response.accepted, response.message)
        finally:
            coordinator_executor.shutdown()
            thread.join(timeout=5.)
            coordinator_executor.remove_node(self.coordinator)
            self.executor.add_node(self.coordinator)

    def test_expired_policy_batches_do_not_release_or_move(self):
        self.coordinator.source = 'policy'
        self.assertTrue(self.call(self.start, Trigger.Request()).success)
        self.left.delay_target_commit = self.right.delay_target_commit = .28
        request = self.target_request()
        request.source_id = 'policy'
        request.valid_for.sec, request.valid_for.nanosec = 0, 250_000_000
        result = self.call(self.target, request)
        self.assertFalse(result.accepted)
        self.assertNotEqual(result.message, TELEOP_TARGET_DROPPED)
        self.assertEqual(self.coordinator.phase, 'holding')
        self.assertTrue(all(m.torque == 1 and sum(k == 'goal' for k, _ in m.writes) == 1
                            for bus in (self.left, self.right) for m in bus.motors.values()))

    def test_expired_teleop_batches_are_skipped_and_the_next_target_runs(self):
        self.coordinator.source = 'teleop'
        self.assertTrue(self.call(self.start, Trigger.Request()).success)
        self.left.delay_target_commit = self.right.delay_target_commit = .38
        request = self.target_request()
        request.source_id = 'teleop'
        request.valid_for.sec = 0
        request.valid_for.nanosec = 350_000_000
        result = self.call(self.target, request)
        self.assertFalse(result.accepted)
        self.assertEqual(result.message, TELEOP_TARGET_DROPPED)
        self.assertTrue(all(m.torque == 1 and sum(k == 'goal' for k, _ in m.writes) == 1
                            for m in (*self.left.motors.values(), *self.right.motors.values())))

        self.left.delay_target_commit = self.right.delay_target_commit = 0.
        request = self.target_request()
        request.source_id = 'teleop'
        request.sequence = 2
        result = self.call(self.target, request)
        self.assertTrue(result.accepted, result.message)

    def test_unconfirmed_stream_outcome_releases_both_buses(self):
        self.assertTrue(self.call(self.start, Trigger.Request()).success)
        self.right.delay_target_commit = .50
        request = self.target_request()
        request.valid_for.sec, request.valid_for.nanosec = 0, 250_000_000
        result = self.call(self.target, request)
        self.assertFalse(result.accepted)
        self.assertNotEqual(result.message, TELEOP_TARGET_DROPPED)
        self.assertEqual(self.coordinator.phase, 'fault')
        self.assertTrue(all(m.torque == 0 for bus in (self.left, self.right)
                            for m in bus.motors.values()))

    def test_one_bus_cannot_complete_hold_leaves_no_fake_motor_powered(self):
        self.right.fail_hold_commit = True
        result = self.call(self.start, Trigger.Request())
        self.assertFalse(result.success)
        self.assertEqual(self.coordinator.phase, 'fault')
        self.assertTrue(all(m.torque == 0 for m in (*self.left.motors.values(), *self.right.motors.values())))


    def test_optional_auto_hold_starts_once(self):
        class Timer:
            canceled = False
            def cancel(self):
                self.canceled = True
        timer = Timer()
        self.coordinator.auto_timer = timer
        until = time.monotonic() + 5.
        while time.monotonic() < until and not all(c.service_is_ready() for c in self.coordinator.bus_clients.values()):
            time.sleep(.02)
        task = self.executor.create_task(self.coordinator.auto_start())
        until = time.monotonic() + 5.
        while time.monotonic() < until and not task.done():
            time.sleep(.01)
        self.assertTrue(task.done())
        task.result()
        self.assertTrue(timer.canceled)
        self.assertEqual(self.coordinator.phase, 'holding')
        self.assertTrue(all(m.torque == 1 for m in (*self.left.motors.values(), *self.right.motors.values())))



@unittest.skipUnless(os.name == 'posix', 'Linux backend ownership')
class TestBackendLease(unittest.TestCase):
    def test_duplicate_launch_is_rejected_and_shutdown_releases_the_lock(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch('xyc.transport.lease.running_backend_pids', return_value=[]):
            first, second = BackendLease(directory), BackendLease(directory)
            options = dict(with_motors=True, with_cameras=True, recording=True)
            first.acquire(**options)
            try:
                with self.assertRaisesRegex(RuntimeError, '已在运行'):
                    second.acquire(**options)
            finally:first.close()
            second.acquire(**options)
            second.close()

    def test_existing_project_components_are_detected_before_any_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            project, proc = Path(directory) / 'project', Path(directory) / 'proc'
            install = project / 'ros_ws/install'
            expected = {
                111: install / 'xyc_hardware/lib/xyc_hardware/motor_state_node',
                222: install / 'xyc_hardware/lib/xyc_hardware/teleop_backend',
                333: Path(directory) / 'another/ros_ws/install/xyc_hardware/lib/xyc_hardware/teleop_backend',
            }
            for pid, path in expected.items():
                folder = proc / str(pid)
                folder.mkdir(parents=True)
                (folder / 'cmdline').write_bytes(b'python\x00' + str(path).encode() + b'\x00')
            options = dict(with_cameras=False, recording=False, proc_root=proc)
            self.assertEqual(running_backend_pids(project, with_motors=True, **options), [111, 222])
            self.assertEqual(running_backend_pids(project, with_motors=False, **options), [222])
            with patch('xyc.transport.lease.running_backend_pids', return_value=[222]):
                lease = BackendLease(project)
                with self.assertRaisesRegex(RuntimeError, 'PID=\\[222\\]'):
                    lease.acquire(with_motors=True, with_cameras=False, recording=False)
                self.assertIsNone(lease.handle)


class TestTeleopAcknowledgement(unittest.TestCase):
    def test_half_close_blocks_dispatch_before_socket_reader_observes_eof(self):
        link = LeaderLink()
        callback = Mock()
        link.on_fault = callback
        local, peer = socket.socketpair()
        with local, peer:
            link.connect('session', local)
            self.assertTrue(link.claim('session'))
            send = Mock()
            link.dispatch(send, 'connected')
            peer.shutdown(socket.SHUT_WR)
            with self.assertRaisesRegex(ConnectionError, '主臂连接已断开'):
                link.dispatch(send, 'stale')
            self.assertFalse(link.accepts('session'))
            send.assert_called_once_with('connected')
            callback.assert_called_once()

    def test_event_wakeups_do_not_restart_a_latched_fault(self):
        node = SimpleNamespace(control_wake=Mock(), pending=None, leaders={}, stopped=False,
                               update_targets=Mock(side_effect=RuntimeError('hardware fault')),
                               timer=Mock(), get_logger=Mock(), fault_stop=Mock(), stop_pending=False, link=None)
        node.fault_stop.poll.return_value = ()
        node.wake_control = lambda: TeleopController.wake_control(node)
        TeleopController.on_leader(node, 'left', 'first frame')
        node.control_wake.trigger.assert_called_once()
        TeleopController.tick(node)
        self.assertTrue(node.stopped)
        TeleopController.on_leader(node, 'left', 'next frame')
        TeleopController.tick(node)
        node.control_wake.trigger.assert_called_once()
        node.update_targets.assert_called_once()
        node.fault_stop.begin.assert_called_once()
        self.assertIn('hardware fault', node.fault_stop.begin.call_args.args[0])

    def test_slow_acknowledgement_waits_and_expired_preflight_replans(self):
        reply = Mock()
        reply.done.return_value = False
        node = SimpleNamespace(pending=reply, response_warn_at=.4, response_deadline=1.2,
                               last_sent=[1.] * 14, acknowledged=0, get_logger=Mock())
        TeleopController.check_target_reply(node, .41)
        self.assertIs(node.pending, reply)
        self.assertEqual(node.response_warn_at, 0.)
        reply.done.return_value = True
        reply.result.return_value = SimpleNamespace(accepted=False, message=TELEOP_TARGET_DROPPED)
        TeleopController.check_target_reply(node, .47)
        self.assertIsNone(node.pending)
        self.assertIsNone(node.last_sent)

        node.pending = reply
        reply.result.return_value = SimpleNamespace(accepted=True)
        TeleopController.check_target_reply(node, .51)
        self.assertEqual(node.acknowledged, 1)

        node.pending = reply
        reply.done.return_value = False
        with self.assertRaisesRegex(ValueError, '执行状态待查'):
            TeleopController.check_target_reply(node, 1.21)

    def test_new_leader_connection_cannot_reuse_an_active_control_session(self):
        calibration = ('cal-left', 'cal-right')
        node = SimpleNamespace(source_identity=('old', calibration), last_seen=42, last_input=1.0, samples=42)
        accept = TeleopController.accept_leader_session
        with self.assertRaisesRegex(ValueError, '主臂连接'):
            accept(node, 'new', calibration, 1, 2.0)
        self.assertEqual(node.last_input, 1.)
        accept(node, 'old', calibration, 42, 2.0)
        self.assertEqual(node.last_input, 1.)
        accept(node, 'old', calibration, 43, 3.0)
        self.assertEqual((node.last_input, node.last_seen), (3.0, 43))


@unittest.skipUnless(os.environ.get('XYC_ISOLATED_TEST') == '1' and
                     os.environ.get('ROS_DOMAIN_ID') == '181', 'requires isolated ROS domain 181')
class TestMotorGroupInitialization(unittest.TestCase):
    def test_live_controller_composes_distinct_nodes_without_starting_motors(self):
        rclpy.init(args=['--ros-args', '-p', 'live:=true', '-p', 'with_motors:=true'])
        nodes = []
        try:
            link = LeaderLink()
            nodes = controller_components(link)
            self.assertEqual([node.get_name() for node in nodes],
                             ['teleop_controller', 'arm_group_coordinator'])
            self.assertTrue(nodes[0].client.in_process)
            self.assertTrue(nodes[0].start_client.in_process)
            self.assertIs(nodes[0].link, nodes[1].teleop_link)
            self.assertEqual(nodes[1].phase, 'idle')
            self.assertEqual(nodes[1].source, 'teleop')
        finally:
            for node in reversed(nodes):node.destroy_node()
            rclpy.shutdown()

    def test_motor_node_initializes_seven_controls_without_writes(self):
        spec, _, doc = fixture('right')
        names = list(arm_names('right'))
        class FakeDriver:
            writes = []
            def __init__(self, port, ids, **kwargs):
                self.bus = kwargs['bus']
                self.joint_id = kwargs['joint_id']
            def open(self):
                pass
            def close(self):
                pass
            def read_models(self):
                return {doc['motors'][name]['id']: doc['motors'][name]['model_number'] for name in names}
            def read_coordinate_settings(self, ids):
                return {doc['motors'][name]['id']: {
                    'hardware_homing_offset': doc['motors'][name]['hardware_homing_offset'],
                    'operating_mode': doc['motors'][name]['operating_mode'],
                } for name in names if doc['motors'][name]['id'] in ids}
            def verify_speed_units(self):
                pass
            def motion_profile(self, motor_id):
                return (2400, 100) if motor_id == 7 else (2400, 120)
            def write_arm_goals(self, goals, *, deadline):
                FakeDriver.writes.append(('group_goals', goals))
            def read_arm_control_snapshot(self):
                return {}
        with tempfile.TemporaryDirectory() as temp:
            calibration = Path(temp) / 'right.json'
            calibration.write_text(json.dumps(doc))
            parameters = Path(temp) / 'parameters.yaml'
            parameters.write_text(yaml.safe_dump({'/**': {'ros__parameters': {
                'port': spec['port'], 'bus': 'right', 'calibration_file': str(calibration),
                'group_mode': True,
                'motor_ids': list(range(1, 8)), 'motor_names': names,
                'model_numbers': [doc['motors'][name]['model_number'] for name in names],
            }}}))
            rclpy.init(args=['--ros-args', '--params-file', str(parameters)])
            node = None
            try:
                with patch('xyc_hardware.motor_state_node.JointDriver', FakeDriver):
                    link = LeaderLink()
                    node = MotorStateNode(teleop_link=link)
                    self.assertEqual(node.reader.motion_gate, link.dispatch)
                    self.assertTrue(node.group_mode)
                    self.assertEqual(set(node.joint_controls), set(names))
                    self.assertEqual(node.arm_transaction.group.state, 'idle')
                    services = dict(node.get_service_names_and_types())
                    self.assertEqual(services[stop_service('right')], ['xyc_interfaces/srv/ProtectiveStop'])
                    self.assertEqual(FakeDriver.writes, [])
            finally:
                if node is not None:
                    node.destroy_node()
                rclpy.shutdown()
