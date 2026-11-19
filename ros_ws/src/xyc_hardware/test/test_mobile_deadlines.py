# SPDX-License-Identifier: Apache-2.0
"""Command deadlines at ROS receipt, controller execution and the serial boundary."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from builtin_interfaces.msg import Time

from xyc.contracts import CommandExpired
from xyc_hardware.chassis_bridge import PCChassisBridge
from xyc_hardware.chassis_control import ChassisDriver, ChassisRos, mobile_message
from xyc_hardware.motion_replay import MotionReplay
from xyc_hardware.arm_joint_control import JointDriver
from xyc_hardware.lift_tracking import LiftTracker
from test_chassis_control import make_control


class TestMobileDeadlines(unittest.TestCase):
    def make_ros(self):
        control, driver, clock, samples = make_control()
        control.enable()
        control.tracker = LiftTracker(control.lift_geometry, reference_id='test',
                                      zero_tick=2000, reference_time_s=clock[0])
        control.height = 0.
        def now():
            ns = round(clock[0] * 1e9)
            return SimpleNamespace(nanoseconds=ns, to_msg=lambda: Time(sec=ns // 10**9, nanosec=ns % 10**9))
        node = SimpleNamespace(get_clock=lambda: SimpleNamespace(now=now), get_logger=Mock(),
                               group_session_id='test-left')
        ros = object.__new__(ChassisRos)
        ros.teleop_link = None
        ros.node, ros.control = node, control
        ros.pending_command, ros.latest_command_ns = None, 0
        ros.motion_fault = ''
        ros.recording_command_publisher = Mock()
        return ros, driver, clock, samples

    def result(self, ros):
        return json.loads(ros.recording_command_publisher.publish.call_args.args[0].data)

    def test_verified_zero_does_not_wait_for_a_feedback_poll_or_rewrite_speed(self):
        ros, driver, clock, samples = self.make_ros()
        command = mobile_message(ros.node, [0, 0, 0, 0], expires_ns=1_050_000_000)
        writes = len(driver.writes)
        clock[0] = 1.048  # The next feedback tick would be too late.
        ros.mobile_command(command)
        event = self.result(ros)
        self.assertTrue(event['accepted'])
        self.assertEqual(event['targets'], [0, 0, 0, 0])
        self.assertEqual(event['accepted_ns'], 1_048_000_000)
        self.assertEqual(len(driver.writes), writes)
        self.assertIsNone(ros.pending_command)
        clock[0] = 1.066
        ros.control.observe(samples)
        ros.record_mobile_command()
        ros.recording_command_publisher.publish.assert_called_once()

    def test_zero_after_motion_waits_for_a_verified_stop(self):
        ros, driver, clock, samples = self.make_ros()
        ros.mobile_command(mobile_message(ros.node, [0, 1, 0, 0]))
        clock[0] = 1.01
        ros.control.observe(samples)
        ros.record_mobile_command()
        self.assertTrue(any(driver.writes[-1].values()))
        ros.recording_command_publisher.publish.reset_mock()
        clock[0] = 1.02
        ros.mobile_command(mobile_message(ros.node, [0, 0, 0, 0]))
        ros.recording_command_publisher.publish.assert_not_called()
        clock[0] = 1.03
        ros.control.observe(samples)
        ros.record_mobile_command()
        self.assertFalse(any(driver.writes[-1].values()))
        self.assertTrue(self.result(ros)['accepted'])

    def test_zero_after_motion_with_failed_serial_readback_is_rejected(self):
        ros, driver, clock, samples = self.make_ros()
        ros.mobile_command(mobile_message(ros.node, [0, 1, 0, 0]))
        clock[0] = 1.01
        ros.control.observe(samples)
        ros.record_mobile_command()
        driver.fail_write = True
        clock[0] = 1.02
        ros.mobile_command(mobile_message(ros.node, [0, 0, 0, 0]))
        clock[0] = 1.03
        ros.control.observe(samples)
        ros.record_mobile_command()
        event = self.result(ros)
        self.assertFalse(event['accepted'])
        self.assertIn('velocity readback failure', event['error'])

    def test_protective_stop_cancels_active_homing_and_refuses_new_homing(self):
        from std_srvs.srv import Trigger
        ros, driver, clock, samples = self.make_ros()
        ros.control.begin_home()
        clock[0] += .01
        ros.control.observe(samples)
        self.assertNotEqual(ros.control.sent[11], 0)

        ros.protective_stop('right arm write failed')
        self.assertFalse(ros.control.homing)
        self.assertEqual(driver.writes[-1], dict.fromkeys(driver.velocity_owned, 0))
        self.assertIsNone(ros.control.height)

        for operation in (ros.home, ros.enable):
            response = operation(Trigger.Request(), Trigger.Response())
            self.assertFalse(response.success)
            self.assertIn('right arm write failed', response.message)
        ros.mobile_command(mobile_message(ros.node, [1, 1, 0, 0]))
        clock[0] += .01
        ros.control.observe(samples)
        self.assertTrue(all(value == 0 for value in ros.control.sent.values()))

    def test_fifty_ms_target_processed_at_eighty_ms_never_reaches_motors(self):
        ros, driver, clock, samples = self.make_ros()
        command = mobile_message(ros.node, [1, 1, 0, 0], expires_ns=1_050_000_000)
        clock[0] = 1.08
        ros.mobile_command(command)
        ros.control.observe(samples)
        ros.record_mobile_command()
        self.assertTrue(all(not any(row.values()) for row in driver.writes))
        self.assertEqual(ros.control.base, (0., 0., 0.))
        self.assertFalse(self.result(ros)['accepted'])
        self.assertIn('expired in the ROS queue', self.result(ros)['error'])
        self.assertIn('expired in the ROS queue', ros.control.reason)

    def test_accepted_command_expiring_before_feedback_tick_is_not_written_or_acknowledged(self):
        ros, driver, clock, samples = self.make_ros()
        command = mobile_message(ros.node, [1, 1, 0, 0], expires_ns=1_050_000_000)
        clock[0] = 1.02
        ros.mobile_command(command)
        clock[0] = 1.08
        ros.control.observe(samples)
        ros.record_mobile_command()
        self.assertTrue(all(not any(row.values()) for row in driver.writes))
        self.assertFalse(self.result(ros)['accepted'])
        self.assertIn('expired before velocity confirmation', self.result(ros)['error'])

    def test_late_serial_confirmation_stops_immediately_and_cannot_acknowledge(self):
        ros, driver, clock, samples = self.make_ros()
        command = mobile_message(ros.node, [1, 1, 0, 0], expires_ns=1_050_000_000)
        ros.mobile_command(command)
        original = driver.write_velocities
        def delayed(values, **kwargs):
            original(values, **kwargs)
            if any(values.values()):
                clock[0] = 1.08
        driver.write_velocities = delayed
        clock[0] = 1.02
        ros.control.observe(samples)
        ros.record_mobile_command()
        self.assertFalse(any(driver.writes[-1].values()))
        self.assertFalse(ros.control.fault)  # Ordinary expiry is a stop, not an electrical fault.
        self.assertFalse(self.result(ros)['accepted'])

    def test_expiry_during_lift_enable_cannot_apply_partial_base_motion(self):
        ros, driver, clock, samples = self.make_ros()
        def enable():
            driver.velocity_owned.add(11)
            clock[0] = 1.08
        driver.enable_lift = enable
        ros.mobile_command(mobile_message(ros.node, [1, 1, 0, 0], expires_ns=1_050_000_000))
        ros.control.observe(samples)
        self.assertEqual(ros.control.base, (0., 0., 0.))
        self.assertEqual(ros.control.lift_speed, 0.)
        self.assertTrue(all(not any(row.values()) for row in driver.writes))

    def test_stop_or_home_supersedes_any_unapplied_mobile_command(self):
        for operation in ('stop', 'home'):
            with self.subTest(operation=operation):
                ros, _, _, samples = self.make_ros()
                ros.mobile_command(mobile_message(ros.node, [0, 1, 0, 0]))
                getattr(ros, operation)(None, SimpleNamespace(success=False, message=''))
                self.assertIsNone(ros.pending_command)
                ros.control.observe(samples)
                ros.record_mobile_command()
                ros.recording_command_publisher.publish.assert_not_called()
                self.assertEqual(ros.control.base, (0., 0., 0.))

    def test_queued_command_cannot_restart_velocity_after_stop_acknowledgment(self):
        ros, driver, clock, samples = self.make_ros()
        queued = mobile_message(ros.node, [0, 1, 0, 0], expires_ns=1_050_000_000)
        clock[0] = 1.02
        reply = ros.stop(None, SimpleNamespace(success=False, message=''))
        self.assertTrue(reply.success)
        clock[0] = 1.03
        ros.mobile_command(queued)
        ros.control.observe(samples)
        self.assertTrue(all(not any(row.values()) for row in driver.writes))
        self.assertIsNone(ros.pending_command)

        ros.mobile_command(mobile_message(ros.node, [0, -1, 0, 0]))
        ros.mobile_command(queued)  # DDS can also deliver an old packet behind a new one.
        clock[0] = 1.04
        ros.control.observe(samples)
        self.assertTrue(any(driver.writes[-1].values()))
        self.assertEqual(ros.control.base, (-.15, 0., 0.))

    def test_serial_writer_checks_deadline_after_encoding_before_first_packet(self):
        driver = object.__new__(ChassisDriver)
        driver._send_joint_packet, driver._receive = Mock(), Mock()
        clock = [1.02]
        from xyc_hardware.chassis_control import integer
        def encode(*args):
            value = integer(*args)
            clock[0] = 1.08
            return value
        with patch('xyc_hardware.chassis_control.integer', encode), \
                patch('xyc_hardware.chassis_control.time.monotonic', lambda: clock[0]):
            with self.assertRaisesRegex(CommandExpired, 'serial dispatch'):
                driver.write_velocities({8: 100}, deadline=1.05)
        driver._send_joint_packet.assert_not_called()
        driver._receive.assert_not_called()

    def test_disconnected_link_blocks_serial_motion_after_buffer_preparation(self):
        import socket
        from xyc_hardware.teleop_nodes import LeaderLink
        from xyc.transport.teleop import MotionSessionEnded

        for writer, values in ((JointDriver.write_arm_goals, {1: 2000}),
                               (ChassisDriver.write_velocities, {8: 100, 11: 100})):
            with self.subTest(writer=writer.__qualname__):
                driver = object.__new__(ChassisDriver)
                driver.port, driver.bus = Mock(), 'left'
                driver.port.write.side_effect = len
                link = LeaderLink()
                driver.motion_gate = link.dispatch
                robot, pc = socket.socketpair()
                with robot, pc:
                    link.connect('operator', robot)
                    self.assertTrue(link.claim('operator'))
                    driver.port.reset_input_buffer.side_effect = lambda: pc.shutdown(socket.SHUT_WR)
                    with self.assertRaises(MotionSessionEnded):
                        writer(driver, values)
                    driver.port.write.assert_not_called()
                    driver.port.reset_input_buffer.side_effect = None
                    driver._send_joint_packet(1, 3, b'\x28\x00')  # Protective torque-off stays available.
                    driver.port.write.assert_called_once()

    def test_arm_and_mobile_deadlines_are_checked_after_serial_buffer_preparation(self):
        for writer, values in ((JointDriver.write_arm_goals, {1: 2000}),
                               (ChassisDriver.write_velocities, {8: 100})):
            with self.subTest(writer=writer.__qualname__):
                driver = object.__new__(ChassisDriver)
                clock = [1.02]
                driver.port, driver.bus, driver._receive = Mock(), 'left', Mock()
                def delayed_reset(): clock[0] = 1.08
                driver.port.reset_input_buffer.side_effect = delayed_reset
                with patch('xyc_hardware.arm_joint_control.time.monotonic', lambda: clock[0]):
                    with self.assertRaisesRegex(CommandExpired, 'serial port write'):
                        writer(driver, values, deadline=1.05)
                driver.port.write.assert_not_called()
                driver._receive.assert_not_called()

    def test_fault_latched_during_serial_preparation_blocks_arm_and_mobile_writes(self):
        for writer, values in ((JointDriver.write_arm_goals, {1: 2000}),
                               (ChassisDriver.write_velocities, {8: 100})):
            with self.subTest(writer=writer.__qualname__):
                driver = object.__new__(ChassisDriver)
                driver.port, driver.bus, driver._receive = Mock(), 'left', Mock()
                driver.port.reset_input_buffer.side_effect = lambda: setattr(driver, 'motion_fault', 'peer disconnected')
                with self.assertRaisesRegex(CommandExpired, 'peer disconnected'):
                    writer(driver, values)
                driver.port.write.assert_not_called()
                driver._receive.assert_not_called()

    def test_clock_conversion_cannot_give_queued_commands_a_new_lifetime(self):
        ros, driver, clock, samples = self.make_ros()
        command = mobile_message(ros.node, [0, 1, 0, 0], expires_ns=1_050_000_000)
        clock[0] = 1.02
        old_clock = ros.node.get_clock
        def delayed_ros_read():
            value = old_clock().now()
            clock[0] = 1.08
            return value
        ros.node.get_clock = lambda: SimpleNamespace(now=delayed_ros_read)
        ros.mobile_command(command)
        ros.control.observe(samples)
        self.assertTrue(all(not any(row.values()) for row in driver.writes))
        self.assertEqual(ros.control.base, (0., 0., 0.))

    def test_clock_read_delay_cannot_refresh_expired_keyboard_input(self):
        ros, _, clock, _ = self.make_ros()
        bridge = object.__new__(PCChassisBridge)
        bridge.commands, bridge.command_until = [0, 1, 0, 0], 1.05
        bridge.received, bridge.state, bridge.mobile_pub = 1., {'enabled': True}, Mock()
        old_clock = ros.node.get_clock
        def delayed_ros_read():
            value = old_clock().now()
            clock[0] = 1.08
            return value
        bridge.get_clock = lambda: SimpleNamespace(now=delayed_ros_read)
        with patch('xyc_hardware.chassis_bridge.time.monotonic', lambda: clock[0]):
            bridge.publish_velocity()
        command = bridge.mobile_pub.publish.call_args.args[0]
        self.assertEqual(list(command.commands), [0, 0, 0, 0])

    def test_keyboard_and_model_publish_the_same_message_without_extending_deadlines(self):
        ros, _, clock, _ = self.make_ros()
        bridge = object.__new__(PCChassisBridge)
        bridge.commands, bridge.command_until = [1, 1, 0, 0], 1.05
        bridge.received, bridge.state = 1., {'enabled': True}
        bridge.get_clock, bridge.mobile_pub = ros.node.get_clock, Mock()
        with patch('xyc_hardware.chassis_bridge.time.monotonic', lambda: clock[0]):
            PCChassisBridge.publish_velocity(bridge)
        keyboard = bridge.mobile_pub.publish.call_args.args[0]
        model = SimpleNamespace(get_clock=ros.node.get_clock, mobile_pub=Mock(),
                                lock=threading.RLock(), require_active_link=lambda: None)
        original = ros.node.get_clock().now().to_msg()
        clock[0] = 1.02
        MotionReplay._publish_mobile(model, [1, 1, 0, 0], stamp=original, expires_ns=1_050_000_000)
        predicted = model.mobile_pub.publish.call_args.args[0]
        self.assertEqual(type(keyboard), type(predicted))
        self.assertEqual(keyboard.commands.tolist(), predicted.commands.tolist())
        self.assertEqual(keyboard.expires_at, predicted.expires_at)
        ros.mobile_command(predicted)
        self.assertAlmostEqual(ros.control.mobile_until, 1.05)
        model.mobile_pub.reset_mock()
        clock[0] = 1.08
        with self.assertRaises(TimeoutError):
            MotionReplay._publish_mobile(model, [1, 1, 0, 0], stamp=original, expires_ns=1_050_000_000)
        model.mobile_pub.publish.assert_not_called()
