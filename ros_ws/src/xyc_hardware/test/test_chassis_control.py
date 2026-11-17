# SPDX-License-Identifier: Apache-2.0
"""Chassis logic and on-wire velocity checks with fake actuators."""
import math
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from xyc_hardware.base_kinematics import OmniGeometry
from xyc_hardware.chassis_control import (
    AXIS_IDS, HOMING_SPEED_M_S, ChassisControl, ChassisDriver, ChassisNotReady, ChassisRos,
)
from xyc_hardware.chassis_bridge import PCChassisBridge
from xyc_hardware.lift_tracking import LiftGeometry, LiftTracker
from test_feetech_reader import FakeSerial


class FakeChassisDriver:
    def __init__(self):
        self.velocity_owned = set()
        self.writes = []
        self.fail_write = False

    def enable_velocity_axes(self):
        self.velocity_owned.update(AXIS_IDS)
        self.write_velocities(dict.fromkeys(AXIS_IDS, 0))

    def enable_lift(self):
        self.velocity_owned.add(11)

    def axis_read(self, motor_id, address, size):
        return b'\x01'

    def write_velocities(self, values):
        self.writes.append(dict(values))
        if self.fail_write:
            raise ConnectionError('injected velocity readback failure')

    def release_velocity_axes(self, ids=None):
        selected = self.velocity_owned.copy() if ids is None else set(ids)
        self.velocity_owned.difference_update(selected)


def make_control():
    driver = FakeChassisDriver()
    now = [1.]
    geometry = OmniGeometry(.063, .195, tuple(map(math.radians, (150., -90., 30.))),
                            (1, 1, 1), 3000 * math.tau / 4096)
    lift = LiftGeometry(.131, -1, 4000., .2, 0., .6)
    control = ChassisControl(driver, geometry, lift, clock=lambda: now[0])
    samples = {i: {'position_ticks': 2000, 'velocity_raw': 0, 'moving': False,
                   'temperature_c': 22., 'current_a': 0.} for i in AXIS_IDS}
    for index in range(13):
        now[0] = .76 + index * .02
        control.observe(samples)
    now[0] = 1.
    return control, driver, now, samples


class TestChassisControl(unittest.TestCase):
    def test_stationary_encoder_window_accepts_noisy_speed_but_rejects_real_motion(self):
        control, driver, now, samples = make_control()
        samples[11].update(velocity_raw=-50, moving=True)
        control.enable()
        self.assertTrue(control.enabled)
        control, driver, now, samples = make_control()
        for _ in range(12):
            now[0] += .02
            samples[11]['position_ticks'] += 1
            control.observe(samples)
        with self.assertRaisesRegex(ValueError, 'ID 11.*移动'):
            control.enable()
        self.assertFalse(control.enabled)
        self.assertEqual(driver.writes, [])

    def test_partial_enable_failure_is_cleaned_and_original_error_is_preserved(self):
        control, driver, _, _ = make_control()
        def fail_enable():
            driver.velocity_owned.add(8)
            raise ConnectionError('register 40 reply missing')
        driver.enable_velocity_axes = fail_enable
        with self.assertRaisesRegex(RuntimeError, 'register 40 reply missing'):
            control.enable()
        self.assertFalse(control.enabled)
        self.assertEqual(driver.velocity_owned, set())
        self.assertIn('使能未完成', control.fault)

    def test_recording_pairs_new_base_and_lift_requests_by_their_send_timestamp(self):
        from geometry_msgs.msg import TwistStamped
        control, _, _, _ = make_control()
        control.enable()
        control.height = .1
        control.tracker = LiftTracker(control.lift_geometry, reference_id='test-reference',
                                      zero_tick=2000, reference_time_s=1.)
        ros = object.__new__(ChassisRos)
        ros.control, ros.valid_command = control, MagicMock()
        ros.recording_command_publisher = MagicMock()
        ros.recording_requests, ros.last_recording_command_ns = {}, 0
        ros.node = SimpleNamespace(group_session_id='left-test',
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=200_000_000)))
        base, lift = TwistStamped(), TwistStamped()
        base.header.stamp.nanosec = lift.header.stamp.nanosec = 100_000_000
        base.twist.linear.x = .15
        lift.twist.linear.z = .015
        ros.base_command(base)
        ros.recording_command_publisher.publish.assert_not_called()
        ros.lift_command(lift)
        event = json.loads(ros.recording_command_publisher.publish.call_args.args[0].data)
        self.assertEqual(event['targets'], [.015, .15, 0., 0.])
        self.assertEqual(event['generated_ns'], 100_000_000)
        self.assertEqual(event['accepted_ns'], 200_000_000)
        ros.lift_command(lift)
        self.assertEqual(ros.recording_command_publisher.publish.call_count, 1)
        ros.valid_command.side_effect = ValueError('invalid')
        lift.header.stamp.nanosec = 120_000_000
        ros.lift_command(lift)
        self.assertEqual(ros.recording_command_publisher.publish.call_count, 1)
    def test_startup_and_read_only_feedback_write_nothing(self):
        control, driver, _, _ = make_control()
        self.assertFalse(control.enabled)
        self.assertEqual(driver.writes, [])

    def test_base_forward_sideways_and_yaw_follow_reference_geometry(self):
        for velocity, signs in (((.1, 0., 0.), (1, 0, -1)),
                                ((0., .1, 0.), (-1, 1, -1)),
                                ((0., 0., .4), (1, 1, 1))):
            with self.subTest(velocity=velocity):
                control, driver, now, samples = make_control()
                control.enable()
                control.set_base(*velocity)
                now[0] += .03
                control.observe(samples)
                speeds = driver.writes[-1]
                self.assertEqual(tuple(0 if speeds.get(i, 0) == 0 else
                                       1 if speeds[i] > 0 else -1 for i in (8, 9, 10)), signs)
                measured = control.geometry.wheels_to_body(
                    tuple(speeds.get(i, 0) * math.tau / 4096 for i in (8, 9, 10)))
                for expected, actual in zip(velocity, measured):
                    self.assertAlmostEqual(expected, actual, delta=.001)

    def test_missing_command_times_out_to_zero_with_valid_feedback(self):
        control, driver, now, samples = make_control()
        control.enable()
        control.set_base(.1, 0., 0.)
        now[0] += .03
        control.observe(samples)
        now[0] += .30
        control.observe(samples)
        self.assertTrue(control.enabled)
        self.assertEqual(driver.writes[-1], {8: 0, 10: 0})

    def test_stop_and_failed_write_zero_and_release_owned_axes(self):
        control, driver, now, samples = make_control()
        control.enable()
        control.set_base(.1, 0., 0.)
        control.observe(samples)
        control.stop()
        self.assertEqual(driver.writes[-1], dict.fromkeys(AXIS_IDS, 0))
        control.set_base(.1, 0., 0.)
        driver.fail_write = True
        control.observe(samples)
        self.assertFalse(control.enabled)
        self.assertEqual(driver.velocity_owned, set())
        self.assertIn('injected', control.fault)

    def test_missing_feedback_fault_latches_and_preserves_arm_separation(self):
        control, driver, now, samples = make_control()
        control.enable()
        samples.pop(9)
        now[0] += .21
        control.observe(samples)
        self.assertFalse(control.enabled)
        self.assertTrue(control.fault)
        self.assertTrue(all(set(write) <= set(AXIS_IDS) for write in driver.writes))
        with self.assertRaises(ValueError):
            control.set_base(.1, 0., 0.)

    def test_one_missing_poll_pauses_velocity_but_recovers_without_losing_height_reference(self):
        control, driver, now, samples = make_control()
        control.enable()
        control.tracker = LiftTracker(control.lift_geometry, reference_id='verified',
                                      zero_tick=2000, reference_time_s=now[0])
        samples[11]['position_ticks'] = 1900
        now[0] += .03
        control.observe(samples)
        tracker = control.tracker
        control.set_base(.1, 0., 0.)
        control.set_lift(.005)
        now[0] += .03
        control.observe(samples)
        self.assertTrue(any(control.sent.values()))
        incomplete = {i: row for i, row in samples.items() if i != 9}
        now[0] += .02
        control.observe(incomplete)
        self.assertTrue(control.enabled)
        self.assertTrue(control.feedback_paused)
        self.assertFalse(control.fault)
        self.assertEqual(control.sent, dict.fromkeys(AXIS_IDS, 0))
        self.assertEqual(control.missing_feedback_ids, (9,))
        for operation in (lambda: control.set_base(.1, 0., 0.),
                          lambda: control.set_lift(.005), control.begin_home):
            with self.assertRaises(ValueError):
                operation()
        now[0] += .03
        control.observe(samples)
        self.assertTrue(control.enabled)
        self.assertFalse(control.feedback_paused)
        self.assertIs(control.tracker, tracker)
        self.assertGreater(control.height, 0.)
        self.assertEqual(control.sent, dict.fromkeys(AXIS_IDS, 0))
        control.set_base(.1, 0., 0.)
        now[0] += .02
        control.observe(samples)
        self.assertTrue(any(control.sent.values()))

    def test_missing_samples_do_not_refresh_the_feedback_deadline(self):
        control, driver, now, samples = make_control()
        control.enable()
        missing = {i: row for i, row in samples.items() if i not in (8, 9, 10, 11)}
        for delay in (.03, .09, .15):
            now[0] = 1. + delay
            control.observe(missing, missing_detail='read deadline')
            self.assertTrue(control.enabled)
            self.assertEqual(control.last_feedback, 1.)
        now[0] = 1.21
        control.observe(missing, missing_detail='read deadline')
        self.assertFalse(control.enabled)
        self.assertFalse(driver.velocity_owned)
        self.assertIn('8,9,10,11', control.fault)
        self.assertIn('read deadline', control.fault)

    def test_late_recovery_cannot_reuse_an_ambiguous_reference(self):
        control, _, now, samples = make_control()
        control.enable()
        now[0] += .03
        control.observe({})
        now[0] += .21
        control.observe(samples)
        self.assertFalse(control.enabled)
        self.assertIn('反馈持续缺失', control.fault)

    def test_explicit_mobile_servo_alarm_is_not_treated_as_a_missing_poll(self):
        control, driver, _, _ = make_control()
        control.enable()
        ros = object.__new__(ChassisRos)
        ros.control, ros.last_status = control, time.monotonic()
        ros.node = SimpleNamespace(get_logger=lambda: MagicMock())
        ros.state_publisher = MagicMock()
        ros.observe(SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace()), motors=[SimpleNamespace(
            motor_id=11, valid=False, servo_error=4, error='hardware alarm')]))
        self.assertFalse(control.enabled)
        self.assertFalse(driver.velocity_owned)
        self.assertIn('ID 11', control.fault)
        self.assertIn('0x04', control.fault)

    def test_limits_and_nonfinite_values_are_rejected(self):
        control, _, _, _ = make_control()
        control.enable()
        for velocity in ((.3, 0., 0.), (0., 0., .9), (math.nan, 0., 0.)):
            with self.subTest(velocity=velocity), self.assertRaises(ValueError):
                control.set_base(*velocity)
        with self.assertRaises(ValueError):
            control.set_lift(.02)
        with self.assertRaisesRegex(ValueError, '零点'):
            control.set_lift(.01)

    def test_lift_uses_speed_mode_and_opposite_encoder_direction(self):
        control, driver, now, samples = make_control()
        control.enable()
        control.tracker = LiftTracker(control.lift_geometry, reference_id='verified',
                                      zero_tick=2000, reference_time_s=now[0])
        now[0] += .1
        samples[11]['position_ticks'] = 1687
        control.observe(samples)
        self.assertGreater(control.height, .009)
        control.set_lift(.01)
        now[0] += .03
        control.observe(samples)
        self.assertLess(driver.writes[-1][11], 0)
        control.set_lift(-.01)
        now[0] += .03
        control.observe(samples)
        self.assertGreater(driver.writes[-1][11], 0)

    def test_lift_descent_stops_at_the_software_floor(self):
        control, driver, now, samples = make_control()
        control.enable()
        control.tracker = LiftTracker(control.lift_geometry, reference_id='verified',
                                      zero_tick=2000, reference_time_s=now[0])
        control.set_lift(-.01)
        now[0] += .03
        control.observe(samples)
        self.assertEqual(control.sent[11], 0)

    def test_stalled_lift_stops_even_with_low_reported_current(self):
        control, driver, now, samples = make_control()
        control.enable()
        control.tracker = LiftTracker(control.lift_geometry, reference_id='verified',
                                      zero_tick=2000, reference_time_s=now[0])
        for _ in range(30):
            if not control.enabled:
                break
            control.set_lift(.01)
            now[0] += .03
            control.observe(samples)
        self.assertFalse(control.enabled)
        self.assertIn('未向指令方向移动', control.fault)
        self.assertEqual(driver.velocity_owned, set())

    def test_homing_requires_contact_then_settles_before_establishing_zero(self):
        control, driver, now, samples = make_control()
        control.enable()
        control.begin_home()
        now[0] += .03
        control.observe(samples)
        self.assertEqual(control.sent[11], round(HOMING_SPEED_M_S * 4096 / .131))
        samples[11]['current_a'] = .35
        for _ in range(5):
            now[0] += .03
            control.home_until = now[0] + .3
            control.observe(samples)
        self.assertEqual(control.homing, 'settling')
        self.assertNotIn(11, driver.velocity_owned)
        self.assertIsNone(control.height)
        samples[11]['current_a'] = 0.
        for _ in range(40):
            now[0] += .03
            control.home_until = now[0] + .3
            control.observe(samples)
        self.assertEqual(control.homing, '')
        self.assertEqual(control.height, 0.)
        self.assertIsNotNone(control.tracker)

    def test_standstill_without_contact_does_not_invent_lift_zero(self):
        control, _, now, samples = make_control()
        control.enable()
        control.begin_home()
        for _ in range(20):
            now[0] += .03
            control.home_until = now[0] + .3
            control.observe(samples)
        self.assertFalse(control.enabled)
        self.assertIsNone(control.tracker)
        self.assertIn('没有接触电流', control.fault)

    def test_bottom_contact_uses_encoder_window_despite_noisy_velocity(self):
        control, driver, now, samples = make_control()
        control.enable()
        control.begin_home()
        samples[11]['current_a'] = 5.23
        for index in range(10):
            now[0] += .02
            samples[11]['velocity_raw'] = 50 if index % 2 else -50
            samples[11]['position_ticks'] = 2000 + index % 2
            control.home_until = now[0] + .3
            control.observe(samples)
        self.assertEqual(control.homing, 'settling')
        self.assertTrue(control.enabled)
        self.assertNotIn(11, driver.velocity_owned)
        self.assertEqual(control.fault, '')
        samples[11]['current_a'] = 0.
        for index in range(65):
            now[0] += .02
            samples[11]['velocity_raw'] = 50 if index % 2 else -50
            samples[11]['position_ticks'] = 2000 + index % 2
            control.home_until = now[0] + .3
            control.observe(samples)
        self.assertEqual(control.height, 0.)
        self.assertEqual(control.homing, '')

    def test_slow_moving_encoder_is_not_mistaken_for_bottom_contact(self):
        control, _, now, samples = make_control()
        control.enable()
        control.begin_home()
        samples[11]['current_a'] = .35
        for _ in range(20):
            now[0] += .02
            samples[11]['position_ticks'] += 1
            samples[11]['velocity_raw'] = 0  # Even a misleading speed cannot establish zero.
            control.home_until = now[0] + .3
            control.observe(samples)
        self.assertEqual(control.homing, 'seeking')
        self.assertIsNone(control.height)

    def test_homing_retains_both_sustained_and_near_stall_current_protection(self):
        for current, ticks_per_sample, count in ((5.23, 5, 40), (8.5, 0, 8)):
            with self.subTest(current=current):
                control, driver, now, samples = make_control()
                control.enable()
                control.begin_home()
                samples[11]['current_a'] = current
                for _ in range(count):
                    now[0] += .02
                    samples[11]['position_ticks'] += ticks_per_sample
                    control.home_until = now[0] + .3
                    control.observe(samples)
                    if control.fault:
                        break
                self.assertIn('持续过流', control.fault)
                self.assertFalse(control.enabled)
                self.assertIsNone(control.height)
                self.assertEqual(driver.velocity_owned, set())

    def test_one_current_spike_does_not_establish_a_lift_zero(self):
        control, _, now, samples = make_control()
        control.enable()
        control.begin_home()
        for index in range(15):
            now[0] += .02
            samples[11]['current_a'] = .5 if index == 6 else 0.
            control.home_until = now[0] + .3
            control.observe(samples)
        self.assertEqual(control.homing, 'seeking')
        self.assertIsNone(control.height)

    def test_repeated_home_request_does_not_restart_homing(self):
        control, driver, now, samples = make_control()
        control.enable()
        control.begin_home()
        started = control.home_started
        now[0] += .03
        control.observe(samples)
        writes = list(driver.writes)
        control.begin_home()
        self.assertEqual(control.home_started, started)
        self.assertEqual(control.homing, 'seeking')
        self.assertEqual(driver.writes, writes)

    def test_floor_jitter_after_homing_keeps_base_enabled_but_real_overrun_stops(self):
        control, driver, now, samples = make_control()
        control.enable()
        control.tracker = LiftTracker(control.lift_geometry, reference_id='homed',
                                      zero_tick=2000, reference_time_s=now[0])
        for tick in (2001, 2004, 2010, 2000):
            now[0] += .03
            samples[11]['position_ticks'] = tick
            control.observe(samples)
            self.assertTrue(control.enabled)
            self.assertEqual(control.height, 0.)
        now[0] += .03
        samples[11]['position_ticks'] = 2011
        control.observe(samples)
        self.assertFalse(control.enabled)
        self.assertIn('Measured height', control.fault)
        self.assertFalse(driver.velocity_owned)

    def test_homing_loses_its_lease_when_the_keyboard_disconnects(self):
        control, _, now, samples = make_control()
        control.enable()
        control.begin_home()
        now[0] += .31
        control.observe(samples)
        self.assertFalse(control.enabled)
        self.assertIsNone(control.tracker)
        self.assertIn('控制端断开', control.fault)

    def test_direction_commands_during_homing_renew_connection_without_overriding_motion(self):
        from geometry_msgs.msg import TwistStamped
        for speed in (-.015, 0., .015):
            with self.subTest(speed=speed):
                control, driver, now, samples = make_control()
                control.enable()
                control.begin_home()
                ros = object.__new__(ChassisRos)
                ros.control, ros.valid_command = control, MagicMock()
                for _ in range(6):
                    now[0] += .1
                    message = TwistStamped()
                    message.twist.linear.z = speed
                    ros.lift_command(message)
                    control.observe(samples)
                    # Keep the simulated lift moving, rather than inventing
                    # stationary contact at its mechanical stop.
                    samples[11]['position_ticks'] += 50
                    samples[11]['velocity_raw'] = 500
                self.assertTrue(control.enabled)
                self.assertEqual(control.homing, 'seeking')
                self.assertEqual(control.sent[11], round(HOMING_SPEED_M_S * 4096 / .131))
                self.assertEqual(control.lift_speed, 0.)
                now[0] += .31
                control.observe(samples)
                self.assertFalse(control.enabled)
                self.assertIn('控制端断开', control.fault)


    def test_disabled_zero_heartbeats_preserve_the_initial_failure_reason(self):
        from geometry_msgs.msg import TwistStamped
        control, _, _, _ = make_control()
        control.reason = '操作被拒绝：ID 8 寄存器核对失败'
        ros = object.__new__(ChassisRos)
        ros.control, ros.valid_command = control, MagicMock()
        for callback in (ros.base_command, ros.lift_command):
            callback(TwistStamped())
            self.assertIn('ID 8 寄存器核对失败', control.reason)


class TestChassisPackets(unittest.TestCase):
    def test_failed_torque_enable_rolls_back_before_later_axes_and_keeps_uncertainty(self):
        for fail_release in (False, True):
            with self.subTest(fail_release=fail_release):
                driver = object.__new__(ChassisDriver)
                driver.velocity_owned = set()
                memory = {(i, a): v for i in AXIS_IDS
                          for a, v in ((18, b'\x04'), (33, b'\x01'), (40, b'\x00'))}
                driver.axis_read = lambda i, a, size: memory[i, a]
                def write(i, a, value):
                    if i == 8 and a == 40 and value == b'\x00' and fail_release:
                        raise ConnectionError('off not confirmed')
                    memory[i, a] = value
                    if i == 8 and a == 40 and value == b'\x01':
                        raise ConnectionError('on readback lost after write')
                driver.axis_write = write
                driver.write_velocities = MagicMock()
                with self.assertRaisesRegex((ConnectionError, RuntimeError), 'on readback lost'):
                    driver.enable_velocity_axes()
                self.assertTrue(all(memory[i, 40] == b'\x00' for i in (9, 10, 11)))
                self.assertEqual(driver.velocity_owned, {8} if fail_release else set())
                if not fail_release:
                    self.assertEqual(memory[8, 40], b'\x00')

    def test_axis_read_retries_only_reads_and_rejects_invalid_torque_bytes(self):
        from xyc_hardware.feetech_reader import Reply
        driver = object.__new__(ChassisDriver)
        driver._send_joint_packet = MagicMock()
        driver._receive = MagicMock(side_effect=[{}, {8: Reply(b'\x32', 0, 1)},
                                                 {8: Reply(b'\x00', 0, 2)}])
        self.assertEqual(driver.axis_read(8, 40, 1), b'\x00')
        self.assertEqual(driver._send_joint_packet.call_count, 3)
        self.assertTrue(all(call.args[1] == 2 for call in driver._send_joint_packet.call_args_list))

    def test_configuration_primes_zero_before_enabling_and_never_touches_arm_ids(self):
        driver = object.__new__(ChassisDriver)
        driver.velocity_owned = set()
        memory = {(i, a): v for i in AXIS_IDS
                  for a, v in ((18, b'\x04'), (33, b'\x00'), (40, b'\x00'))}
        operations = []
        driver.axis_read = lambda i, a, size: memory[i, a]
        def write(i, a, value):
            operations.append((i, a, value))
            memory[i, a] = value
        driver.axis_write = write
        driver.write_velocities = lambda values: operations.append(dict(values))
        driver.enable_velocity_axes()
        zero_index = operations.index(dict.fromkeys(AXIS_IDS, 0))
        enabled = [index for index, op in enumerate(operations)
                   if isinstance(op, tuple) and op[1:] == (40, b'\x01')]
        self.assertEqual(len(enabled), 4)
        self.assertTrue(all(index > zero_index for index in enabled))
        self.assertTrue(all(op[0] in AXIS_IDS for op in operations if isinstance(op, tuple)))
        self.assertTrue(all(memory[i, 33] == b'\x01' for i in AXIS_IDS))
        self.assertTrue(all(memory[i, 55] == b'\x01' for i in AXIS_IDS))
        for motor_id in AXIS_IDS:
            self.assertLess(operations.index((motor_id, 85, b'\xfe')),
                            operations.index((motor_id, 41, b'\xfe')))

    def test_signed_speed_batch_has_only_velocity_ids_and_registers(self):
        targets = {8: 600, 9: -500, 10: 0, 11: -200}
        replies = []
        for motor_id, speed in targets.items():
            data = (abs(speed) | (0x8000 if speed < 0 else 0)).to_bytes(2, 'little')
            body = bytes([motor_id, 4, 0]) + data
            replies.append(b'\xff\xff' + body + bytes([~sum(body) & 255]))
        serial = FakeSerial([b''.join(replies)])
        with patch('xyc_hardware.feetech_reader.serial.Serial', return_value=serial):
            driver = ChassisDriver('/unused', list(range(1, 12)), bus='left', joint_id=6)
        driver.write_velocities(targets)
        self.assertEqual(serial.writes[0][4:7], b'\x83\x2e\x02')
        self.assertEqual(serial.writes[1][4:7], b'\x82\x2e\x02')
        self.assertIn(b'\x09\xf4\x81', serial.writes[0])
        self.assertIn(b'\x0b\xc8\x80', serial.writes[0])
        for values in ({7: 10}, {11: 1301}, {8: 3001}):
            with self.assertRaises(ValueError):
                driver.write_velocities(values)
        with self.assertRaises(ValueError):
            driver.axis_write(1, 40, b'\x01')


@unittest.skipUnless(os.environ.get('XYC_ISOLATED_TEST') == '1' and
                     os.environ.get('ROS_DOMAIN_ID') == '181', 'requires isolated ROS domain 181')
class TestPCChassisROS(unittest.TestCase):
    def setUp(self):
        import rclpy
        import yaml
        from rclpy.executors import SingleThreadedExecutor
        from xyc_hardware.feetech_reader import Reply
        from xyc_hardware.motor_state_node import MotorStateNode
        self.rclpy = rclpy
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        config = Path(__file__).parents[1] / 'config/motors.yaml'
        parameters = yaml.safe_load(config.read_text())['/motors/left/motor_state']['ros__parameters']
        parameters.update(port='/fake/chassis', chassis_mode=True)
        path = Path(self.directory.name) / 'parameters.yaml'
        path.write_text(yaml.safe_dump({'/**': {'ros__parameters': parameters}}))
        rclpy.init(args=['--ros-args', '--params-file', str(path)])
        self.addCleanup(rclpy.try_shutdown)
        class Reader(FakeChassisDriver):
            closed = False
            contact = False
            def open(self):
                pass
            def close(self):
                self.closed = True
            def read_models(self):
                return {i: 2569 if i in (2, 3, 11) else 2825 for i in range(1, 12)}
            def read_feedback(self):
                output = {}
                for i in range(1, 12):
                    raw = bytearray(15)
                    raw[:2] = (2000).to_bytes(2, 'little')
                    raw[6], raw[7] = 120, 22
                    raw[13:15] = (54 if i == 11 and self.contact else 0).to_bytes(2, 'little')
                    output[i] = Reply(bytes(raw), 0, time.time_ns())
                return output
        self.reader = Reader()
        with patch('xyc_hardware.chassis_control.ChassisDriver', return_value=self.reader):
            self.motor = MotorStateNode()
        self.keyboard = PCChassisBridge()
        self.operator = None
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.motor)
        self.executor.add_node(self.keyboard)
        self.spin_until(lambda: self.keyboard.received is not None)

    def spin_until(self, predicate, timeout=4.):
        until = time.monotonic() + timeout
        while time.monotonic() < until and not predicate():
            if self.operator is not None:
                self.operator.receive({'chassis_supported': True, **self.keyboard.snapshot()})
                self.keyboard.receive_packet('operator', self.operator.command())
            self.executor.spin_once(timeout_sec=.02)
        self.assertTrue(predicate())

    def tearDown(self):
        self.executor.shutdown()
        self.keyboard.destroy_node()
        self.motor.destroy_node()
        self.rclpy.try_shutdown()
        self.directory.cleanup()

    def connect_operator(self):
        from test_pc_teleop import pc
        self.keyboard.begin_session('operator')
        self.operator = pc.LocalChassisKeyboard()
        self.spin_until(lambda: self.keyboard.state.get('enabled') and self.operator.operation is None)

    def test_pc_keyboard_drives_then_stops_on_key_expiry(self):
        self.assertEqual(self.reader.writes, [])
        self.connect_operator()
        self.operator.handle_key('W')
        self.spin_until(lambda: self.motor.chassis.control.sent[8] > 0)
        self.spin_until(lambda: self.motor.chassis.control.sent[8] == 0)
        self.assertTrue(self.motor.chassis.control.enabled)
        self.assertTrue(all(set(write) <= set(AXIS_IDS) for write in self.reader.writes))

    def create_pc_bridge(self):
        self.keyboard.timer.cancel()
        bridge = PCChassisBridge()
        self.executor.add_node(bridge)
        self.addCleanup(bridge.destroy_node)
        self.addCleanup(self.executor.remove_node, bridge)
        self.spin_until(lambda: bridge.received is not None)
        bridge.begin_session('test')
        return bridge

    def test_automatic_enable_waits_for_service_discovery_then_reaches_hardware(self):
        bridge = self.create_pc_bridge()
        bridge.timer.cancel()
        payload = {'base': [0., 0., 0.], 'lift_m_s': 0.,
                   'operation': {'id': 1, 'name': 'enable'}}
        bridge.receive_packet('test', payload)
        client = bridge.control_clients['enable']
        with patch.object(client, 'service_is_ready', return_value=False):
            bridge.publish_commands()
        self.assertTrue(bridge.snapshot()['chassis_ack']['pending'])
        self.assertEqual(bridge.last_operation_id, 0)
        self.assertEqual(self.reader.writes, [])
        self.spin_until(client.service_is_ready)
        def accepted():
            bridge.receive_packet('test', payload)
            bridge.publish_commands()
            return bridge.ack is not None and bridge.ack.get('accepted')
        self.spin_until(accepted)
        self.assertTrue(self.motor.chassis.control.enabled)

    def test_automatic_enable_retries_only_known_prewrite_startup_rejections(self):
        bridge = self.create_pc_bridge()
        bridge.timer.cancel()
        payload = {'base': [0., 0., 0.], 'lift_m_s': 0.,
                   'operation': {'id': 1, 'name': 'enable'}}
        self.spin_until(bridge.control_clients['enable'].service_is_ready)
        original = self.motor.chassis.control.enable
        calls = []
        def initially_waiting():
            calls.append(len(self.reader.writes))
            if len(calls) == 1:
                raise ChassisNotReady('等待底盘和升降的新鲜反馈')
            return original()
        self.motor.chassis.control.enable = initially_waiting
        def accepted():
            bridge.receive_packet('test', payload)
            bridge.publish_commands()
            return bridge.ack is not None and bridge.ack.get('accepted')
        self.spin_until(accepted)
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(all(count == 0 for count in calls))
        self.assertTrue(self.motor.chassis.control.enabled)

    def test_failed_hardware_enable_is_reported_without_automatic_write_retry(self):
        bridge = self.create_pc_bridge()
        bridge.timer.cancel()
        self.spin_until(bridge.control_clients['enable'].service_is_ready)
        self.motor.chassis.control.enable = MagicMock(side_effect=ConnectionError('ID 8 寄存器核对失败'))
        payload = {'base': [0., 0., 0.], 'lift_m_s': 0.,
                   'operation': {'id': 1, 'name': 'enable'}}
        def rejected():
            bridge.receive_packet('test', payload)
            bridge.publish_commands()
            return bridge.ack is not None and bridge.ack.get('accepted') is False
        self.spin_until(rejected)
        bridge.receive_packet('test', payload)
        bridge.publish_commands()
        self.motor.chassis.control.enable.assert_called_once()
        self.assertIn('ID 8', bridge.ack['message'])
        self.assertIn('ID 8', self.motor.chassis.control.reason)

    def test_keyboard_zero_heartbeats_allow_homing_then_space_stops_and_exit_releases(self):
        self.connect_operator()
        self.operator.handle_key('h')
        self.reader.contact = True
        self.spin_until(lambda: self.motor.chassis.control.homing == 'settling')
        self.reader.contact = False
        self.spin_until(lambda: self.motor.chassis.control.height == 0.
                        and self.operator.state.get('height_m') == 0.
                        and not self.operator.state.get('homing'))
        self.operator.handle_key('i')
        self.spin_until(lambda: self.motor.chassis.control.sent[11] < 0)
        self.operator.handle_key(' ')
        self.spin_until(lambda: self.motor.chassis.control.sent[11] == 0)
        self.motor.chassis.close()
        self.assertEqual(self.reader.velocity_owned, set())

    def test_stale_topic_command_and_feedback_loss_stop_the_chassis(self):
        from geometry_msgs.msg import TwistStamped
        self.connect_operator()
        c = self.motor.chassis.control
        c.set_base(.1, 0., 0.)
        expired = TwistStamped()
        expired.header.frame_id = 'base_link'
        expired.header.stamp.sec = 1
        expired.twist.linear.x = .2
        self.motor.chassis.base_command(expired)
        self.assertEqual(c.base, (0., 0., 0.))
        c.last_feedback = time.monotonic() - .3
        self.motor.chassis.check_feedback()
        self.assertFalse(c.enabled)
        self.assertEqual(self.reader.velocity_owned, set())

    def test_pc_tcp_session_relays_keyboard_and_stops_on_disconnect(self):
        import socket
        import threading
        from test_pc_teleop import pc
        from xyc_hardware.teleop_nodes import LeaderStateBridge
        from xyc_hardware.teleop_transport import read_frame, send_frame
        self.keyboard.timer.cancel()
        bridge = LeaderStateBridge()
        self.executor.add_node(bridge)
        self.executor.add_node(bridge.pc_chassis)
        self.spin_until(lambda: bridge.pc_chassis.received is not None)
        pair = socket.socketpair()
        failures, emitted = [], []
        class Complete(Exception):
            pass
        original_publish = bridge.publish_frame
        def publish(*args):
            emitted.append(self.motor.chassis.control.enabled)
            original_publish(*args)
            if (len(emitted) >= 8 and any(write.get(8, 0) > 0 for write in self.reader.writes)
                    or len(emitted) >= 100):
                raise Complete
        bridge.publish_frame = publish
        def server():
            try:
                with pair[0], pair[0].makefile('rwb') as channel:
                    try:
                        bridge.publish_session(channel)
                    except Complete:
                        pass
            except Exception as exc:
                failures.append(exc)
        def client():
            keyboard = pc.LocalChassisKeyboard()
            try:
                with pair[1], pair[1].makefile('rwb') as channel:
                    sample = 0
                    while True:
                        request = read_frame(channel)
                        keyboard.receive(request)
                        keyboard.handle_key('w')
                        sample += 1
                        send_frame(channel, {'sequence': request['sequence'],
                            'sample_id': sample, 'sample_age_ns': 0,
                            'positions': {'left': [0.] * 7, 'right': [0.] * 7},
                            'calibration_ids': {'left': '0' * 64, 'right': '1' * 64},
                            'chassis': keyboard.command()})
            except (EOFError, BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                failures.append(exc)
        threads = [threading.Thread(target=server, daemon=True),
                   threading.Thread(target=client, daemon=True)]
        try:
            for thread in threads:
                thread.start()
            self.spin_until(lambda: not threads[0].is_alive(), timeout=5.)
            self.spin_until(lambda: self.motor.chassis.control.sent.get(8) == 0)
            self.assertEqual(failures, [])
            self.assertGreaterEqual(len(emitted), 8)
            self.assertLessEqual(len(emitted), 100)
            self.assertTrue(any(write.get(8, 0) > 0 for write in self.reader.writes))
            self.assertIsNone(bridge.pc_chassis.packet)
        finally:
            for endpoint in pair:
                try:
                    endpoint.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                endpoint.close()
            for thread in threads:
                thread.join(timeout=2.)
            self.executor.remove_node(bridge.pc_chassis)
            self.executor.remove_node(bridge)
            bridge.pc_chassis.destroy_node()
            bridge.destroy_node()

    def test_leader_frames_start_without_any_chassis_enable_ack(self):
        import json
        from xyc_hardware.teleop_nodes import LeaderStateBridge
        bridge = LeaderStateBridge()
        class Complete(Exception):
            pass
        frame = {'sequence': 1, 'sample_id': 1, 'sample_age_ns': 0,
                 'positions': {'left': [0.] * 7}, 'calibration_ids': {'left': '0' * 64},
                 'chassis': {'base': [0., 0., 0.], 'lift_m_s': 0., 'operation': None}}
        channel = MagicMock()
        channel.readline.return_value = json.dumps(frame).encode() + b'\n'
        bridge.publish_frame = MagicMock(side_effect=Complete)
        try:
            with self.assertRaises(Complete):
                bridge.publish_session(channel)
            bridge.publish_frame.assert_called_once()
            self.assertFalse(self.motor.chassis.control.enabled)
        finally:
            bridge.pc_chassis.destroy_node()
            bridge.destroy_node()


if __name__ == '__main__':
    unittest.main()
