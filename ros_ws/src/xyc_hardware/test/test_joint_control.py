# SPDX-License-Identifier: Apache-2.0
"""Persistent joint-control checks use fake motors and never open a serial port."""
import json
import unittest
import serial
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from xyc_hardware.arm_calibration import arm_names, denormalize_tick, normalize_tick
from xyc_hardware.arm_joint_control import (
    ArmBusHold, ArmBusTransaction, JointControl, JointDriver,
)
from test_arm_calibration import fixture


class FakeJoint:
    def __init__(self, entry):
        self.entry = entry.copy()
        self.joint_id = entry['id']
        self.bus = 'right'
        self.position = 2000
        self.torque = 0
        self.writes = []
        self.fail_enable = False
        self.fail_release = False
        self.fail_goal_number = 0
        self.torque_limit = None

    def read_coordinate_settings(self, ids):
        assert ids == [self.joint_id]
        return {self.joint_id: {key: self.entry[key] for key in ('hardware_homing_offset', 'operating_mode')}}

    def read_torque(self):
        return self.torque

    def read_joint(self):
        return {'position_ticks': self.position, 'velocity_raw': 0, 'load_raw': 0,
                'temperature_c': 25., 'current_a': 0., 'moving': False}

    def write_goal(self, ticks):
        self.writes.append(('goal', ticks))
        if sum(kind == 'goal' for kind, _ in self.writes) == self.fail_goal_number:
            raise OSError('Goal readback failed')

    def write_torque_limit(self, value):
        self.torque_limit = value
        self.writes.append(('torque_limit', value))

    def set_torque(self, enabled):
        self.writes.append(('torque', enabled))
        if not enabled and self.fail_release:
            raise OSError('torque-off not confirmed')
        self.torque = int(enabled)
        if enabled and self.fail_enable:
            raise OSError('enable reply lost')


class TestTorqueOffConfirmation(unittest.TestCase):
    def driver(self):
        driver = JointDriver.__new__(JointDriver)
        driver._write_verified = MagicMock()
        driver.read_torque = MagicMock()
        return driver

    def test_off_verification_failure_rechecks_hardware_before_retrying(self):
        driver = self.driver()
        driver._write_verified.side_effect = [OSError('readback mismatch'), None]
        driver.read_torque.return_value = 1
        driver.set_torque(False)
        self.assertEqual(driver._write_verified.call_count, 2)
        self.assertTrue(all(call.args == (40, b'\x00') for call in driver._write_verified.call_args_list))
        driver.read_torque.assert_called_once()

    def test_lost_off_reply_is_success_when_hardware_confirms_off(self):
        driver = self.driver()
        driver._write_verified.side_effect = OSError('lost readback')
        driver.read_torque.return_value = 0
        driver.set_torque(False)
        driver._write_verified.assert_called_once_with(40, b'\x00')

    def test_unknown_hardware_state_is_not_retried_or_reported_off(self):
        driver = self.driver()
        driver._write_verified.side_effect = OSError('write uncertain')
        driver.read_torque.side_effect = OSError('no feedback')
        with self.assertRaisesRegex(ConnectionError, 'state unavailable'):
            driver.set_torque(False)
        driver._write_verified.assert_called_once()

    def test_persistent_off_failure_is_bounded_and_reported(self):
        driver = self.driver()
        driver._write_verified.side_effect = OSError('readback mismatch')
        driver.read_torque.return_value = 1
        with self.assertRaisesRegex(ConnectionError, 'remains enabled after 3'):
            driver.set_torque(False)
        self.assertEqual(driver._write_verified.call_count, 3)

    def test_enable_is_never_repeated_after_a_lost_reply(self):
        driver = self.driver()
        driver._write_verified.side_effect = OSError('lost enable reply')
        with self.assertRaises(OSError):
            driver.set_torque(True)
        driver._write_verified.assert_called_once_with(40, b'\x01')
        driver.read_torque.assert_not_called()


class TestArmBusHold(unittest.TestCase):
    def make_group(self, *, write_group_goals=None):
        document = fixture('right')[2]
        controls = {}
        motors = {}
        for name in arm_names('right'):
            entry = document['motors'][name]
            motors[name] = FakeJoint(entry)
            controls[name] = JointControl(motors[name], entry)
        clock = [0.0]
        def read_group_snapshot():
            return {
                motor.joint_id: (motor.read_coordinate_settings([motor.joint_id])[motor.joint_id],
                                 motor.read_torque(), motor.read_joint())
                for motor in motors.values()
            }
        group = ArmBusHold('right', controls, max_prepare_age_s=1.0, clock=lambda: clock[0],
                           write_group_goals=write_group_goals,
                           read_group_snapshot=read_group_snapshot)
        return group, motors, clock

    def test_preflight_is_read_only_then_holds_all_seven(self):
        group, motors, _ = self.make_group()
        group.prepare()
        self.assertTrue(all(not motor.writes for motor in motors.values()))
        group.start()
        self.assertEqual(group.state, 'holding')
        for name, motor in motors.items():
            expected = ([('torque_limit', 100)] if name.endswith('_gripper') else [])
            expected += [('goal', 2000), ('torque', True)]
            self.assertEqual(motor.writes, expected, name)
            self.assertTrue(group.controls[name].owned)

    def test_replay_margin_zero_accepts_recorded_endpoint_and_still_rejects_real_overrun(self):
        for joint, start, target in (
            ('elbow_flex', 97.80018, 99.35838),
            ('elbow_flex', 98., 100.),
            ('elbow_flex', -98., -100.),
            ('gripper', 2., 0.),
            ('gripper', 98., 100.),
        ):
            with self.subTest(joint=joint, target=target):
                entry = fixture('right')[2]['motors']['arm_right_' + joint]
                motor = FakeJoint(entry)
                motor.position = denormalize_tick(entry, start)
                control = JointControl(motor, entry)
                control.hold_current_position()
                control.set_absolute_target(target)
                self.assertEqual(control.target_ticks, denormalize_tick(entry, target))
                self.assertEqual(motor.torque, 1)
                before = list(motor.writes)
                overrun = -0.01 if joint == 'gripper' else -100.01
                for value in (overrun, 100.01):
                    with self.assertRaisesRegex(ValueError, 'normalization range'):
                        control.inspect_absolute_target(value)
                self.assertEqual(motor.writes, before)

    def test_absolute_goal_is_kept_in_full_until_the_servo_moves(self):
        entry = fixture('right')[2]['motors']['arm_right_elbow_flex']
        motor = FakeJoint(entry)
        control = JointControl(motor, entry)
        control.hold_current_position()
        before = list(motor.writes)
        plan = control.inspect_absolute_target(80.)
        self.assertEqual(plan.target_ticks, 2800)
        self.assertEqual(motor.writes, before)
        control.set_absolute_target(80.)
        self.assertEqual(motor.writes[-1], ('goal', 2800))
        self.assertEqual(control.target_ticks, 2800)

    def test_large_targets_and_reversals_are_preserved_for_joint_and_gripper(self):
        for joint, targets in (('wrist_roll', (80., -80., 100., -100.)),
                               ('gripper', (100., 0., 85., 15.))):
            with self.subTest(joint=joint):
                entry = fixture('right')[2]['motors']['arm_right_' + joint]
                motor = FakeJoint(entry)
                control = JointControl(motor, entry)
                control.hold_current_position()
                for target in targets:
                    control.set_absolute_target(target)
                    expected = denormalize_tick(entry, target)
                    self.assertEqual(control.target_ticks, expected)
                    self.assertEqual(motor.writes[-1], ('goal', expected))
                self.assertEqual(motor.torque, 1)

    def test_full_goal_preserves_wrap_rejection_instead_of_cropping_the_target(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll'].copy()
        entry.update(reference_tick=4090, min_delta_ticks=-80, max_delta_ticks=86)
        motor = FakeJoint(entry)
        motor.position = 4080
        control = JointControl(motor, entry)
        control.hold_current_position()
        before = list(motor.writes)
        with self.assertRaisesRegex(ValueError, 'encoder wrap'):
            control.inspect_absolute_target(80.)
        self.assertEqual(motor.writes, before)

    def test_preflight_failure_writes_nothing(self):
        group, motors, _ = self.make_group()
        motors[group.names[4]].torque = 1
        with self.assertRaisesRegex(ValueError, 'preflight failed'):
            group.prepare()
        self.assertTrue(all(not motor.writes for motor in motors.values()))

    def test_endpoint_feedback_can_hold_raw_position_then_move_inward(self):
        for tick, target in ((984, -100.), (984, -99.5), (3016, 100.), (3016, 99.5)):
            with self.subTest(tick=tick):
                group, motors, _ = self.make_group()
                name = group.names[1]
                motor = motors[name]
                motor.position = tick
                group.prepare()
                self.assertTrue(all(not item.writes for item in motors.values()))
                group.start()
                self.assertEqual(motor.writes, [('goal', tick), ('torque', True)])
                targets = [0.] * 6 + [50.]
                targets[1] = target
                group.apply_targets(targets)
                self.assertIsNone(group.controls[name].observe(motor.read_joint()))
                self.assertEqual(motor.torque, 1)
                self.assertGreater((group.controls[name].target_ticks - tick) * -target, 0.)

    def test_startup_beyond_endpoint_tolerance_rejects_before_any_writes(self):
        for tick in (983, 3017):
            with self.subTest(tick=tick):
                group, motors, _ = self.make_group()
                motors[group.names[1]].position = tick
                with self.assertRaisesRegex(ValueError, 'outside the calibrated range'):
                    group.prepare()
                self.assertTrue(all(not motor.writes for motor in motors.values()))

    def test_stale_or_changed_positions_write_nothing(self):
        for mode in ('stale', 'moved', 'backwards'):
            with self.subTest(mode=mode):
                group, motors, clock = self.make_group()
                group.prepare()
                if mode == 'stale':
                    clock[0] = 2.0
                elif mode == 'backwards':
                    clock[0] = -1.0
                else:
                    motors[group.names[3]].position += 3
                with self.assertRaises((ValueError, RuntimeError)):
                    group.start()
                self.assertTrue(all(not motor.writes for motor in motors.values()))

    def test_lost_enable_ack_releases_previously_touched_joints(self):
        group, motors, _ = self.make_group()
        motors[group.names[3]].fail_enable = True
        group.prepare()
        with self.assertRaisesRegex(RuntimeError, 'enable reply lost'):
            group.start()
        self.assertEqual(group.state, 'fault')
        self.assertTrue(all(motor.torque == 0 for motor in motors.values()))
        self.assertTrue(all(not motors[name].writes for name in group.names[4:]))
        with self.assertRaises(ValueError):
            group.start()

    def test_release_failure_reported_while_other_joints_are_released(self):
        group, motors, _ = self.make_group()
        motors[group.names[1]].fail_release = True
        motors[group.names[3]].fail_enable = True
        group.prepare()
        with self.assertRaisesRegex(RuntimeError, 'torque-off unconfirmed'):
            group.start()
        self.assertEqual(group.state, 'fault')
        self.assertEqual(motors[group.names[1]].torque, 1)
        self.assertEqual(motors[group.names[0]].torque, 0)

    def test_bus_identity_must_match(self):
        group, _, _ = self.make_group()
        group.controls[group.names[0]].driver.bus = 'left'
        with self.assertRaisesRegex(ValueError, 'identity'):
            ArmBusHold('right', group.controls, max_prepare_age_s=1.0)

    def test_inspection_that_takes_too_long_is_not_prepared(self):
        group, motors, clock = self.make_group()
        motor = motors[group.names[-1]]
        original = motor.read_joint
        def slow_read():
            clock[0] = 2.0
            return original()
        motor.read_joint = slow_read
        with self.assertRaisesRegex(ValueError, 'expired'):
            group.prepare()
        self.assertEqual(group.state, 'idle')
        self.assertTrue(all(not motor.writes for motor in motors.values()))

    def test_interrupted_enable_attempt_still_releases_touched_joints(self):
        group, motors, _ = self.make_group()
        motor = motors[group.names[3]]
        original = motor.set_torque
        def interrupted(enabled):
            original(enabled)
            if enabled:
                raise KeyboardInterrupt()
        motor.set_torque = interrupted
        group.prepare()
        with self.assertRaises(KeyboardInterrupt):
            group.start()
        self.assertTrue(all(motor.torque == 0 for motor in motors.values()))
        self.assertEqual(group.state, 'fault')

    def test_group_targets_are_checked_before_any_goal_write(self):
        group, motors, _ = self.make_group()
        group.prepare()
        group.start()
        targets = [control.inspect_absolute_target(0.).current_normalized
                   for control in list(group.controls.values())[:6]] + [50.]
        targets[0] += 2.
        targets[-1] = float('nan')
        before = {name: list(motor.writes) for name, motor in motors.items()}
        with self.assertRaisesRegex(ValueError, 'inspection failed'):
            group.apply_targets(targets)
        self.assertEqual(before, {name: motor.writes for name, motor in motors.items()})
        self.assertTrue(all(motor.torque == 1 for motor in motors.values()))

    def test_group_target_rejection_reports_latched_joint_fault(self):
        group, motors, _ = self.make_group()
        plans = group.prepare()
        group.start()
        name = group.names[2]
        group.controls[name].fault = 'Joint sustained overcurrent'
        before = {key: list(motor.writes) for key, motor in motors.items()}
        with self.assertRaisesRegex(ValueError, 'Joint fault: Joint sustained overcurrent'):
            group.apply_targets([plan.current_normalized for plan in plans])
        self.assertEqual(before, {key: motor.writes for key, motor in motors.items()})

    def test_group_targets_dispatch_and_remain_powered(self):
        group, motors, _ = self.make_group()
        plans = group.prepare()
        group.start()
        targets = [plan.current_normalized + 1. for plan in plans]
        group.apply_targets(targets)
        self.assertTrue(all(motor.torque == 1 for motor in motors.values()))
        self.assertTrue(all(sum(kind == 'goal' for kind, _ in motor.writes) == 2
                            for motor in motors.values()))

    def test_group_writes_only_changed_joints_and_updates_feedback_monitor(self):
        write_group = MagicMock()
        group, motors, _ = self.make_group(write_group_goals=write_group)
        plans = group.prepare()
        group.start()
        targets = [plan.current_normalized for plan in plans]
        targets[0] += 1.
        targets[4] -= 2.
        group.apply_targets(targets)
        group.apply_targets(targets)
        write_group.assert_called_once_with({1: 2010, 5: 1980})
        for name, tick in ((group.names[0], 2010), (group.names[4], 1980)):
            control = group.controls[name]
            self.assertEqual((control.target_ticks, control.start_ticks), (tick, 2000))
            self.assertTrue(control.busy)
        for name, motor in motors.items():
            expected = ([('torque_limit', 100)] if name.endswith('_gripper') else [])
            expected += [('goal', 2000), ('torque', True)]
            self.assertEqual(motor.writes, expected, name)

    def test_group_releases_owned_joints_on_missing_readback(self):
        write_group = MagicMock(side_effect=ConnectionError('group readback missing'))
        group, motors, _ = self.make_group(write_group_goals=write_group)
        plans = group.prepare()
        group.start()
        with self.assertRaisesRegex(RuntimeError, 'group readback missing'):
            group.apply_targets([plan.current_normalized + 1. for plan in plans])
        self.assertEqual(group.state, 'fault')
        self.assertTrue(all(motor.torque == 0 for motor in motors.values()))

    def test_dispatch_error_releases_all_and_reports_unknown_outcome(self):
        group, motors, _ = self.make_group()
        plans = group.prepare()
        group.start()
        motors[group.names[3]].fail_goal_number = 2
        targets = [plan.current_normalized + 1. for plan in plans]
        with self.assertRaisesRegex(RuntimeError, 'outcome of last attempt may be unknown'):
            group.apply_targets(targets)
        self.assertEqual(group.state, 'fault')
        self.assertTrue(all(motor.torque == 0 for motor in motors.values()))
        self.assertEqual(sum(kind == 'goal' for kind, _ in motors[group.names[4]].writes), 1)

    def test_dispatch_error_reports_unconfirmed_release(self):
        group, motors, _ = self.make_group()
        plans = group.prepare()
        group.start()
        motors[group.names[3]].fail_goal_number = 2
        motors[group.names[1]].fail_release = True
        with self.assertRaisesRegex(RuntimeError, 'torque-off unconfirmed'):
            group.apply_targets([plan.current_normalized + 1. for plan in plans])
        self.assertEqual(group.state, 'fault')
        self.assertEqual(motors[group.names[1]].torque, 1)
        self.assertEqual(motors[group.names[0]].torque, 0)

    def test_transaction_abort_and_expiry_never_enable_motors(self):
        group, motors, clock = self.make_group()
        tx = ArmBusTransaction(group, timeout_s=.5, clock=lambda: clock[0])
        tx.prepare_hold('1')
        with self.assertRaises(ValueError):
            tx.commit('99')
        tx.abort('1')
        with self.assertRaisesRegex(ValueError, 'already used'):
            tx.prepare_hold('1')
        tx.prepare_hold('2')
        clock[0] = .6
        with self.assertRaisesRegex(ValueError, 'matching'):
            tx.commit('2')
        self.assertEqual(group.state, 'idle')
        self.assertTrue(all(not motor.writes for motor in motors.values()))

    def test_transaction_freezes_targets_and_cannot_be_replayed(self):
        group, motors, clock = self.make_group()
        tx = ArmBusTransaction(group, timeout_s=.5, clock=lambda: clock[0])
        tx.prepare_hold('1')
        tx.commit('1')
        plans = [control.inspect_absolute_target(0.) for control in list(group.controls.values())[:6]]
        # Gripper uses 0..100 rather than -100..100.
        gripper = group.controls[group.names[-1]].inspect_absolute_target(50.)
        targets = [plan.current_normalized + 1. for plan in plans] + [gripper.current_normalized + 1.]
        tx.prepare_targets('2', targets)
        targets[0] = 999.
        tx.commit('2')
        with self.assertRaisesRegex(ValueError, 'matching'):
            tx.commit('2')
        self.assertTrue(all(motor.torque == 1 for motor in motors.values()))
        self.assertEqual(motors[group.names[0]].writes[-1], ('goal', 2010))

    def test_group_keeps_the_full_goal_if_feedback_changes_before_commit(self):
        packets = []
        group, motors, clock = self.make_group(write_group_goals=packets.append)
        tx = ArmBusTransaction(group, timeout_s=.5, clock=lambda: clock[0])
        tx.prepare_hold('1')
        tx.commit('1')
        wrist = motors[group.names[4]]
        targets = [0., 0., 0., 0., -18., 0., 50.]
        tx.prepare_targets('2', targets)
        self.assertEqual(packets, [])
        wrist.position = 2070  # Now 25 units from the requested target.
        tx.commit('2')
        self.assertEqual(packets, [{wrist.joint_id: 1820}])
        self.assertEqual(group.controls[group.names[4]].target_ticks, 1820)
        self.assertEqual(wrist.torque, 1)

        targets[4] = -99.
        tx.prepare_targets('3', targets)
        tx.commit('3')
        self.assertEqual(packets[-1], {wrist.joint_id: 1010})

    def test_group_keeps_full_gripper_goal_after_fresh_feedback(self):
        packets = []
        group, motors, clock = self.make_group(write_group_goals=packets.append)
        tx = ArmBusTransaction(group, timeout_s=.5, clock=lambda: clock[0])
        tx.prepare_hold('1')
        tx.commit('1')
        gripper = motors[group.names[-1]]
        targets = [0.] * 6 + [43.]
        tx.prepare_targets('2', targets)
        gripper.position = 2100  # Current 55, so the requested step grew from 7 to 12.
        tx.commit('2')
        self.assertEqual(packets, [{7: 1860}])  # Position 43, exactly the requested goal.
        self.assertEqual(gripper.torque, 1)

    def test_streaming_uses_one_fresh_inspection_and_consumes_each_token_once(self):
        packets = MagicMock()
        group, motors, clock = self.make_group(write_group_goals=packets)
        tx = ArmBusTransaction(group, timeout_s=.5, clock=lambda: clock[0])
        with self.assertRaises(ValueError):
            tx.apply_stream_targets('1', [1.] * 6 + [51.], deadline=.5)
        self.assertTrue(all(not motor.writes for motor in motors.values()))
        group.prepare()
        group.start()
        original = group.read_group_snapshot
        group.read_group_snapshot = MagicMock(side_effect=original)
        tx.apply_stream_targets('2', [1.] * 6 + [51.], deadline=.5)
        group.read_group_snapshot.assert_called_once()
        packets.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'already used'):
            tx.apply_stream_targets('2', [1.] * 6 + [51.], deadline=.5)
        self.assertEqual(packets.call_count, 1)

    def test_streaming_rechecks_feedback_and_deadline_before_any_goal_write(self):
        for failure in ('expired_during_read', 'torque_lost'):
            packets = MagicMock()
            group, motors, clock = self.make_group(write_group_goals=packets)
            tx = ArmBusTransaction(group, timeout_s=.5, clock=lambda: clock[0])
            group.prepare()
            group.start()
            original = group.read_group_snapshot
            def snapshot():
                if failure == 'expired_during_read':
                    clock[0] = 1.
                else:
                    motors[group.names[0]].torque = 0
                return original()
            group.read_group_snapshot = snapshot
            with self.assertRaises(ValueError):
                tx.apply_stream_targets('1', [1.] * 6 + [51.], deadline=.5)
            packets.assert_not_called()

    def test_full_target_does_not_depend_on_the_feedback_velocity_direction(self):
        for position, velocity, target, expected in (
            (2000, 900, 30., 2300),
            (2000, -900, 30., 2300),
            (2700, 900, 98., 2980),
        ):
            with self.subTest(position=position, velocity=velocity):
                packets = []
                group, motors, clock = self.make_group(write_group_goals=packets.append)
                tx = ArmBusTransaction(group, timeout_s=.5, clock=lambda: clock[0])
                wrist = motors[group.names[4]]
                wrist.position = position
                tx.prepare_hold('1')
                tx.commit('1')
                tx.prepare_targets('2', [0., 0., 0., 0., target, 0., 50.])
                original_read = wrist.read_joint
                wrist.read_joint = lambda: dict(original_read(), velocity_raw=velocity, moving=True)
                tx.commit('2')
                self.assertEqual(packets, [{wrist.joint_id: expected}])

    def test_commit_does_not_reject_an_isolated_elbow_current_peak(self):
        packets = []
        group, motors, clock = self.make_group(write_group_goals=packets.append)
        tx = ArmBusTransaction(group, timeout_s=.5, clock=lambda: clock[0])
        tx.prepare_hold('1')
        tx.commit('1')
        elbow = motors[group.names[2]]
        tx.prepare_targets('2', [0., 0., 6., 0., 0., 0., 50.])
        original_read = elbow.read_joint
        elbow.read_joint = lambda: dict(original_read(), current_a=.312, load_raw=504, moving=True)
        tx.commit('2')
        self.assertEqual(packets, [{elbow.joint_id: 2060}])
        self.assertEqual(elbow.torque, 1)
        self.assertEqual(group.state, 'holding')


class TestContinuousJointProtection(unittest.TestCase):
    def make_control(self, joint='elbow_flex'):
        entry = fixture('right')[2]['motors']['arm_right_' + joint]
        motor = FakeJoint(entry)
        clock = [0.0]
        control = JointControl(motor, entry, clock=lambda: clock[0])
        return control, motor, clock

    def test_isolated_elbow_spike_does_not_reject_or_release(self):
        control, motor, clock = self.make_control()
        control.hold_current_position()
        read_joint = motor.read_joint
        motor.read_joint = lambda: dict(read_joint(), load_raw=384, current_a=0.208)
        self.assertEqual(control.inspect_absolute_target(10.).target_ticks, 2100)
        control.set_absolute_target(10.)
        self.assertIsNone(control.observe(motor.read_joint()))
        clock[0] = 1 / 30
        self.assertIsNone(control.observe(read_joint()))
        clock[0] = 0.7
        self.assertIsNone(control.observe(motor.read_joint()))
        self.assertEqual(motor.torque, 1)
        self.assertFalse(control.fault)

    def test_continuous_load_trips_only_if_the_joint_stalls(self):
        for moving in (False, True):
            with self.subTest(moving=moving):
                control, motor, clock = self.make_control()
                control.hold_current_position()
                control.set_absolute_target(20.)
                for frame in range(20):
                    clock[0] = frame / 30
                    motor.position = 2000 + (4 * frame if moving else 0)
                    event = control.observe(dict(motor.read_joint(), load_raw=-384, moving=moving))
                    if not moving and frame == 5:
                        self.assertEqual(event[0], 'warning')
                        self.assertIn('keeping the goal and torque', event[1])
                    else:
                        self.assertIsNone(event)
                    self.assertEqual(motor.torque, 1)
                clock[0] = 20 / 30
                if moving:
                    motor.position += 4
                event = control.observe(dict(motor.read_joint(), load_raw=-384, moving=moving))
                if moving:
                    self.assertIsNone(event)
                    self.assertEqual(motor.torque, 1)
                else:
                    self.assertEqual(event[0], 'error')
                    self.assertIn('high load without progress', event[1])
                    self.assertEqual(motor.torque, 0)

    def test_loaded_joint_at_its_target_can_hold_position(self):
        control, motor, clock = self.make_control()
        control.hold_current_position()
        for frame in range(31):
            clock[0] = frame / 30
            self.assertIsNone(control.observe(dict(motor.read_joint(), load_raw=548, current_a=.169)))
        self.assertEqual(motor.torque, 1)

    def test_high_load_timer_does_not_cross_a_feedback_gap(self):
        control, motor, clock = self.make_control()
        control.hold_current_position()
        control.set_absolute_target(40.)
        loaded = dict(motor.read_joint(), load_raw=548, moving=True)
        self.assertIsNone(control.observe(loaded))
        clock[0] = .70
        self.assertIsNone(control.observe(loaded))
        self.assertEqual(motor.torque, 1)

    def test_transient_feedback_loss_recovers_and_restarts_its_timer(self):
        control, motor, clock = self.make_control()
        control.hold_current_position()
        self.assertEqual(control.observe(None)[0], 'warning')
        clock[0] = .10
        self.assertIsNone(control.observe(motor.read_joint()))
        clock[0] = .15
        self.assertEqual(control.observe(None)[0], 'warning')
        clock[0] = .30
        self.assertIsNone(control.observe(None))
        self.assertEqual(motor.torque, 1)

    def test_near_stall_current_requires_continuous_samples(self):
        control, motor, clock = self.make_control()
        control.hold_current_position()
        high = dict(motor.read_joint(), current_a=-7.845)
        self.assertIsNone(control.observe(high))
        clock[0] = .022
        self.assertIsNone(control.observe(motor.read_joint()))
        for timestamp in (.200, .300, .322, .344, .366):
            clock[0] = timestamp
            self.assertIsNone(control.observe(high))
        clock[0] = .388
        result = control.observe(high)
        self.assertEqual(result[0], 'error')
        self.assertIn('near-stall current', result[1])
        self.assertIn('limit 7.84 A', result[1])
        self.assertEqual(motor.torque, 0)

    def test_continuous_current_limits_depend_on_servo_model(self):
        for joint, current, expected_limit in (
            ('elbow_flex', 4.401, '4.4'),
            ('wrist_yaw', 2.801, '2.8'),
        ):
            with self.subTest(joint=joint):
                control, motor, clock = self.make_control(joint)
                control.hold_current_position()
                below = dict(motor.read_joint(), current_a=current - .02)
                high = dict(motor.read_joint(), current_a=current)
                for frame in range(34):
                    clock[0] = frame * .020
                    self.assertIsNone(control.observe(below))
                for frame in range(33):
                    clock[0] = 1.0 + frame * .020
                    self.assertIsNone(control.observe(high))
                clock[0] = 1.66
                result = control.observe(high)
                self.assertEqual(result[0], 'error')
                self.assertIn(f'limit {expected_limit} A', result[1])
                self.assertEqual(motor.torque, 0)

    def test_gripper_contact_keeps_torque_and_reduces_closing_goal(self):
        control, motor, clock = self.make_control('gripper')
        control.hold_current_position()
        control.set_absolute_target(40.)
        loaded = dict(motor.read_joint(), load_raw=384, current_a=.312, moving=True)
        for frame in range(5):
            clock[0] = frame / 30
            event = control.observe(loaded)
        self.assertIsNotNone(event)
        self.assertTrue(control.gripper_contact_latched)
        self.assertEqual(motor.torque, 1)
        self.assertEqual(motor.writes[-1], ('goal', control.gripper_contact_goal_ticks))

    def start_wrist_replay(self):
        control, motor, clock = self.make_control('wrist_yaw')
        motor.position = 1845
        control.hold_current_position()
        self.assertEqual(control.set_absolute_target(-18.3).target_ticks, 1817)
        return control, motor, clock

    def test_wrist_one_frame_overshoot_can_recover(self):
        control, motor, clock = self.start_wrist_replay()
        sample = dict(motor.read_joint(), moving=True, velocity_raw=-1150)
        self.assertIsNone(control.observe(dict(sample, position_ticks=1838)))
        clock[0] = 1 / 30
        self.assertIsNone(control.observe(dict(sample, position_ticks=1801, velocity_raw=-1100)))
        clock[0] = 2 / 30
        self.assertIsNone(control.observe(dict(sample, position_ticks=1780, velocity_raw=-650)))
        clock[0] = 3 / 30
        self.assertIsNone(control.observe(dict(sample, position_ticks=1812, velocity_raw=150)))
        self.assertEqual(motor.torque, 1)
        self.assertFalse(control.fault)

    def test_invalid_feedback_still_trips(self):
        for scenario in ('out_of_range', 'lost_feedback'):
            with self.subTest(scenario=scenario):
                control, motor, clock = self.start_wrist_replay()
                sample = dict(motor.read_joint(), position_ticks=1801, moving=True, velocity_raw=-1100)
                if scenario == 'out_of_range':
                    sample['position_ticks'] = 983
                elif scenario == 'lost_feedback':
                    sample = None
                event = control.observe(sample)
                if scenario == 'lost_feedback':
                    self.assertEqual(event[0], 'warning')
                    self.assertEqual(motor.torque, 1)
                    clock[0] += .21
                    event = control.observe(None)
                self.assertEqual(event[0], 'error')
                self.assertEqual(motor.torque, 0)

    def test_reversal_uses_calibrated_travel_not_new_goal_distance(self):
        control, motor, clock = self.make_control('wrist_flex')
        motor.position = 1376
        control.hold_current_position()
        self.assertEqual(control.set_absolute_target(-61.6).target_ticks, 1384)
        # The previous move was still braking when a new, nearby goal arrived.
        for frame, (position, velocity) in enumerate(
                [(1349, -1100), (1312, -1100), (1285, -750), (1271, -550)]):
            clock[0] = frame / 30
            sample = dict(motor.read_joint(), moving=True, position_ticks=position,
                          velocity_raw=velocity)
            self.assertIsNone(control.observe(sample))
        self.assertEqual(motor.torque, 1)


class TestJointControl(unittest.TestCase):
    def test_temperature_confirmation_survives_two_high_readings_between_low_values(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_yaw']
        motor = FakeJoint(entry)
        motor.read_temperature = MagicMock(side_effect=[22, 150, 22])
        read_joint = motor.read_joint
        motor.read_joint = MagicMock(side_effect=[
            dict(read_joint(), temperature_c=value) for value in (150., 22., 22.)
        ])
        report = MagicMock()
        control = JointControl(motor, entry,
                               report_temperature_recheck=report)
        sample = dict(read_joint(), temperature_c=62.)
        with patch('xyc_hardware.arm_joint_control.time.sleep') as pause:
            control._validate_motion_sample(sample)
        self.assertEqual(pause.call_count, 5)
        self.assertTrue(all(call.args == (.01,) for call in pause.call_args_list))
        self.assertEqual(motor.read_temperature.call_count, 3)
        self.assertEqual(motor.read_joint.call_count, 3)
        self.assertIn('[22, 150.0, 150, 22.0, 22, 22.0]', report.call_args.args[0])
        self.assertEqual(motor.writes, [])

    def test_temperature_confirmation_rejects_persistent_heat_or_unstable_feedback(self):
        for direct, full in (([62, 62, 62], [62., 62., 62.]),
                             ([22, 22, 22], [150., 150., 150.])):
            with self.subTest(direct=direct, full=full):
                entry = fixture('right')[2]['motors']['arm_right_wrist_yaw']
                motor = FakeJoint(entry)
                read_joint = motor.read_joint
                motor.read_temperature = MagicMock(side_effect=direct)
                motor.read_joint = MagicMock(side_effect=[
                    dict(read_joint(), temperature_c=value) for value in full
                ])
                report = MagicMock()
                control = JointControl(motor, entry,
                                       report_temperature_recheck=report)
                with patch('xyc_hardware.arm_joint_control.time.sleep'), self.assertRaisesRegex(
                        ValueError, 'temperature alarm not cleared'):
                    control._validate_motion_sample(dict(read_joint(), temperature_c=62.))
                self.assertEqual(motor.read_temperature.call_count, 3)
                self.assertEqual(motor.read_joint.call_count, 3)
                report.assert_not_called()
                self.assertEqual(motor.writes, [])

    def test_explicit_hold_primes_current_position_then_enables_once(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        control = JointControl(motor, entry)
        plan = control.hold_current_position()
        self.assertEqual(plan.target_ticks, 2000)
        self.assertEqual(motor.writes, [('goal', 2000), ('torque', True)])
        self.assertTrue(control.holding)
        self.assertTrue(control.owned)
        self.assertEqual(motor.torque, 1)

    def test_explicit_hold_rejects_unsafe_initial_state_without_writes(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        motor.torque = 1
        with self.assertRaises(ValueError):
            JointControl(motor, entry).hold_current_position()
        self.assertEqual(motor.writes, [])
        motor = FakeJoint(entry)
        motor.position = 500
        with self.assertRaises(ValueError):
            JointControl(motor, entry).hold_current_position()
        self.assertEqual(motor.writes, [])

    def test_hold_enable_ack_failure_attempts_release_and_latches(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        motor.fail_enable = True
        control = JointControl(motor, entry)
        with self.assertRaises(OSError):
            control.hold_current_position()
        self.assertEqual(motor.writes[-1], ('torque', False))
        self.assertEqual(motor.torque, 0)
        self.assertTrue(control.fault)

    def test_hold_enable_ack_and_release_failure_remain_unconfirmed(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        motor.fail_enable = True
        motor.fail_release = True
        control = JointControl(motor, entry)
        with self.assertRaisesRegex(RuntimeError, 'torque-off unconfirmed'):
            control.hold_current_position()
        self.assertTrue(control.owned)
        self.assertTrue(control.fault)

    def test_absolute_updates_keep_torque_and_reject_invalid_goals(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        control = JointControl(motor, entry)
        control.hold_current_position()
        before = list(motor.writes)
        self.assertEqual(control.set_absolute_target(0.).target_ticks, 2000)
        self.assertEqual(motor.writes, before)
        self.assertEqual(control.set_absolute_target(2.).target_ticks, 2020)
        self.assertEqual(motor.writes[-1], ('goal', 2020))
        before = list(motor.writes)
        for invalid in (float('nan'), float('inf'), True, 100.01):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                control.set_absolute_target(invalid)
        self.assertEqual(motor.writes, before)
        self.assertEqual(motor.torque, 1)

    def test_absolute_current_target_replaces_an_in_progress_goal(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        control = JointControl(motor, entry)
        control.hold_current_position()
        control.set_absolute_target(2.)
        self.assertEqual(motor.writes[-1], ('goal', 2020))
        control.set_absolute_target(0.)
        self.assertEqual(motor.writes[-1], ('goal', 2000))
        self.assertEqual(motor.torque, 1)

    def test_repeated_absolute_target_does_not_rewrite_goal_while_moving(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        control = JointControl(motor, entry)
        control.hold_current_position()
        control.set_absolute_target(2.)
        before = list(motor.writes)
        self.assertTrue(control.busy)
        self.assertEqual(control.set_absolute_target(2.).target_ticks, 2020)
        self.assertEqual(motor.writes, before)
        motor.torque = 0
        with self.assertRaisesRegex(ValueError, 'torque unexpectedly disabled'):
            control.inspect_absolute_target(2.)
        self.assertEqual(motor.writes, before)

    def test_full_calibrated_range_is_available_for_absolute_targets(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        motor.position = 1002
        control = JointControl(motor, entry)
        control.hold_current_position()
        before = list(motor.writes)
        current = control.inspect_absolute_target(-99.8).current_normalized
        control.set_absolute_target(current)
        self.assertEqual(motor.writes, before)
        control.set_absolute_target(-99.9)
        self.assertEqual(motor.writes[-1], ('goal', 1001))
        self.assertEqual(control.set_absolute_target(-99.).target_ticks, 1010)
        self.assertEqual(motor.torque, 1)

    def test_group_can_hold_unchanged_joint_at_boundary_margin(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        motor.position = 2992
        control = JointControl(motor, entry)
        plan = control.hold_current_position()
        before = list(motor.writes)
        control.inspect_absolute_target(plan.current_normalized)
        control.set_absolute_target(plan.current_normalized)
        self.assertEqual(motor.writes, before)
        control.set_absolute_target(99.9)
        self.assertEqual(motor.writes[-1], ('goal', 2999))

    def test_persistent_hold_updates_without_releasing_or_reenabling(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        now = [0.]
        control = JointControl(motor, entry, clock=lambda: now[0])
        control.hold_current_position()
        control.set_absolute_target(2.)
        motor.position = 2020
        for _ in range(3):
            control.observe(motor.read_joint())
        self.assertTrue(control.holding)
        self.assertEqual(control.last_outcome, 'holding')
        self.assertIn('保持上力', control.last_message)
        self.assertEqual(motor.torque, 1)
        now[0] = 10.
        self.assertIsNone(control.observe(motor.read_joint()))
        plan = control.set_absolute_target(4.)
        self.assertEqual(plan.target_ticks, 2040)
        self.assertEqual(sum(write == ('torque', True) for write in motor.writes), 1)
        self.assertNotIn(('torque', False), motor.writes)
        motor.position = 2030
        control.set_absolute_target(5.)
        self.assertEqual(motor.writes[-1], ('goal', 2050))
        self.assertNotIn(('torque', False), motor.writes)

    def test_persistent_hold_fault_releases_and_latches(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        now = [0.]
        control = JointControl(motor, entry, clock=lambda: now[0])
        control.hold_current_position()
        control.set_absolute_target(2.)
        sample = motor.read_joint()
        sample['current_a'] = 3.5
        self.assertIsNone(control.observe(sample))
        now[0] = .05
        self.assertIsNone(control.observe(sample))
        now[0] = .09
        result = control.observe(sample)
        self.assertEqual(result[0], 'error')
        self.assertEqual(motor.writes[-1], ('torque', False))
        self.assertTrue(control.fault)
        with self.assertRaises(ValueError):
            control.set_absolute_target(2.)

    def test_holding_deviation_warns_after_three_samples_and_can_recover(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        now = [0.]
        control = JointControl(motor, entry, clock=lambda: now[0])
        control.hold_current_position()
        for i in range(35):
            now[0] = i / 30.
            sample = motor.read_joint()
            sample['voltage_v'] = 12.1
            self.assertIsNone(control.observe(sample))
        sample = motor.read_joint()
        sample['position_ticks'] += 13
        sample['voltage_v'] = 11.9
        self.assertIsNone(control.observe(sample))
        sample['position_ticks'] = 2031
        self.assertIsNone(control.observe(sample))
        self.assertIsNone(control.observe(sample))
        level, message = control.observe(sample)
        self.assertEqual(level, 'warning')
        self.assertEqual(control.fault, '')
        self.assertEqual(motor.torque, 1)
        diagnostic = json.loads(message.split('; diagnostic=', 1)[1])
        self.assertEqual(diagnostic['target_ticks'], 2000)
        self.assertEqual(len(control.recent_feedback), 30)
        self.assertEqual(diagnostic['latest_feedback']['sample']['position_ticks'], 2031)
        self.assertEqual(diagnostic['latest_feedback']['sample']['voltage_v'], 11.9)
        self.assertLess(len(message), 1000)
        self.assertIsNone(control.observe(sample))
        sample['position_ticks'] = 2005
        self.assertEqual(control.observe(sample)[0], 'info')
        self.assertEqual(motor.torque, 1)
        self.assertNotIn(('torque', False), motor.writes)

    def test_held_joint_near_calibration_endpoint_keeps_target_on_small_excursion(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        motor.position = 2998
        control = JointControl(motor, entry)
        control.hold_current_position()
        sample = motor.read_joint()
        sample['position_ticks'] = 3003  # Outside the hand-recorded +100 bound.
        self.assertIsNone(control.observe(sample))
        self.assertEqual(control.target_ticks, 2998)
        self.assertEqual(motor.torque, 1)

    def test_repeated_updates_report_stall_without_dropping_persistent_hold(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        now = [0.]
        control = JointControl(motor, entry, clock=lambda: now[0])
        control.hold_current_position()
        control.set_absolute_target(2.)
        now[0] = 1.
        control.set_absolute_target(2.)
        now[0] = 1.6
        result = control.observe(motor.read_joint())
        self.assertEqual(result[0], 'warning')
        self.assertIn('not progressed', result[1])
        self.assertEqual(motor.torque, 1)
        self.assertTrue(control.busy)
        now[0] = 3.2
        self.assertIsNone(control.observe(motor.read_joint()))
        motor.position = control.target_ticks
        for _ in range(2):
            self.assertIsNone(control.observe(motor.read_joint()))
        self.assertEqual(control.observe(motor.read_joint())[0], 'info')
        self.assertTrue(control.holding)
        self.assertEqual(motor.torque, 1)

    def test_owned_goal_write_failure_attempts_protective_release(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        control = JointControl(motor, entry)
        control.hold_current_position()
        control.set_absolute_target(2.)
        motor.fail_goal_number = 3
        with self.assertRaises(OSError):
            control.set_absolute_target(4.)
        self.assertEqual(motor.writes[-1], ('torque', False))
        self.assertTrue(control.fault)
        with self.assertRaises(ValueError):
            control.set_absolute_target(2.)

    def test_owned_goal_failure_reports_unconfirmed_release(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        control = JointControl(motor, entry)
        control.hold_current_position()
        control.set_absolute_target(2.)
        motor.fail_goal_number = 3
        motor.fail_release = True
        with self.assertRaisesRegex(RuntimeError, 'torque-off unconfirmed'):
            control.set_absolute_target(4.)
        self.assertTrue(control.owned)
        self.assertTrue(control.fault)

    def test_all_arm_joints_use_their_own_calibration_and_selected_id(self):
        for arm in ('left', 'right'):
            _, _, document = fixture(arm)
            for name in arm_names(arm):
                with self.subTest(joint=name):
                    entry = document['motors'][name]
                    motor = FakeJoint(entry)
                    controller = JointControl(motor, entry)
                    self.assertEqual(motor.writes, [])
                    controller.hold_current_position()
                    plan = controller.set_absolute_target(normalize_tick(entry, motor.position) + 1.)
                    expected_ticks = 2020 if name.endswith('_gripper') else 2010
                    self.assertEqual(plan.target_ticks, expected_ticks)
                    expected = [('torque_limit', 100)] if name.endswith('_gripper') else []
                    expected += [('goal', 2000), ('torque', True), ('goal', expected_ticks)]
                    self.assertEqual(motor.writes, expected)
                    sample = motor.read_joint()
                    sample['position_ticks'] = expected_ticks
                    controller.observe(sample)
                    controller.observe(sample)
                    self.assertEqual(controller.observe(sample)[0], 'info')
                    self.assertEqual(motor.torque, 1)
                    controller.release()
                    self.assertEqual(motor.writes[-1], ('torque', False))

    def test_failed_startup_goal_never_enables_torque(self):
        entry = fixture('right')[2]['motors']['arm_right_wrist_roll']
        motor = FakeJoint(entry)
        motor.fail_goal_number = 1
        control = JointControl(motor, entry)
        with self.assertRaises(OSError):
            control.hold_current_position()
        self.assertFalse(control.owned)
        self.assertEqual(motor.torque, 0)
        self.assertNotIn(('torque', True), motor.writes)

    def test_feedback_fault_keeps_unconfirmed_release_visible_and_blocks_new_targets(self):
        entry = fixture('left')[2]['motors']['arm_left_wrist_roll']
        motor = FakeJoint(entry)
        clock = [0.]
        control = JointControl(motor, entry, clock=lambda: clock[0])
        control.hold_current_position()
        control.set_absolute_target(1.)
        motor.fail_release = True
        self.assertEqual(control.observe(None)[0], 'warning')
        clock[0] = .21
        self.assertIn('UNCONFIRMED', control.observe(None)[1])
        self.assertTrue(control.owned)
        with self.assertRaises(ValueError):
            control.set_absolute_target(-1.)

    def test_transport_only_permits_selected_joint_and_profile(self):
        for bus in ('left', 'right'):
            for joint_id in range(1, 8):
                serial = MagicMock()
                serial.write.side_effect = len
                ids = list(range(1, 12 if bus == 'left' else 8))
                with patch('xyc_hardware.feetech_reader.serial.Serial', return_value=serial):
                    driver = JointDriver('/unused', ids, bus=bus, joint_id=joint_id)
                for other in ({1, 2, 6, 7, 8, 11, 254} - {joint_id}):
                    with self.assertRaises(ValueError):
                        driver._request(other, 3, b'\x28\x01')
                with self.assertRaises(ValueError):
                    driver._request(joint_id, 3, b'\x1f\x00\x00')
                self.assertEqual(serial.write.call_count, 0)
                speed, acceleration = driver.motion_profile(joint_id)
                profile = bytes([acceleration]) + (2010).to_bytes(2, 'little') + b'\x00\x00' + speed.to_bytes(2, 'little')
                driver._request(joint_id, 3, bytes([41]) + profile)
                packet = serial.write.call_args.args[0]
                self.assertEqual(packet[2], joint_id)
                self.assertEqual(packet[-1], (~sum(packet[2:-1])) & 255)

    def test_failed_profile_readback_is_not_accepted(self):
        serial = MagicMock()
        serial.write.side_effect = len
        with patch('xyc_hardware.feetech_reader.serial.Serial', return_value=serial):
            driver = JointDriver('/unused', list(range(1, 8)), bus='right', joint_id=5)
        with patch.object(driver, '_receive', return_value={}), \
             patch.object(driver, '_read_joint_register', return_value=b'bad'):
            with self.assertRaises(ConnectionError):
                driver.write_goal(2100)

    def test_missing_register_reply_retries_reads_without_repeating_a_write(self):
        serial = MagicMock()
        serial.write.side_effect = len
        with patch('xyc_hardware.feetech_reader.serial.Serial', return_value=serial):
            driver = JointDriver('/unused', list(range(1, 8)), bus='right', joint_id=2)
        reply = SimpleNamespace(servo_error=0, data=b'\x01')
        with patch.object(driver, '_receive', side_effect=[{}, {2: reply}]):
            self.assertEqual(driver.read_torque(), 1)
        packets = [call.args[0] for call in serial.write.call_args_list]
        self.assertEqual(len(packets), 2)
        self.assertTrue(all(packet[4:7] == b'\x02\x28\x01' for packet in packets))
        serial.write.reset_mock()
        reply.servo_error = 1
        with patch.object(driver, '_receive', return_value={2: reply}):
            with self.assertRaisesRegex(ConnectionError, 'right ID 2.*servo error'):
                driver.read_torque()
        self.assertEqual(serial.write.call_count, 1)
        serial.write.reset_mock()
        with patch.object(driver, '_receive', return_value={}):
            with self.assertRaisesRegex(ConnectionError, 'after 3 reads'):
                driver.read_torque()
        self.assertEqual(serial.write.call_count, 3)

    def test_group_preflight_retries_transmit_timeout_without_writing_goals(self):
        port = MagicMock()
        port.write.side_effect = [serial.SerialTimeoutException('Write timeout'), 15]
        with patch('xyc_hardware.feetech_reader.serial.Serial', return_value=port):
            driver = JointDriver('/unused', list(range(1, 8)), bus='right', joint_id=2)
        data = bytearray(40)
        data[9] = 1
        data[25:27] = (2000).to_bytes(2, 'little')
        data[31:33] = bytes([120, 22])
        replies = {i: SimpleNamespace(servo_error=0, data=bytes(data)) for i in range(1, 8)}
        with patch.object(driver, '_receive', return_value=replies):
            self.assertEqual(len(driver.read_arm_control_snapshot()), 7)
        self.assertEqual(port.write.call_count, 2)
        self.assertEqual(port.write.call_args_list[0], port.write.call_args_list[1])
        self.assertEqual(port.write.call_args.args[0][4], 0x82)

    def test_persistent_preflight_failure_still_fails_and_goal_write_is_never_replayed(self):
        for instruction, attempts in ((0x82, 2), (0x83, 1), (3, 1)):
            port = MagicMock()
            port.write.side_effect = serial.SerialTimeoutException('Write timeout')
            with patch('xyc_hardware.feetech_reader.serial.Serial', return_value=port):
                driver = JointDriver('/unused', list(range(1, 8)), bus='right', joint_id=2)
            with self.assertRaisesRegex(serial.SerialTimeoutException, 'right ID.*transmission timed out'):
                driver._send_joint_packet(2, instruction, b'\x28\x01')
            self.assertEqual(port.write.call_count, attempts)
    def test_group_packets_and_target_readback_use_continuous_profile(self):
        serial = MagicMock()
        serial.write.side_effect = len
        with patch('xyc_hardware.feetech_reader.serial.Serial', return_value=serial):
            driver = JointDriver('/unused', list(range(1, 12)), bus='left', joint_id=1)
        with self.assertRaises(ValueError):
            driver.write_arm_goals({8: 2000})
        serial.write.assert_not_called()
        replies = {1: SimpleNamespace(servo_error=0, data=(2000).to_bytes(2, 'little')),
                   7: SimpleNamespace(servo_error=0, data=(2100).to_bytes(2, 'little'))}
        with patch.object(driver, '_receive', return_value=replies):
            driver.write_arm_goals({7: 2100, 1: 2000})
        write, read = [call.args[0] for call in serial.write.call_args_list]
        self.assertEqual(write[2:-1], bytes([254, 10, 0x83, 42, 2, 1, 208, 7, 7, 52, 8]))
        self.assertEqual(read[2:-1], bytes([254, 6, 0x82, 42, 2, 1, 7]))
        for packet in (write, read):
            self.assertEqual(packet[-1], (~sum(packet[2:-1])) & 255)
        with patch.object(driver, '_receive', return_value={1: replies[1]}):
            with self.assertRaisesRegex(ConnectionError, 'left IDs \\[7\\]'):
                driver.write_arm_goals({1: 2000, 7: 2100})


if __name__ == '__main__':
    unittest.main()
