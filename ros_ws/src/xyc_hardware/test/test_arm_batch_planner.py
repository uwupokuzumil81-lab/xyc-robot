# SPDX-License-Identifier: Apache-2.0
"""Two-arm planning tests with fake feedback; no motion is executed."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from xyc_hardware.arm_batch_planner import ArmBatchPlanner, ArmCommandPreflight, JointSample
from xyc_hardware.arm_calibration import arm_names
from xyc_hardware.command_contract import COMMAND_SCHEMA
from xyc_hardware.teleop_control import FollowTargets
from test_arm_calibration import fixture


class TestCompleteTeleopTargets(unittest.TestCase):
    def make_planner(self, *, absolute=True):
        follower = [0.] * 6 + [50.] + [0.] * 6 + [50.]
        leader = {'left': list(follower[:7]), 'right': list(follower[7:])}
        return FollowTargets(leader, follower, absolute=absolute), leader

    def test_large_joint_and_gripper_goals_are_not_shortened(self):
        planner, leader = self.make_planner()
        leader['left'] = [80., -90., 99.36, -75., 98., -98., 100.]
        leader['right'] = [-80., 90., -99.36, 75., -98., 98., 0.]
        targets = planner.update(leader)
        self.assertEqual(targets, leader['left'] + leader['right'])
        self.assertEqual(planner.reference, targets)

    def test_rapid_reversal_and_unchanged_goal_preserve_latest_request(self):
        planner, leader = self.make_planner()
        for value in (80., -80., 100., -100., 20., 20.):
            leader['left'][5] = value
            self.assertEqual(planner.update(leader)[5], value)

    def test_relative_mapping_preserves_offset_and_rejects_overrun_without_clipping(self):
        follower = [10.] * 6 + [50.] + [0.] * 6 + [50.]
        leader = {'left': [0.] * 6 + [50.]}
        planner = FollowTargets(leader, follower)
        leader['left'][0] = 80.
        self.assertEqual(planner.update(leader)[0], 90.)
        previous = list(planner.previous)
        leader['left'][0] = 100.
        with self.assertRaises(ValueError): planner.update(leader)
        self.assertEqual(planner.previous, previous)

    def test_invalid_goal_rejects_whole_batch_and_preserves_previous_target(self):
        planner, leader = self.make_planner()
        before = list(planner.previous)
        for invalid in (100.01, -100.01, float('nan'), float('inf')):
            leader['right'][3] = invalid
            with self.assertRaises(ValueError): planner.update(leader)
            self.assertEqual(planner.previous, before)

    def test_other_arm_keeps_its_goal_and_returned_list_does_not_alias_state(self):
        follower = [0.] * 6 + [50.] + [7.] * 6 + [50.]
        leader = {'left': [0.] * 6 + [50.]}
        planner = FollowTargets(leader, follower, absolute=True)
        targets = planner.update(leader)
        self.assertEqual(targets[7:], follower[7:])
        targets[0] = 90.
        self.assertEqual(planner.previous[0], 0.)
        self.assertEqual(leader['left'][0], 0.)

    def test_leader_sides_cannot_change_mid_session(self):
        planner, leader = self.make_planner()
        del leader['right']
        with self.assertRaises(ValueError): planner.update(leader)


class TestArmBatchPlanner(unittest.TestCase):
    def setUp(self):
        self.docs = {side: fixture(side)[2] for side in ('left', 'right')}
        self.names = tuple(n for side in ('left', 'right') for n in arm_names(side))
        self.planner = ArmBatchPlanner(self.docs, max_age_s=.2, max_skew_s=.05, future_tolerance_s=.01)
        self.now = 100_000_000_000
        self.samples = []
        for index, name in enumerate(self.names):
            side = 'left' if index < 7 else 'right'
            entry = self.docs[side]['motors'][name]
            self.samples.append(JointSample(name, side, entry['id'], entry['model_number'],
                                            2000, self.now - 10_000_000, True, 0, ''))
        self.targets = [1.] * 14
        self.targets[6] = self.targets[13] = 51.

    def plan(self):
        return self.planner.plan(targets=self.targets, samples=self.samples,
                                 calibration_ids=self.planner.ids, now_ns=self.now)

    def test_fixed_name_order_even_with_reordered_feedback(self):
        self.samples.reverse()
        targets = self.plan()
        self.assertEqual(tuple(item.name for item in targets), self.names)
        self.assertEqual([item.bus for item in targets], ['left'] * 7 + ['right'] * 7)
        self.assertEqual(targets[0].target_ticks, 2010)
        self.assertEqual(targets[6].target_ticks, 2020)

    def test_any_bad_joint_rejects_entire_batch(self):
        mutations = [dict(valid=False), dict(servo_error=1), dict(model_number=777),
                     dict(bus='left'), dict(stamp_ns=self.now - 300_000_000)]
        for mutation in mutations:
            original = self.samples[8]
            self.samples[8] = replace(original, **mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.plan()
            self.samples[8] = original
        self.samples[8] = self.samples[0]
        with self.assertRaises(ValueError):
            self.plan()

    def test_out_of_range_nonfinite_step_or_stale_calibration_rejects(self):
        for index, value in ((3, float('nan')), (6, -1.), (5, 100.01)):
            original = self.targets[index]
            self.targets[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.plan()
            self.targets[index] = original
        with self.assertRaises(ValueError):
            self.planner.plan(targets=self.targets, samples=self.samples,
                              calibration_ids=['stale', self.planner.ids[1]], now_ns=self.now)

    def test_cross_arm_skew_is_checked_and_explicit_limits_required(self):
        self.samples[0] = replace(self.samples[0], stamp_ns=self.now - 70_000_000)
        with self.assertRaisesRegex(ValueError, 'skew'):
            self.plan()
        with self.assertRaises(ValueError):
            ArmBatchPlanner(self.docs, max_age_s=-1., max_skew_s=.05, future_tolerance_s=.01)

    def test_full_calibrated_endpoint_targets_are_preserved(self):
        self.samples[0] = replace(self.samples[0], position_ticks=1002)
        self.targets[0] = -99.8
        self.assertEqual(self.plan()[0].target_ticks, 1002)
        self.targets[0] = -99.
        self.assertEqual(self.plan()[0].target_ticks, 1010)
        self.targets[0] = -99.9
        self.plan()

    def test_endpoint_feedback_preserves_raw_ticks_and_allows_inward_target(self):
        for tick, target, expected in ((984, -99.5, 1005), (3016, 99.5, 2995)):
            with self.subTest(tick=tick):
                self.samples[0] = replace(self.samples[0], position_ticks=tick)
                self.targets[0] = target
                plan = self.plan()[0]
                self.assertEqual(plan.initial_ticks, tick)
                self.assertEqual(plan.target_ticks, expected)
        for tick in (983, 3017):
            self.samples[0] = replace(self.samples[0], position_ticks=tick)
            with self.subTest(tick=tick), self.assertRaisesRegex(ValueError, 'outside'):
                self.plan()

    def test_joint_and_gripper_endpoints_are_not_inset(self):
        self.samples[0] = replace(self.samples[0], position_ticks=2998)
        self.targets[0] = 99.8
        self.samples[6] = replace(self.samples[6], position_ticks=1002)
        self.targets[6] = .1
        plan = self.plan()
        self.assertEqual((plan[0].target_ticks, plan[6].target_ticks), (2998, 1002))
        self.targets[0] = 99.
        self.targets[6] = .5
        plan = self.plan()
        self.assertEqual((plan[0].target_ticks, plan[6].target_ticks), (2990, 1010))
        self.targets[0] = 99.9
        self.plan()
        self.targets[0] = 99.
        self.targets[6] = .05
        self.plan()

    def test_caller_cannot_change_calibration_after_planner_creation(self):
        before = self.plan()
        self.docs['left']['motors'][self.names[0]]['reference_tick'] = 1000
        after = self.plan()
        self.assertEqual(before, after)


class TestArmCommandPreflight(unittest.TestCase):
    def setUp(self):
        fixture_case = TestArmBatchPlanner()
        fixture_case.setUp()
        self.planner = fixture_case.planner
        self.now = fixture_case.now
        self.preflight = ArmCommandPreflight(self.planner, max_lifetime_s=.5, future_tolerance_s=.01)
        self.feedback = {'left': SimpleNamespace(motors=[]), 'right': SimpleNamespace(motors=[])}
        for sample in fixture_case.samples:
            stamp = SimpleNamespace(sec=sample.stamp_ns // 1_000_000_000,
                                    nanosec=sample.stamp_ns % 1_000_000_000)
            item = SimpleNamespace(name=sample.name, bus=sample.bus, motor_id=sample.motor_id,
                                   model_number=sample.model_number, position_ticks=sample.position_ticks,
                                   header=SimpleNamespace(stamp=stamp), valid=sample.valid,
                                   servo_error=sample.servo_error, error=sample.error)
            self.feedback[sample.bus].motors.append(item)
        self.status = {'left': {}, 'right': {}}
        self.command = SimpleNamespace(
            schema_id=COMMAND_SCHEMA, source_id='teleop', sequence=1,
            header=SimpleNamespace(stamp=SimpleNamespace(sec=100, nanosec=0)),
            valid_for=SimpleNamespace(sec=0, nanosec=200_000_000),
            arm_calibration_ids=list(self.planner.ids), arm_position_normalized=fixture_case.targets,
        )

    def check(self):
        return self.preflight.check(self.command, self.feedback, self.status, now_ns=self.now)

    def test_real_feedback_structure_and_sequence(self):
        result = self.check()
        self.assertEqual(len(result), 14)
        self.assertEqual([entry.bus for entry in result], ['left'] * 7 + ['right'] * 7)
        with self.assertRaisesRegex(ValueError, 'sequence'):
            self.check()

    def test_controller_fault_and_missing_feedback_fail_closed(self):
        self.status['right'] = {'arm_right_wrist_roll': {
            'phase': 'fault', 'owned': False, 'fault': 'stalled'}}
        with self.assertRaisesRegex(ValueError, 'stalled'):
            self.check()
        self.status['right'] = {}
        self.feedback['right'] = None
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.check()
        self.assertEqual(self.preflight.last_sequence, {})

    def test_stale_command_calibration_and_sample_fail(self):
        self.command.header.stamp.sec = 99
        with self.assertRaisesRegex(ValueError, 'expired'):
            self.check()
        self.command.header.stamp.sec = 100
        self.command.arm_calibration_ids.reverse()
        with self.assertRaisesRegex(ValueError, 'Calibration'):
            self.check()
        self.command.arm_calibration_ids.reverse()
        self.feedback['right'].motors[0].header.stamp.sec = 98
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.check()
        self.assertEqual(self.preflight.last_sequence, {})

    def test_wrong_bus_duplicate_and_invalid_status_fail(self):
        item = self.feedback['right'].motors[0]
        item.bus = 'left'
        with self.assertRaisesRegex(ValueError, 'wrong bus'):
            self.check()
        item.bus = 'right'
        self.feedback['right'].motors.append(item)
        with self.assertRaises(ValueError):
            self.check()
        self.feedback['right'].motors.pop()
        self.status['left'] = {'arm_left_wrist_roll': {'phase': 'holding', 'owned': False, 'fault': ''}}
        with self.assertRaisesRegex(ValueError, 'ownership'):
            self.check()


if __name__ == '__main__':
    unittest.main()
