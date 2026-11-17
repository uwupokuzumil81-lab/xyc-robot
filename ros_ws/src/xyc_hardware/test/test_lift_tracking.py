# SPDX-License-Identifier: Apache-2.0
"""Lift maths use simulated reference and geometry; no motor I/O."""
import unittest

from xyc_hardware.lift_tracking import LiftGeometry, LiftTracker


class TestLiftTracking(unittest.TestCase):
    def geometry(self, direction=1):
        return LiftGeometry(.1, direction, 1000., .5, -.4, .6)

    def test_wraparound_retains_continuous_height(self):
        tracker = LiftTracker(self.geometry(), reference_id='test-session', zero_tick=4090, reference_time_s=1.)
        height = tracker.update(5, 1.02)
        self.assertAlmostEqual(height, 11 * .1 / 4096)
        height = tracker.update(4090, 1.04)
        self.assertAlmostEqual(height, 0.)
        self.assertEqual(tracker.extended_delta_ticks, 0)

    def test_direction_and_target_ticks_follow_verified_reference(self):
        tracker = LiftTracker(self.geometry(-1), reference_id='test-session', zero_tick=1000, reference_time_s=1.)
        self.assertAlmostEqual(tracker.update(990, 1.02), 10 * .1 / 4096)
        self.assertEqual(tracker.height_to_delta_ticks(.1), -4096)
        with self.assertRaises(ValueError):
            tracker.height_to_delta_ticks(.61)

    def test_lost_feedback_latches_invalid_and_requires_new_reference(self):
        for raw, timestamp in ((1000, 2.), (1300, 1.02), (1000, .5), (4096, 1.02)):
            tracker = LiftTracker(self.geometry(), reference_id='test-session', zero_tick=1000, reference_time_s=1.)
            with self.subTest(raw=raw, timestamp=timestamp), self.assertRaises(ValueError):
                tracker.update(raw, timestamp)
            self.assertFalse(tracker.valid)
            with self.assertRaisesRegex(ValueError, 'new reference'):
                tracker.update(1000, 1.03)

    def test_missing_reference_and_ambiguous_sampling_are_rejected(self):
        with self.assertRaises(ValueError):
            LiftTracker(self.geometry(), reference_id='', zero_tick=1000, reference_time_s=1.)
        with self.assertRaises(ValueError):
            LiftGeometry(.1, 1, 5000., .5, -.4, .6).validate()
        with self.assertRaises(ValueError):
            LiftGeometry(.1, 0, 1000., .5, -.4, .6).validate()
        with self.assertRaises(ValueError):
            LiftGeometry(.1, 1, 1000., .5, .1, .6).validate()

    def test_homed_floor_tolerates_encoder_jitter_without_rebasing(self):
        geometry = LiftGeometry(.131, -1, 4000., .2, 0., .6)
        tracker = LiftTracker(geometry, reference_id='homed', zero_tick=2000,
                              reference_time_s=1.)
        for frame, tick in enumerate((2001, 2005, 2010, 2000, 1990), 1):
            height = tracker.update(tick, 1. + frame * .03)
            self.assertGreaterEqual(height, 0.)
            self.assertTrue(tracker.valid)
        self.assertAlmostEqual(height, 10 * .131 / 4096)
        self.assertEqual(tracker.extended_delta_ticks, -10)

    def test_boundary_tolerance_is_small_and_never_relaxes_target_limits(self):
        geometry = LiftGeometry(.131, 1, 4000., .2, 0., 100 * .131 / 4096)
        for tick, expected in ((1990, 0.), (2110, geometry.max_height_m)):
            tracker = LiftTracker(geometry, reference_id='homed', zero_tick=2000,
                                  reference_time_s=1.)
            self.assertEqual(tracker.update(tick, 1.1), expected)
            with self.assertRaisesRegex(ValueError, 'Target height'):
                tracker.height_to_delta_ticks(-.00001)
        for tick in (1989, 2111):
            tracker = LiftTracker(geometry, reference_id='homed', zero_tick=2000,
                                  reference_time_s=1.)
            with self.assertRaisesRegex(ValueError, 'Measured height'):
                tracker.update(tick, 1.1)
            self.assertFalse(tracker.valid)


if __name__ == '__main__':
    unittest.main()
