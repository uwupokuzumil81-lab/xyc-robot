# SPDX-License-Identifier: Apache-2.0
"""Mathematical tests with hypothetical geometry; no real robot access."""
import math
import unittest

from xyc_hardware.base_kinematics import OmniGeometry


class TestOmniGeometry(unittest.TestCase):
    def setUp(self):
        self.geometry = OmniGeometry(.05, .2, tuple(map(math.radians, (150, -90, 30))), (1, 1, 1), 10.)

    def test_forward_inverse_preserves_body_velocity(self):
        for body in ((0., 0., 0.), (.1, 0., 0.), (0., -.1, 0.),
                     (0., 0., .2), (.1, -.07, .3)):
            with self.subTest(body=body):
                wheels = self.geometry.body_to_wheels(*body)
                actual = self.geometry.wheels_to_body(wheels)
                for a, b in zip(actual, body):
                    self.assertAlmostEqual(a, b)

    def test_pure_rotation_has_equal_wheel_velocity(self):
        wheels = self.geometry.body_to_wheels(0., 0., .5)
        for wheel in wheels:
            self.assertAlmostEqual(wheel, 2.)

    def test_motor_sign_is_explicit_and_invertible(self):
        signed = OmniGeometry(.05, .2, self.geometry.wheel_angles_rad, (-1, 1, 1), 10.)
        normal = self.geometry.body_to_wheels(.1, .05, .1)
        inverted = signed.body_to_wheels(.1, .05, .1)
        self.assertAlmostEqual(inverted[0], -normal[0])
        self.assertAlmostEqual(inverted[1], normal[1])
        for a, b in zip(signed.wheels_to_body(inverted), (.1, .05, .1)):
            self.assertAlmostEqual(a, b)

    def test_singular_geometry_and_bad_limits_are_rejected(self):
        for geometry in (OmniGeometry(0., .2, (0., 1., 2.), (1, 1, 1), 10.),
                         OmniGeometry(.05, .2, (0., 0., 0.), (1, 1, 1), 10.),
                         OmniGeometry(.05, .2, (0., 1., 2.), (1, 0, 1), 10.),
                         OmniGeometry(.05, .2, (0., 1., 2.), (1, 1, 1), float('nan'))):
            with self.assertRaises(ValueError):
                geometry.validate()

    def test_unachievable_targets_are_rejected_instead_of_silently_clipped(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            self.geometry.body_to_wheels(100., 0., 0.)
        with self.assertRaises(ValueError):
            self.geometry.body_to_wheels(float('nan'), 0., 0.)
        with self.assertRaises(ValueError):
            self.geometry.wheels_to_body((1., 2.))


if __name__ == '__main__':
    unittest.main()
