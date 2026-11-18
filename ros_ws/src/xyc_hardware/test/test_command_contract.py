# SPDX-License-Identifier: Apache-2.0
"""The running transport and learning stack use one canonical robot contract."""

import subprocess
import sys
import unittest

from xyc import contracts


class TestCommandContract(unittest.TestCase):
    def test_ternary_commands_map_to_the_existing_fixed_speeds(self):
        self.assertEqual(contracts.physical_mobile_action([1, -1, 0, 1]), [.015, -.15, 0., .60])
        self.assertEqual(contracts.physical_mobile_action([-1, 0, 1, -1]), [-.015, 0., .15, -.60])
        for values in ([], [1, 0, 0], [1, 0, 0, 1, 0], [.5, 0, 0, 0], [float('nan'), 0, 0, 0]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                contracts.physical_mobile_action(values)

    def test_strict_commands_keep_full_arm_bounds_and_reject_fractional_mobile_values(self):
        values = [0.] * 18
        values[0], values[2], values[6] = -100., 100., 100.
        self.assertEqual(contracts.checked_action(values), values)
        for index, value in ((0, -100.01), (2, 100.01), (6, -1.), (13, 101.),
                             (14, .5), (17, 2.), (1, float('nan')), (1, True)):
            bad = list(values)
            bad[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                contracts.checked_action(bad)
        for size in (0, 14, 17, 19):
            with self.assertRaises(ValueError):
                contracts.checked_action([0.] * size)

    def test_protocol_contract_import_needs_no_gpu_or_ros_runtime(self):
        code = ('import xyc.transport.motion, sys; '
                'assert "torch" not in sys.modules and "rclpy" not in sys.modules')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)



if __name__ == '__main__':
    unittest.main()
