# SPDX-License-Identifier: Apache-2.0
"""Calibrated states retain identities, quality flags and measured timestamps."""

from copy import deepcopy
import math
import unittest
from unittest.mock import patch

from xyc_interfaces.msg import MotorFeedback, MotorFeedbackArray
from xyc_hardware.arm_state import ArmStateConverter
from test_arm_calibration import fixture


def batch(spec):
    message = MotorFeedbackArray()
    message.header.stamp.sec = 100
    message.header.stamp.nanosec = 30_000_000
    for index, (name, (motor_id, model)) in enumerate(spec['motors'].items()):
        item = MotorFeedback()
        item.name, item.bus = name, spec['arm']
        item.motor_id, item.model_number = motor_id, model
        item.valid = True
        item.position_ticks = 2000
        item.header.stamp.sec = 100
        item.header.stamp.nanosec = index * 1_000_000
        message.motors.append(item)
    return message


class TestArmState(unittest.TestCase):
    def setup_converter(self, arm='left'):
        spec, _, document = fixture(arm)
        hardware = deepcopy(document['motors'])
        return spec, document, ArmStateConverter(document, hardware, spec)

    def test_order_and_normalized_values_ignore_nonarm_motors(self):
        for arm in ('left', 'right'):
            spec, _, converter = self.setup_converter(arm)
            message = batch(spec)
            wheel = MotorFeedback()
            wheel.name = 'base_left_wheel'
            wheel.motor_id = 8
            wheel.valid = False
            message.motors.insert(0, wheel)
            message.motors.reverse()
            with patch('xyc.robot.feetech.serial.Serial') as port:
                state = converter.convert(message)
            port.assert_not_called()
            self.assertEqual(state.names, list(spec['motors']))
            self.assertEqual(list(state.position_normalized), [0.] * 6 + [50.])
            self.assertTrue(state.complete)
            self.assertTrue(all(state.valid))
            self.assertEqual(state.arm, arm)

    def test_each_sample_timestamp_and_batch_time_are_preserved(self):
        spec, _, converter = self.setup_converter()
        message = batch(spec)
        state = converter.convert(message)
        self.assertEqual(state.header, message.header)
        self.assertEqual([stamp.nanosec for stamp in state.sample_stamps],
                         [i * 1_000_000 for i in range(7)])

    def test_missing_or_invalid_feedback_never_reuses_previous_value(self):
        spec, _, converter = self.setup_converter()
        message = batch(spec)
        self.assertTrue(converter.convert(message).complete)
        message.motors[0].valid = False
        message.motors[0].error = 'Read timeout'
        message.motors.pop()
        state = converter.convert(message)
        self.assertFalse(state.complete)
        for index in (0, 6):
            self.assertFalse(state.valid[index])
            self.assertTrue(math.isnan(state.position_normalized[index]))
            self.assertTrue(state.errors[index])
            self.assertEqual(state.sample_stamps[index].sec, 0)
        self.assertTrue(all(state.valid[1:6]))

    def test_feedback_tolerance_at_calibration_endpoints(self):
        spec, _, converter = self.setup_converter()
        message = batch(spec)
        for tick, expected in ((984, -100.), (3016, 100.)):
            message.motors[0].position_ticks = tick
            state = converter.convert(message)
            self.assertTrue(state.complete)
            self.assertEqual(state.position_normalized[0], expected)
            self.assertEqual(message.motors[0].position_ticks, tick)
        for tick in (983, 3017):
            message.motors[0].position_ticks = tick
            state = converter.convert(message)
            self.assertFalse(state.valid[0])
            self.assertTrue(math.isnan(state.position_normalized[0]))
            self.assertIn('outside', state.errors[0])

    def test_wrong_id_model_bus_or_duplicate_motor_is_rejected(self):
        spec, _, converter = self.setup_converter()
        for field, value in (('motor_id', 7), ('model_number', 777), ('bus', 'right')):
            message = batch(spec)
            setattr(message.motors[0], field, value)
            state = converter.convert(message)
            self.assertFalse(state.valid[0])
        message = batch(spec)
        message.motors.append(deepcopy(message.motors[0]))
        self.assertFalse(converter.convert(message).valid[0])

    def test_mismatched_hardware_cannot_construct_a_converter(self):
        spec, document, _ = self.setup_converter()
        for key, value in (('hardware_homing_offset', 0), ('operating_mode', 1)):
            hardware = deepcopy(document['motors'])
            hardware[next(iter(hardware))][key] = value
            with self.assertRaises(ValueError):
                ArmStateConverter(document, hardware, spec)

    def test_calibration_identity_tracks_asset_content_not_json_key_order(self):
        spec, document, converter = self.setup_converter()
        reordered = dict(reversed(list(document.items())))
        same = ArmStateConverter(reordered, document['motors'], spec)
        self.assertEqual(same.calibration_id, converter.calibration_id)
        changed = deepcopy(document)
        changed['source'] = 'different_asset_version'
        other = ArmStateConverter(changed, changed['motors'], spec)
        self.assertNotEqual(other.calibration_id, converter.calibration_id)
        self.assertEqual(len(other.calibration_id), 64)


if __name__ == '__main__':
    unittest.main()
