# SPDX-License-Identifier: Apache-2.0
"""Whole-robot command validation tests; no hardware access."""

from types import SimpleNamespace as N
import unittest

from xyc_hardware.command_contract import COMMAND_SCHEMA, CommandLimits, CommandValidator


def command():
    return N(
        header=N(stamp=N(sec=100, nanosec=0)), source_id='policy', sequence=1,
        schema_id=COMMAND_SCHEMA, arm_calibration_ids=['left-v1', 'right-v1'],
        lift_reference_id='lift-session-1', arm_position_normalized=[0.] * 14,
        lift_command=1., base_x_command=1., base_y_command=-1., base_yaw_command=0.,
        valid_for=N(sec=0, nanosec=200_000_000),
    )


class TestCommandContract(unittest.TestCase):
    def setUp(self):
        self.validator = CommandValidator(CommandLimits(.5, .5),
                                          ['left-v1', 'right-v1'], 'lift-session-1')
        self.now = 100_100_000_000

    def test_complete_valid_command_and_per_source_sequence(self):
        data = command()
        self.assertIsNone(self.validator.validate(data, now_ns=self.now))
        with self.assertRaisesRegex(ValueError, 'sequence'):
            self.validator.validate(data, now_ns=self.now)
        data.sequence += 1
        self.validator.validate(data, now_ns=self.now)
        data.source_id, data.sequence = 'replay', 1
        self.validator.validate(data, now_ns=self.now)

    def test_invalid_commands_do_not_consume_sequence(self):
        data = command()
        data.base_x_command = .5
        with self.assertRaisesRegex(ValueError, 'exactly'):
            self.validator.validate(data, now_ns=self.now)
        data.base_x_command = 1.
        self.validator.validate(data, now_ns=self.now)

    def test_calibration_schema_and_time_are_enforced(self):
        for mutation in ('calibration', 'lift_reference', 'schema', 'expired', 'future', 'lifetime'):
            data = command()
            if mutation == 'calibration': data.arm_calibration_ids.reverse()
            if mutation == 'lift_reference': data.lift_reference_id = 'old-session'
            if mutation == 'schema': data.schema_id = 'other'
            if mutation == 'expired': data.header.stamp.sec = 99
            if mutation == 'future': data.header.stamp.sec = 101
            if mutation == 'lifetime': data.valid_for.nanosec = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validator.validate(data, now_ns=self.now)

    def test_every_dimension_is_finite_and_inside_its_discrete_contract(self):
        for index, value in ((0, 101.), (6, -1.), (13, 101.), (8, float('nan'))):
            data = command()
            data.arm_position_normalized[index] = value
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.validator.validate(data, now_ns=self.now)
        for field, value in (('base_x_command', .2), ('base_y_command', float('inf')),
                             ('base_yaw_command', -2.), ('lift_command', .5)):
            data = command()
            setattr(data, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validator.validate(data, now_ns=self.now)

    def test_malformed_dimensions_and_limits_are_rejected(self):
        data = command()
        data.arm_position_normalized = [0.] * 13
        with self.assertRaises(ValueError):
            self.validator.validate(data, now_ns=self.now)
        with self.assertRaises(ValueError):
            CommandValidator(CommandLimits(0., .5), ['left-v1', 'right-v1'], 'lift-session-1')
        with self.assertRaises(ValueError):
            CommandValidator(CommandLimits(.5, .5), ['left-v1', ''], 'lift-session-1')

    def test_ros_wire_roundtrip_preserves_18_value_contract(self):
        from rclpy.serialization import deserialize_message, serialize_message
        from xyc_interfaces.msg import RobotCommand

        msg = RobotCommand()
        msg.header.stamp.sec = 100
        msg.source_id, msg.sequence, msg.schema_id = 'policy', 1, COMMAND_SCHEMA
        msg.arm_calibration_ids = ['left-v1', 'right-v1']
        msg.lift_reference_id = 'lift-session-1'
        msg.arm_position_normalized = [0.] * 14
        msg.lift_command, msg.base_x_command = 1., 1.
        msg.base_y_command, msg.base_yaw_command = -1., 0.
        msg.valid_for.nanosec = 200_000_000
        decoded = deserialize_message(serialize_message(msg), RobotCommand)
        self.validator.validate(decoded, now_ns=self.now)
        self.assertEqual(len(decoded.arm_position_normalized), 14)
        self.assertEqual((decoded.lift_command, decoded.base_x_command,
                          decoded.base_y_command, decoded.base_yaw_command), (1., 1., -1., 0.))


if __name__ == '__main__':
    unittest.main()
