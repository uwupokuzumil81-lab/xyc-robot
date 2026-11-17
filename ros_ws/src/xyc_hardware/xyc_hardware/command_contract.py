# SPDX-License-Identifier: Apache-2.0
"""Validate the v3 arm-position plus ternary-motion command contract."""

from dataclasses import dataclass
import math


COMMAND_SCHEMA = 'xyc.robot_command.v3'


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _time_ns(value, name):
    if type(value.sec) is not int or type(value.nanosec) is not int:
        raise ValueError(f'{name} must have integer sec and nanosec')
    if value.sec < 0 or not 0 <= value.nanosec < 1_000_000_000:
        raise ValueError(f'{name} is malformed')
    return value.sec * 1_000_000_000 + value.nanosec


@dataclass(frozen=True)
class CommandLimits:
    """No inferred hardware limits: callers must supply measured, verified limits."""

    max_valid_for_s: float
    future_tolerance_s: float

    def validate(self):
        if _finite(self.max_valid_for_s, 'max_valid_for_s') <= 0:
            raise ValueError('max_valid_for_s must be positive')
        if _finite(self.future_tolerance_s, 'future_tolerance_s') < 0:
            raise ValueError('future_tolerance_s must not be negative')
        return self


class CommandValidator:
    """Check individual messages and per-source replay; not an actuator watchdog."""

    def __init__(self, limits, arm_calibration_ids, lift_reference_id):
        self.limits = limits.validate()
        self.calibration_ids = tuple(arm_calibration_ids)
        if len(self.calibration_ids) != 2 or any(not isinstance(v, str) or not v for v in self.calibration_ids):
            raise ValueError('Provide verified left and right calibration IDs')
        if not isinstance(lift_reference_id, str) or not lift_reference_id:
            raise ValueError('Provide a verified lift reference ID')
        self.lift_reference_id = lift_reference_id
        self.last_sequence = {}

    def validate(self, command, *, now_ns):
        if type(now_ns) is not int or now_ns <= 0:
            raise ValueError('now_ns must be a positive ROS clock time')
        if command.schema_id != COMMAND_SCHEMA:
            raise ValueError('Command schema does not match')
        if (not isinstance(command.source_id, str) or not command.source_id.strip()
                or len(command.source_id) > 64):
            raise ValueError('source_id must contain 1..64 nonblank characters')
        if type(command.sequence) is not int or command.sequence < 0:
            raise ValueError('sequence must be a nonnegative integer')
        if tuple(command.arm_calibration_ids) != self.calibration_ids:
            raise ValueError('Arm calibration versions do not match')
        if command.lift_reference_id != self.lift_reference_id:
            raise ValueError('Lift reference does not match')

        stamp = _time_ns(command.header.stamp, 'header stamp')
        lifetime = _time_ns(command.valid_for, 'valid_for')
        if stamp <= 0 or not 0 < lifetime <= self.limits.max_valid_for_s * 1e9:
            raise ValueError('Missing timestamp or invalid command lifetime')
        if now_ns - stamp > lifetime:
            raise ValueError('Command has expired')
        if stamp - now_ns > self.limits.future_tolerance_s * 1e9:
            raise ValueError('Command timestamp is in the future')

        positions = command.arm_position_normalized
        if len(positions) != 14:
            raise ValueError('Exactly 14 arm/gripper positions are required')
        for index, value in enumerate(positions):
            lower = 0.0 if index in (6, 13) else -100.0
            if not lower <= _finite(value, f'arm position {index}') <= 100.0:
                raise ValueError(f'arm position {index} is outside calibration units')
        for field in ('lift_command', 'base_x_command', 'base_y_command', 'base_yaw_command'):
            value = _finite(getattr(command, field), field)
            if value not in (-1., 0., 1.):
                raise ValueError(f'{field} must be exactly -1, 0 or +1')
        if command.sequence <= self.last_sequence.get(command.source_id, -1):
            raise ValueError('Command sequence repeated or moved backwards')
        self.last_sequence[command.source_id] = command.sequence
