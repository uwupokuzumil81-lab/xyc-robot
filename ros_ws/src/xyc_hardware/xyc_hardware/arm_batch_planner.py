# SPDX-License-Identifier: Apache-2.0
"""All-or-nothing two-arm target planning; this module performs no I/O."""
from dataclasses import dataclass
from copy import deepcopy
import hashlib
import json
import math

from xyc_hardware.arm_calibration import (
    FEEDBACK_ENDPOINT_TOLERANCE_TICKS, arm_names, normalize_tick,
    position_target_tick, validate_document,
)
from xyc_hardware.command_contract import COMMAND_SCHEMA, _time_ns


def _digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(value)


@dataclass(frozen=True)
class JointSample:
    name: str
    bus: str
    motor_id: int
    model_number: int
    position_ticks: int
    stamp_ns: int
    valid: bool
    servo_error: int
    error: str


@dataclass(frozen=True)
class JointTarget:
    name: str
    bus: str
    motor_id: int
    initial_ticks: int
    target_ticks: int
    initial_normalized: float
    target_normalized: float


class ArmBatchPlanner:
    """Require validated calibration, complete fresh feedback, and explicit limits.

    Plans are advisory. A real executor must independently verify hardware,
    control ownership, command age, motion limits, stop behavior and bus failures.
    """

    def __init__(self, calibrations, *, max_age_s,
                 max_skew_s, future_tolerance_s):
        if set(calibrations) != {'left', 'right'}:
            raise ValueError('Two arm calibration documents are required')
        self.docs = {side: validate_document(deepcopy(calibrations[side])) for side in ('left', 'right')}
        if any(self.docs[side]['arm'] != side for side in ('left', 'right')):
            raise ValueError('Calibration arm identity mismatch')
        self.names = tuple(name for side in ('left', 'right') for name in arm_names(side))
        self.ids = tuple(_digest(self.docs[side]) for side in ('left', 'right'))
        self.max_age_s = _finite(max_age_s, 'max_age_s')
        self.max_skew_s = _finite(max_skew_s, 'max_skew_s')
        self.future_tolerance_s = _finite(future_tolerance_s, 'future_tolerance_s')
        if self.max_age_s <= 0 or self.max_skew_s <= 0 or self.future_tolerance_s < 0:
            raise ValueError('Invalid timing limits')

    def plan(self, *, targets, samples, calibration_ids, now_ns):
        if type(now_ns) is not int or now_ns <= 0:
            raise ValueError('now_ns must be positive')
        if tuple(calibration_ids) != self.ids:
            raise ValueError('Calibration identities differ')
        if len(targets) != 14 or len(samples) != 14:
            raise ValueError('Exactly 14 joint targets and samples are required')
        by_name = {sample.name: sample for sample in samples}
        if len(by_name) != 14 or set(by_name) != set(self.names):
            raise ValueError('Joint sample names are missing, duplicated or unexpected')
        output, stamps = [], []
        for index, name in enumerate(self.names):
            side = 'left' if index < 7 else 'right'
            entry = self.docs[side]['motors'][name]
            sample = by_name[name]
            if (sample.bus != side or sample.motor_id != entry['id']
                    or sample.model_number != entry['model_number']):
                raise ValueError(f'{name}: motor identity mismatch')
            if sample.valid is not True or sample.servo_error or sample.error:
                raise ValueError(f'{name}: feedback invalid')
            if type(sample.stamp_ns) is not int or sample.stamp_ns <= 0:
                raise ValueError(f'{name}: timestamp invalid')
            age = (now_ns - sample.stamp_ns) / 1e9
            if age > self.max_age_s or age < -self.future_tolerance_s:
                raise ValueError(f'{name}: feedback is stale or in the future')
            current = float(normalize_tick(
                entry, sample.position_ticks,
                feedback_tolerance_ticks=FEEDBACK_ENDPOINT_TOLERANCE_TICKS,
            ))
            target = _finite(targets[index], name)
            target_ticks = position_target_tick(entry, sample.position_ticks, target)
            output.append(JointTarget(name, side, entry['id'], sample.position_ticks,
                                      target_ticks, current, target))
            stamps.append(sample.stamp_ns)
        if (max(stamps) - min(stamps)) / 1e9 > self.max_skew_s:
            raise ValueError('Arm feedback time skew exceeds limit')
        return tuple(output)


class ArmCommandPreflight:
    """Read-only 14-axis check of RobotCommand; never executes the command."""

    def __init__(self, planner, *, max_lifetime_s, future_tolerance_s):
        if not isinstance(planner, ArmBatchPlanner):
            raise TypeError('A validated arm planner is required')
        self.planner = planner
        self.max_lifetime_s = _finite(max_lifetime_s, 'max_lifetime_s')
        self.future_tolerance_s = _finite(future_tolerance_s, 'future_tolerance_s')
        if self.max_lifetime_s <= 0 or self.future_tolerance_s < 0:
            raise ValueError('Invalid command timing limits')
        self.last_sequence = {}

    def check(self, command, feedback_by_bus, status_by_bus, *, now_ns):
        if type(now_ns) is not int or now_ns <= 0:
            raise ValueError('now_ns must be positive')
        if command.schema_id != COMMAND_SCHEMA:
            raise ValueError('Command schema does not match')
        if (not isinstance(command.source_id, str) or not command.source_id.strip()
                or len(command.source_id) > 64):
            raise ValueError('source_id must contain 1..64 nonblank characters')
        if type(command.sequence) is not int or command.sequence < 0:
            raise ValueError('sequence must be a nonnegative integer')
        if command.sequence <= self.last_sequence.get(command.source_id, -1):
            raise ValueError('Command sequence repeated or moved backwards')
        if command.source_id not in self.last_sequence and len(self.last_sequence) >= 32:
            raise ValueError('Too many command sources; restart the preflight node')
        stamp = _time_ns(command.header.stamp, 'header stamp')
        lifetime = _time_ns(command.valid_for, 'valid_for')
        if stamp <= 0 or not 0 < lifetime <= self.max_lifetime_s * 1e9:
            raise ValueError('Missing timestamp or invalid command lifetime')
        if now_ns - stamp >= lifetime:
            raise ValueError('Command has expired')
        if stamp - now_ns > self.future_tolerance_s * 1e9:
            raise ValueError('Command timestamp is in the future')
        if set(feedback_by_bus) != {'left', 'right'}:
            raise ValueError('Both arm feedback topics are required')
        if set(status_by_bus) != {'left', 'right'}:
            raise ValueError('Both controller status snapshots are required')
        for side in ('left', 'right'):
            status = status_by_bus[side]
            if not isinstance(status, dict):
                raise ValueError(f'{side} controller status is unavailable')
            expected = set(arm_names(side))
            if any(name not in expected for name in status):
                raise ValueError(f'{side} controller status contains unexpected joints')
            for name, detail in status.items():
                if (not isinstance(detail, dict) or type(detail.get('owned')) is not bool
                        or detail.get('phase') not in ('idle', 'moving', 'holding', 'fault')
                        or not isinstance(detail.get('fault'), str)):
                    raise ValueError(f'{name}: malformed controller status')
                if detail['fault'] or detail['phase'] == 'fault':
                    raise ValueError(f'{name}: controller fault: {detail["fault"] or "unspecified"}')
                if detail['phase'] in ('moving', 'holding') and not detail['owned']:
                    raise ValueError(f'{name}: inconsistent controller ownership')
        wanted = set(self.planner.names)
        samples = []
        for side in ('left', 'right'):
            batch = feedback_by_bus[side]
            if batch is None:
                raise ValueError(f'{side} feedback is missing')
            for item in batch.motors:
                if item.name not in wanted:
                    continue
                if item.bus != side:
                    raise ValueError('Feedback appeared on the wrong bus topic')
                samples.append(JointSample(
                    item.name, item.bus, item.motor_id, item.model_number,
                    item.position_ticks, _time_ns(item.header.stamp, 'sample stamp'),
                    item.valid, item.servo_error, item.error,
                ))
        plan = self.planner.plan(targets=command.arm_position_normalized, samples=samples,
                                 calibration_ids=command.arm_calibration_ids, now_ns=now_ns)
        self.last_sequence[command.source_id] = command.sequence
        return plan
