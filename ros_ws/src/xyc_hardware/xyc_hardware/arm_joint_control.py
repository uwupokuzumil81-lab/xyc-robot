# SPDX-License-Identifier: Apache-2.0
"""Shared calibrated arm control, persistent goals and gripper contact hold.

Continuous targets use one grouped write per serial bus. Startup torque and
profile setup, contact preload and torque release use verified per-motor writes.
Profile layout follows Feetech's public sms_sts.WritePosEx implementation.
"""

from dataclasses import dataclass
from collections import deque
import json
import math
import time

import serial

from xyc.robot.calibration import (
    FEEDBACK_ENDPOINT_TOLERANCE_TICKS, arm_names, denormalize_tick, integer, normalize_tick,
    position_target_tick,
)
from xyc.robot.feetech import FeetechReader, decode_feedback, signed_magnitude
from xyc.transport.teleop import MotionSessionEnded, TARGET_COMMAND_EXPIRED
from xyc.contracts import CommandExpired
from xyc_hardware.motor_safety import (
    CurrentProtection, FEEDBACK_LOSS_GRACE_S, MOTOR_TEMPERATURE_LIMIT_C, confirm_temperature,
)


ARM_ACCEL_RAW = 120
ARM_SPEED_RAW = 2400
GRIPPER_ACCEL_RAW = 100
ARRIVAL_WARNING_S = 3.0
HOLD_WARNING_TICKS = 30
HOLD_RECOVERY_TICKS = 10
HOLD_WARNING_SAMPLES = 3
MOTOR_LOAD_LIMIT_RAW = 300
# Continuous gripper contact is handled separately from arm-joint faults.
# Register 48 is the servo's volatile maximum-output-torque limit.
GRIPPER_TORQUE_LIMIT_RAW_DEFAULT = 100  # 10% of maximum; tune for the actual fingers/load.
GRIPPER_CONTACT_CURRENT_A = 0.30
GRIPPER_CONTACT_LOAD_RAW = MOTOR_LOAD_LIMIT_RAW
GRIPPER_CONTACT_ERROR_TICKS = 23
GRIPPER_CONTACT_PROGRESS_TICKS = 3
GRIPPER_CONTACT_CONFIRM_S = 0.12
GRIPPER_CONTACT_MAX_SAMPLE_GAP_S = 0.06
GRIPPER_CONTACT_PRELOAD_TICKS = 8
GRIPPER_CONTACT_RELEASE_HYSTERESIS_TICKS = 10
ARM_LOAD_STALL_WARNING_S = 0.150
ARM_LOAD_STALL_TRIP_S = 0.650
ARM_LOAD_MAX_SAMPLE_GAP_S = 0.060
# Round up the reference 2° error and 0.2° progress on a 4096-tick encoder.
ARM_LOAD_STALL_ERROR_TICKS = 23
ARM_LOAD_PROGRESS_TICKS = 3


class JointDriver(FeetechReader):
    """Extra capability with a strict ID/register/profile whitelist."""

    motion_fault = ''
    motion_gate = None

    def __init__(self, device, motor_ids, *, bus, joint_id, **kwargs):
        expected = list(range(1, 12 if bus == 'left' else 8))
        if bus not in ('left', 'right') or list(motor_ids) != expected:
            raise ValueError('Unexpected arm bus profile')
        if type(joint_id) is not int or not 1 <= joint_id <= 7:
            raise ValueError('Only a single configured arm motor ID 1..7 is allowed')
        self.joint_id = joint_id
        self.bus = bus
        super().__init__(device, motor_ids, **kwargs)

    def motion_profile(self, motor_id):
        """Return the single continuous-control speed profile for an arm motor."""
        if motor_id == 7:  # Gripper on either arm bus.
            return ARM_SPEED_RAW, GRIPPER_ACCEL_RAW
        return ARM_SPEED_RAW, ARM_ACCEL_RAW

    def _request(self, motor_id, instruction, parameters):
        allowed = False
        if motor_id == self.joint_id:
            if instruction == 2:
                allowed = parameters in (b'\x12\x01', b'\x28\x01', b'\x28\x1f', b'\x29\x07',
                                         b'\x3f\x01') or (
                    motor_id == 7 and parameters == b'\x30\x02'
                )
            elif instruction == 3:
                allowed = parameters in (b'\x28\x00', b'\x28\x01')
                if len(parameters) == 8 and parameters[0] == 41:
                    speed, acceleration = self.motion_profile(motor_id)
                    allowed = (
                        parameters[1] == acceleration
                        and int.from_bytes(parameters[2:4], 'little') <= 4095
                        and parameters[4:6] == b'\x00\x00'
                        and int.from_bytes(parameters[6:8], 'little') == speed
                    )
                if motor_id == 7 and len(parameters) == 3 and parameters[0] == 48:
                    allowed = 1 <= int.from_bytes(parameters[1:3], 'little') <= 1000
        if not allowed:
            return super()._request(motor_id, instruction, parameters)
        moves = instruction == 3 and (parameters[0] == 41 or parameters == b'\x28\x01')
        self._send_joint_packet(motor_id, instruction, parameters, moves=moves)

    def _send_joint_packet(self, motor_id, instruction, parameters, *, deadline=None, moves=False):
        body = bytes([motor_id, len(parameters) + 2, instruction]) + parameters
        packet = b'\xff\xff' + body + bytes([(~sum(body)) & 255])
        attempts = 2 if instruction in (2, 0x82) else 1
        def write():
            if moves and self.motion_fault:
                raise CommandExpired(self.motion_fault)
            if deadline is not None and time.monotonic() >= deadline:
                raise CommandExpired('Command expired before serial port write')
            return self.port.write(packet)

        for attempt in range(attempts):
            self.port.reset_input_buffer()
            try:
                written = self.motion_gate(write) if moves and self.motion_gate is not None else write()
                if written != len(packet):
                    raise OSError(f'{self.bus} ID {motor_id}: incomplete instruction 0x{instruction:02x} transmission')
                return
            except serial.SerialTimeoutException as exc:
                if attempt + 1 == attempts:
                    raise serial.SerialTimeoutException(
                        f'{self.bus} ID {motor_id}: instruction 0x{instruction:02x} '
                        f'transmission timed out after {attempts} attempt(s)'
                    ) from exc
                # Only reads may be repeated. A timed-out motion/torque write
                # can already have reached the servo and must never be replayed.

    def _read_joint_register(self, address, size):
        # Reissue reads only. A missing reply does not justify repeating a
        # torque/goal write whose outcome may already be successful.
        for _ in range(3):
            self._request(self.joint_id, 2, bytes([address, size]))
            reply = self._receive((self.joint_id,), size, .025).get(self.joint_id)
            if reply is None:
                continue
            if reply.servo_error:
                raise ConnectionError(
                    f'{self.bus} ID {self.joint_id}: register {address} servo error 0x{reply.servo_error:02x}'
                )
            return reply.data
        raise ConnectionError(f'{self.bus} ID {self.joint_id}: no register {address} reply after 3 reads')

    def _write_verified(self, address, data):
        self._request(self.joint_id, 3, bytes([address]) + data)
        reply = self._receive((self.joint_id,), 0, 0.02).get(self.joint_id)
        if reply is not None and reply.servo_error:
            raise ConnectionError(f'Joint write error: {reply.servo_error}')
        if self._read_joint_register(address, len(data)) != data:
            raise ConnectionError(f'Joint write readback failed: register {address}')

    def read_joint(self):
        return decode_feedback(self._read_joint_register(56, 15))

    def read_torque(self):
        return integer(self._read_joint_register(40, 1)[0], 0, 1)

    def read_temperature(self):
        return self._read_joint_register(63, 1)[0]

    def read_torque_feedback(self):
        # STS RAM register 40 holds torque; registers 56..70 hold feedback.
        registers = self._read_joint_register(40, 31)
        return integer(registers[0], 0, 1), decode_feedback(registers[16:31])

    def verify_speed_units(self):
        """Require the STS fine speed unit used by the continuous-control profile."""
        previous = self.joint_id
        try:
            for motor_id in range(1, 8):
                self.joint_id = motor_id
                if not self._read_joint_register(18, 1)[0] & 0x04:
                    raise ValueError(f'{self.bus} ID {motor_id}: continuous control requires fine speed units')
        finally:
            self.joint_id = previous

    def set_torque(self, enabled):
        if type(enabled) is not bool:
            raise ValueError('Torque argument must be a boolean')
        if enabled:
            self._write_verified(40, b'\x01')
            return  # Never repeat an enable whose outcome may be unknown.
        for attempt in range(3):
            try:
                self._write_verified(40, b'\x00')
                return
            except (OSError, ValueError, RuntimeError) as exc:
                # An explicit torque-OFF request is idempotent. Re-read after
                # a failed verification: accept measured OFF, or repeat OFF
                # only when the hardware positively confirms it is still ON.
                try:
                    torque = self.read_torque()
                except (OSError, ValueError, RuntimeError) as read_error:
                    raise ConnectionError(
                        f'Torque-off unconfirmed; hardware state unavailable: {read_error}'
                    ) from exc
                if torque == 0:
                    return
                if attempt == 2:
                    raise ConnectionError(
                        'Torque-off failed: hardware remains enabled after 3 verified attempts'
                    ) from exc

    def write_goal(self, ticks):
        integer(ticks, 0, 4095)
        speed, acceleration = self.motion_profile(self.joint_id)
        profile = bytes([acceleration]) + ticks.to_bytes(2, 'little') + b'\x00\x00'
        profile += speed.to_bytes(2, 'little')
        self._write_verified(41, profile)

    def write_torque_limit(self, limit_raw):
        integer(limit_raw, 1, 1000)
        self._write_verified(48, limit_raw.to_bytes(2, 'little'))

    def write_arm_goals(self, goals, *, deadline=None):
        """Send changed positions on this bus and verify the target registers."""
        if not goals or any(type(i) is not int or not 1 <= i <= 7 for i in goals):
            raise ValueError('Grouped goal writes require arm motor IDs 1..7')
        ids = tuple(sorted(goals))
        encoded = {i: integer(goals[i], 0, 4095).to_bytes(2, 'little') for i in ids}
        # Startup has already verified each joint's speed and acceleration.
        payload = b''.join(bytes([i]) + encoded[i] for i in ids)
        self._send_joint_packet(254, 0x83, b'\x2a\x02' + payload, deadline=deadline, moves=True)
        self._send_joint_packet(254, 0x82, b'\x2a\x02' + bytes(ids))
        replies = self._receive(ids, 2, .025)
        failed = [i for i in ids if i not in replies or replies[i].servo_error
                  or replies[i].data != encoded[i]]
        if failed:
            raise ConnectionError(f'Grouped goal readback failed on {self.bus} IDs {failed}')

    def read_arm_control_snapshot(self):
        """Read the same preflight registers for all seven joints in one request."""
        ids = tuple(range(1, 8))
        self._send_joint_packet(254, 0x82, bytes([31, 40, *ids]))
        replies = self._receive(ids, 40, .025)
        failed = {i: (f'0x{replies[i].servo_error:02x}' if i in replies else 'no reply')
                  for i in ids if i not in replies or replies[i].servo_error}
        if failed:
            raise ConnectionError(f'{self.bus}: grouped preflight feedback failed: {failed}')
        snapshots = {}
        for i in ids:
            data = replies[i].data
            settings = {
                'hardware_homing_offset': signed_magnitude(int.from_bytes(data[:2], 'little'), 11),
                'operating_mode': data[2],
            }
            # Block starts at 31. Torque is register 40, feedback is 56..70.
            snapshots[i] = (settings, integer(data[9], 0, 1), decode_feedback(data[25:40]))
        return snapshots


class BoundJointDriver:
    """Address one joint on a serial bus owned by a single-threaded executor."""

    def __init__(self, driver, joint_id):
        if type(joint_id) is not int or not 1 <= joint_id <= 7:
            raise ValueError('Joint ID must be between 1 and 7')
        self._driver = driver
        self.bus = driver.bus
        self.joint_id = joint_id

    def _call(self, method, *args):
        previous = self._driver.joint_id
        self._driver.joint_id = self.joint_id
        try:
            return getattr(self._driver, method)(*args)
        finally:
            self._driver.joint_id = previous

    def read_coordinate_settings(self, ids):
        if ids != [self.joint_id]:
            raise ValueError('Cannot access another joint calibration')
        return self._call('read_coordinate_settings', ids)

    def read_joint(self):
        return self._call('read_joint')

    def read_torque(self):
        return self._call('read_torque')

    def read_temperature(self):
        return self._call('read_temperature')

    def read_torque_feedback(self):
        return self._call('read_torque_feedback')

    def write_torque_limit(self, limit_raw):
        return self._call('write_torque_limit', limit_raw)

    def set_torque(self, enabled):
        return self._call('set_torque', enabled)

    def write_goal(self, ticks):
        return self._call('write_goal', ticks)


@dataclass(frozen=True)
class JointTarget:
    current_normalized: float
    target_normalized: float
    target_ticks: int


class JointControl:
    """Persistent absolute position control, monitored by the feedback timer."""

    def __init__(self, driver, entry, *, clock=time.monotonic, report_temperature_recheck=None,
                 gripper_torque_limit_raw=GRIPPER_TORQUE_LIMIT_RAW_DEFAULT):
        if (entry['id'] != driver.joint_id or entry['model_number'] not in (2569, 2825)
                or entry['operating_mode'] != 0
                or (entry['normalized_min'], entry['normalized_max']) not in ((-100, 100), (0, 100))):
            raise ValueError('Unexpected joint calibration or motor identity')
        self.driver, self.entry, self.clock = driver, entry, clock
        self.is_gripper = (entry['normalized_min'], entry['normalized_max']) == (0, 100)
        if (type(gripper_torque_limit_raw) is not int
                or not 1 <= gripper_torque_limit_raw <= 1000):
            raise ValueError('gripper_torque_limit_raw must be an integer in [1, 1000]')
        self.gripper_torque_limit_raw = gripper_torque_limit_raw if self.is_gripper else None
        self.gripper_contact_latched = False
        self.gripper_contact_direction = 0
        self.gripper_contact_goal_ticks = None
        self.gripper_contact_candidate = None
        self.is_arm = not self.is_gripper
        self.current_protection = CurrentProtection(entry['model_number'], gripper=self.is_gripper)
        self._reset_load_watch()
        self.report_temperature_recheck = report_temperature_recheck
        self.last_progress = None
        self.busy = False
        self.holding = False
        self.owned = False
        self.fault = ''
        self.arrival_warning_at = 0.0
        self.target_ticks = 0
        self.start_ticks = 0
        self.settled = 0
        self.next_torque_check = 0.0
        self.last_outcome = 'idle'
        self.last_message = ''
        self.progress_tick = None
        self.hold_deviation_warned = False
        self.hold_deviation_count = 0
        self.motion_warning = False
        self.feedback_loss_started = None
        self.feedback_loss_warned = False
        # In-memory context only; written to the normal ROS log on a fault.
        self.recent_feedback = deque(maxlen=30)

    @property
    def phase(self):
        if self.fault:
            return 'fault'
        if self.holding:
            return 'holding'
        if self.busy:
            return 'moving'
        return 'idle'

    @staticmethod
    def validate_sample(sample):
        integer(sample['position_ticks'], 0, 4095)
        for key in ('temperature_c', 'current_a'):
            if not math.isfinite(sample[key]):
                raise ValueError(f'Invalid {key}')
        if sample['temperature_c'] >= MOTOR_TEMPERATURE_LIMIT_C:
            raise ValueError(
                f"Joint temperature reached the control limit ({MOTOR_TEMPERATURE_LIMIT_C} C): "
                f"temperature={sample['temperature_c']:.0f} C, "
                f"position={sample['position_ticks']} ticks, "
                f"current={sample['current_a']:.4f} A, load_raw={sample['load_raw']}"
            )

    def _validate_motion_sample(self, sample, *, now=None):
        now = self.clock() if now is None else now
        if (self.report_temperature_recheck is not None
                and sample['temperature_c'] >= MOTOR_TEMPERATURE_LIMIT_C):
            sample = self._confirm_motion_temperature(sample)
        self.validate_sample(sample)
        reason = self.current_protection.observe(sample['current_a'], now)
        if reason:
            raise ValueError('Joint ' + reason)

    def _confirm_motion_temperature(self, sample):
        return confirm_temperature(
            sample, read_temperature=self.driver.read_temperature,
            read_feedback=self.driver.read_joint, report=self.report_temperature_recheck,
        )

    def _reset_load_watch(self):
        self.load_stall_candidate = None
        self.load_stall_warned = False

    def _observe_gripper_contact(self, sample, now):
        """Treat sustained loaded no-progress during streaming as normal contact."""
        if not self.is_gripper or self.gripper_contact_latched:
            self.gripper_contact_candidate = None
            return ''
        position = sample['position_ticks']
        error = self.target_ticks - position
        load = sample['load_raw']
        current = abs(sample['current_a'])
        loaded = (current >= GRIPPER_CONTACT_CURRENT_A
                  or abs(load) >= GRIPPER_CONTACT_LOAD_RAW)
        if abs(error) < GRIPPER_CONTACT_ERROR_TICKS or not loaded:
            self.gripper_contact_candidate = None
            return ''
        direction = 1 if error > 0 else -1
        candidate = self.gripper_contact_candidate
        if (candidate is None or direction != candidate[2]
                or not 0 <= now - candidate[3] <= GRIPPER_CONTACT_MAX_SAMPLE_GAP_S
                or (position - candidate[1]) * direction >= GRIPPER_CONTACT_PROGRESS_TICKS):
            self.gripper_contact_candidate = (now, position, direction, now)
            return ''
        self.gripper_contact_candidate = (candidate[0], candidate[1], candidate[2], now)
        if now - candidate[0] < GRIPPER_CONTACT_CONFIRM_S:
            return ''

        preload = min(GRIPPER_CONTACT_PRELOAD_TICKS, abs(error))
        contact_goal = position + direction * preload
        try:
            normalize_tick(self.entry, contact_goal)
        except ValueError:
            contact_goal = position
        # Replace a far-closing target with a small calibrated preload. The
        # configured torque cap preserves a bounded, nonzero grasp effort.
        self.driver.write_goal(contact_goal)
        self.gripper_contact_latched = True
        self.gripper_contact_direction = direction
        self.gripper_contact_goal_ticks = contact_goal
        self.gripper_contact_candidate = None
        self._record_goal(contact_goal, position)
        self.last_message = '夹爪接触已确认，进入限扭矩夹持；反向张开指令可解除接触锁存'
        return self.last_message

    def _clear_gripper_contact(self):
        self.gripper_contact_latched = False
        self.gripper_contact_direction = 0
        self.gripper_contact_goal_ticks = None
        self.gripper_contact_candidate = None

    def _observe_motion_load(self, sample, now):
        # Only the periodic feedback loop advances these timers. Target inspection
        # checks current and temperature but cannot count as another feedback frame.
        load = sample['load_raw']
        if abs(load) <= MOTOR_LOAD_LIMIT_RAW:
            self._reset_load_watch()
            return ''
        position = sample['position_ticks']
        error = self.target_ticks - position
        if abs(error) < ARM_LOAD_STALL_ERROR_TICKS:
            self._reset_load_watch()
            return ''
        direction = 1 if error > 0 else -1
        candidate = self.load_stall_candidate
        if (candidate is None or direction != candidate[2]
                or not 0 <= now - candidate[3] <= ARM_LOAD_MAX_SAMPLE_GAP_S
                or (position - candidate[1]) * direction >= ARM_LOAD_PROGRESS_TICKS):
            self.load_stall_candidate = (now, position, direction, now)
            return ''
        self.load_stall_candidate = (candidate[0], candidate[1], candidate[2], now)
        # The STS load field is drive duty cycle. An arm can use substantial
        # duty while moving normally, so only treat it as a stall with no progress.
        duration = now - candidate[0]
        if duration >= ARM_LOAD_STALL_TRIP_S:
            raise ValueError(
                f'Joint sustained high load without progress for {duration:.3f} s; '
                f'load_raw={load}, target error={error} ticks'
            )
        if not self.load_stall_warned and duration >= ARM_LOAD_STALL_WARNING_S:
            self.load_stall_warned = True
            return (f'Joint stall suspected; load_raw={load}, target error={error} ticks; '
                    'keeping the goal and torque while monitoring load')
        return ''

    def inspect_hold_current(self, *, control_snapshot=None):
        """Read-only inspection for taking ownership of a stationary joint."""
        if self.owned or self.busy or self.holding or self.fault:
            raise ValueError('Joint is already owned, active or faulted')
        if control_snapshot is None:
            snapshot = self.driver.read_coordinate_settings([self.entry['id']])[self.entry['id']]
            read_snapshot = getattr(self.driver, 'read_torque_feedback', None)
            if read_snapshot is None:
                torque, sample = self.driver.read_torque(), self.driver.read_joint()
            else:
                torque, sample = read_snapshot()
        else:
            snapshot, torque, sample = control_snapshot
        if any(snapshot[key] != self.entry[key] for key in ('hardware_homing_offset', 'operating_mode')):
            raise ValueError('Joint hardware coordinates changed')
        if torque != 0:
            raise ValueError('Joint is already energized outside this controller')
        if self.report_temperature_recheck is not None and sample['temperature_c'] >= MOTOR_TEMPERATURE_LIMIT_C:
            sample = self._confirm_motion_temperature(sample)
        self.validate_sample(sample)
        if sample['moving'] or abs(sample['velocity_raw']) > 10:
            raise ValueError('Joint must be stationary before holding')
        current = float(normalize_tick(
            self.entry, sample['position_ticks'],
            feedback_tolerance_ticks=FEEDBACK_ENDPOINT_TOLERANCE_TICKS,
        ))
        return JointTarget(current, current, sample['position_ticks'])

    def hold_current_position(self, *, expected_tick=None, tolerance_ticks=2):
        """Seed the fresh measured position, then hold it; called explicitly by an owner."""
        if type(tolerance_ticks) is not int or tolerance_ticks < 0:
            raise ValueError('Invalid startup position tolerance')
        if expected_tick is not None:
            integer(expected_tick, 0, 4095)
        plan = self.inspect_hold_current()
        tick = plan.target_ticks
        if expected_tick is not None and abs(tick - expected_tick) > tolerance_ticks:
            raise ValueError('Joint position changed after startup inspection')
        if self.gripper_torque_limit_raw is not None:
            # Apply and verify the volatile output-torque cap before energizing.
            self.driver.write_torque_limit(self.gripper_torque_limit_raw)
        # Goal is primed and verified with torque off; never reuse a stale goal.
        self.driver.write_goal(tick)
        self.owned = True  # The write may succeed even when its acknowledgement is lost.
        try:
            self.driver.set_torque(True)
        except (OSError, ValueError, RuntimeError) as exc:
            self._protective_release(str(exc))
            raise
        now = self.clock()
        self.start_ticks = tick
        self.target_ticks = tick
        self.progress_tick = tick
        self.last_progress = now
        self.next_torque_check = now + .5
        self.busy = False
        self.holding = True
        self.last_outcome = 'holding'
        self.last_message = 'Holding the measured startup position'
        self.recent_feedback.clear()
        self.hold_deviation_warned = False
        self.hold_deviation_count = 0
        self.motion_warning = False
        self.feedback_loss_started = None
        self.feedback_loss_warned = False
        return plan

    def inspect_absolute_target(self, position, *, control_snapshot):
        """Validate one held-joint target against fresh feedback without writing."""
        if (isinstance(position, bool) or not isinstance(position, (int, float))
                or not math.isfinite(position)):
            raise ValueError('Absolute target must be a finite number')
        if self.fault:
            raise ValueError(f'Joint fault: {self.fault}')
        if not self.owned or not (self.busy or self.holding):
            raise ValueError('Joint is not owned or no longer moving/holding')
        settings, torque, sample = control_snapshot
        if any(settings[key] != self.entry[key] for key in ('hardware_homing_offset', 'operating_mode')):
            raise ValueError('Joint hardware coordinates changed')
        if torque != 1:
            raise ValueError('Joint torque unexpectedly disabled')
        self._validate_motion_sample(sample)
        current = float(normalize_tick(
            self.entry, sample['position_ticks'],
            feedback_tolerance_ticks=FEEDBACK_ENDPOINT_TOLERANCE_TICKS,
        ))
        # Absolute control preserves the requested goal. Servo speed/acceleration
        # settings govern motion; this layer does not replace a far goal by a
        # shorter one based on the latest feedback or on a per-update step cap.
        target = position_target_tick(self.entry, sample['position_ticks'], float(position))
        if self.gripper_contact_latched:
            # Ignore commands that continue pressing in the contact direction.
            # A deliberate reverse/open command clears the latch only after its
            # write succeeds in _record_goal().
            reverse_ticks = self.gripper_contact_direction * (
                target - self.gripper_contact_goal_ticks
            )
            if reverse_ticks > -GRIPPER_CONTACT_RELEASE_HYSTERESIS_TICKS:
                target = self.gripper_contact_goal_ticks
        return JointTarget(current, float(normalize_tick(self.entry, target)), target)

    def _record_goal(self, target, measured_tick):
        """Update the feedback monitor only after the new goal was verified."""
        if self.gripper_contact_latched and (
                self.gripper_contact_direction * (target - self.gripper_contact_goal_ticks)
                < -GRIPPER_CONTACT_RELEASE_HYSTERESIS_TICKS):
            self._clear_gripper_contact()
        now = self.clock()
        if self.holding:
            self.progress_tick = measured_tick
            self.last_progress = now
        self.start_ticks = measured_tick
        self.target_ticks = target
        self.arrival_warning_at = now + ARRIVAL_WARNING_S
        self.holding = False
        self.busy = True
        self.settled = 0
        self.hold_deviation_warned = False
        self.hold_deviation_count = 0
        self.motion_warning = False

    def _protective_release(self, reason):
        try:
            self.release()
        except (OSError, ValueError, RuntimeError) as exc:
            self.fault = f'{reason}; torque-off unconfirmed: {exc}'
            raise RuntimeError(self.fault) from exc
        self.fault = reason
        self.last_outcome = 'failed'
        self.last_message = f'{reason}; torque-off confirmed'

    def release(self):
        was_active = self.owned or self.busy or self.holding
        try:
            self.driver.set_torque(False)
        except (OSError, ValueError, RuntimeError) as exc:
            self.fault = f'Torque-off unconfirmed: {exc}'
            self.busy = False
            self.holding = False
            self.last_outcome = 'failed'
            self.last_message = self.fault
            raise
        self.busy = False
        self.holding = False
        self.owned = False
        self.fault = ''
        self.settled = 0
        self.hold_deviation_warned = False
        self.hold_deviation_count = 0
        self.motion_warning = False
        self.feedback_loss_started = None
        self.feedback_loss_warned = False
        self._reset_load_watch()
        self.current_protection.reset()
        self._clear_gripper_contact()
        if was_active:
            self.last_outcome = 'released'
            self.last_message = 'Motion/holding released before normal completion'

    def observe(self, sample):
        if not self.busy and not self.holding:
            return None
        now = self.clock()
        keys = ('position_ticks', 'velocity_raw', 'load_raw', 'current_a',
                'voltage_v', 'temperature_c', 'moving')
        self.recent_feedback.append({
            'monotonic_s': round(now, 6),
            'sample': {key: sample.get(key) for key in keys} if sample is not None else None,
        })
        reason = ''
        warning = ''
        load_warning = ''
        reached = False
        if sample is None:
            if self.feedback_loss_started is None:
                self.feedback_loss_started = now
            duration = now - self.feedback_loss_started
            if duration >= FEEDBACK_LOSS_GRACE_S:
                reason = f'Joint lost valid feedback for {duration:.3f} s'
            elif not self.feedback_loss_warned:
                self.feedback_loss_warned = True
                self.last_message = (
                    'Joint feedback temporarily unavailable; keeping the existing goal and torque '
                    f'for at most {FEEDBACK_LOSS_GRACE_S:.2f} s'
                )
                return ('warning', self.last_message)
            else:
                return None
        else:
            self.feedback_loss_started = None
            self.feedback_loss_warned = False
            try:
                self._validate_motion_sample(sample, now=now)
                if self.is_arm:
                    load_warning = self._observe_motion_load(sample, now)
                if self.is_gripper:
                    contact_warning = self._observe_gripper_contact(sample, now)
                    if contact_warning:
                        load_warning = contact_warning
                if self.busy:
                    normalize_tick(
                        self.entry, sample['position_ticks'],
                        feedback_tolerance_ticks=FEEDBACK_ENDPOINT_TOLERANCE_TICKS,
                    )
                    if abs(sample['position_ticks'] - self.progress_tick) >= 2:
                        self.progress_tick = sample['position_ticks']
                        self.last_progress = now
                    elif (abs(self.target_ticks - sample['position_ticks']) > 4
                          and now - self.last_progress > 1.5 and not self.motion_warning):
                        warning = 'Joint has not progressed toward its target; keeping the goal and torque'
                if now >= self.next_torque_check:
                    if self.driver.read_torque() != 1:
                        raise ValueError('Joint torque unexpectedly disabled; no automatic re-enable')
                    self.next_torque_check = now + 0.5
            except (OSError, ValueError, RuntimeError) as exc:
                reason = str(exc)
            if not reason and self.busy:
                if now >= self.arrival_warning_at and not self.motion_warning:
                    warning = 'Joint has not arrived by the expected time; keeping the goal and torque'
                if abs(sample['position_ticks'] - self.target_ticks) <= 4 and not sample['moving']:
                    self.settled += 1
                else:
                    self.settled = 0
                reached = self.settled >= 3
        if reached:
            self.busy = False
            self.holding = True
            self.settled = 0
            self.hold_deviation_warned = False
            self.hold_deviation_count = 0
            self.motion_warning = False
            self.last_outcome = 'holding'
            self.last_message = '受控关节已到位并保持上力；保护故障或节点关闭时可能脱力。'
            return ('info', self.last_message)
        if not reason:
            if warning or load_warning:
                if warning:
                    self.motion_warning = True
                self.last_message = load_warning or warning
                return ('warning', self.last_message + '; diagnostic=' + json.dumps({
                    'start_ticks': self.start_ticks,
                    'target_ticks': self.target_ticks,
                    'latest_feedback': self.recent_feedback[-1],
                }, separators=(',', ':'), allow_nan=True))
            if self.holding:
                deviation = abs(sample['position_ticks'] - self.target_ticks)
                self.hold_deviation_count = (
                    self.hold_deviation_count + 1 if deviation > HOLD_WARNING_TICKS else 0
                )
                if (self.hold_deviation_count >= HOLD_WARNING_SAMPLES
                        and not self.hold_deviation_warned):
                    self.hold_deviation_warned = True
                    self.last_message = (f'Holding position deviated by {deviation} ticks; '
                                         'goal remains unchanged and torque stays enabled')
                    return ('warning', self.last_message + '; diagnostic=' + json.dumps({
                        'target_ticks': self.target_ticks,
                        'latest_feedback': self.recent_feedback[-1],
                    }, separators=(',', ':'), allow_nan=True))
                if deviation <= HOLD_RECOVERY_TICKS and self.hold_deviation_warned:
                    self.hold_deviation_warned = False
                    self.last_message = f'Holding position recovered within {HOLD_RECOVERY_TICKS} ticks'
                    return ('info', self.last_message)
            return None
        diagnostic = '; diagnostic=' + json.dumps({
            'start_ticks': self.start_ticks,
            'target_ticks': self.target_ticks,
            'recent_feedback': list(self.recent_feedback),
        }, separators=(',', ':'), allow_nan=True)
        try:
            self.release()
        except (OSError, ValueError, RuntimeError) as exc:
            self.busy = False  # Remain owned/faulted; require explicit recovery.
            self.holding = False
            self.last_outcome = 'failed'
            self.last_message = f'{reason}; TORQUE-OFF UNCONFIRMED: {exc}{diagnostic}'
            return ('error', self.last_message)
        self.fault = reason
        self.last_outcome = 'failed'
        self.last_message = f'{reason}; 已确认受控关节脱力，检查后重新准备控制器。{diagnostic}'
        return ('error', self.last_message)


class ArmBusHold:
    """Explicit seven-joint startup transaction on one owned bus.

    Preflight performs reads only. Sequential enabling is not atomic; on any
    failure every joint touched by this transaction is released best-effort.
    Callers must ensure people support the arm and supervise the real device.
    """

    def __init__(self, side, controls, *, max_prepare_age_s,
                 write_group_goals, read_group_snapshot, clock=time.monotonic):
        if side not in ('left', 'right'):
            raise ValueError('Unknown arm side')
        self.names = tuple(arm_names(side))
        if set(controls) != set(self.names):
            raise ValueError('Exactly seven named arm controls are required')
        if (isinstance(max_prepare_age_s, bool) or not isinstance(max_prepare_age_s, (int, float))
                or not math.isfinite(max_prepare_age_s) or max_prepare_age_s <= 0):
            raise ValueError('max_prepare_age_s must be positive and finite')
        if not callable(write_group_goals) or not callable(read_group_snapshot):
            raise ValueError('Arm group requires batch goal and feedback interfaces')
        self.controls = dict(controls)
        for index, name in enumerate(self.names):
            control = self.controls[name]
            if (not isinstance(control, JointControl)
                    or control.driver.bus != side or control.driver.joint_id != index + 1
                    or control.entry['id'] != index + 1):
                raise ValueError(f'{name}: unexpected controller identity')
        self.clock = clock
        self.write_group_goals = write_group_goals
        self.read_group_snapshot = read_group_snapshot
        self.max_prepare_age_s = float(max_prepare_age_s)
        self.prepared = None
        self.prepared_at = None
        self.state = 'idle'
        self.reason = ''

    def inspect_hold_positions(self):
        snapshots = self.read_group_snapshot()
        plans = []
        for name in self.names:
            try:
                control = self.controls[name]
                snapshot = snapshots[control.entry['id']]
                plans.append(control.inspect_hold_current(control_snapshot=snapshot))
            except (OSError, ValueError, RuntimeError) as exc:
                raise ValueError(f'{name}: preflight failed: {exc}') from exc
        return tuple(plans)

    def prepare(self):
        if self.state != 'idle':
            raise ValueError('Arm group is not idle')
        self.prepared = None
        self.prepared_at = None
        started = self.clock()
        if isinstance(started, bool) or not isinstance(started, (float, int)) or not math.isfinite(started):
            raise ValueError('Invalid monotonic clock')
        plans = self.inspect_hold_positions()
        now = self.clock()
        if (isinstance(now, bool) or not isinstance(now, (float, int)) or not math.isfinite(now)
                or not 0 <= now - started <= self.max_prepare_age_s):
            raise ValueError('Startup inspection expired or clock moved backwards')
        self.prepared = tuple(plans)
        self.prepared_at = float(started)
        self.state = 'prepared'
        return self.prepared

    def start(self, *, deadline=None):
        if self.state != 'prepared' or self.prepared is None:
            raise ValueError('Arm group has no prepared startup')
        now = self.clock()
        if (isinstance(now, bool) or not isinstance(now, (float, int)) or not math.isfinite(now)
                or not 0 <= now - self.prepared_at <= self.max_prepare_age_s
                or (deadline is not None and now > deadline)):
            self.state = 'idle'
            self.prepared = None
            raise ValueError('Startup inspection expired or clock moved backwards')
        plans = self.prepared
        self.prepared = None  # one attempt only; never replay a prepare
        self.state = 'starting'
        try:
            # Freshly verify all seven before the first write.
            fresh = self.inspect_hold_positions()
            for name, plan, current in zip(self.names, plans, fresh):
                if abs(current.target_ticks - plan.target_ticks) > 2:
                    raise ValueError(f'{name}: position changed after inspection')
            for name, plan in zip(self.names, plans):
                now = self.clock()
                if (isinstance(now, bool) or not isinstance(now, (float, int))
                        or not math.isfinite(now)
                        or not 0 <= now - self.prepared_at <= self.max_prepare_age_s
                        or (deadline is not None and now > deadline)):
                    raise ValueError('Startup inspection expired or clock moved backwards')
                self.controls[name].hold_current_position(expected_tick=plan.target_ticks)
        except BaseException as exc:
            failed = []
            for name in reversed(self.names):
                control = self.controls[name]
                if control.owned:
                    try:
                        control.release()
                    except (OSError, ValueError, RuntimeError) as release_exc:
                        failed.append(f'{name}: {release_exc}')
            self.state = 'fault'
            self.reason = str(exc) + ('; torque-off unconfirmed: ' + '; '.join(failed) if failed else '')
            if not isinstance(exc, Exception):
                if hasattr(exc, 'add_note'):
                    exc.add_note(self.reason)
                raise
            raise RuntimeError(self.reason) from exc
        self.state = 'holding'
        return plans

    def inspect_targets(self, targets):
        """Read-only validation of all seven absolute targets on this bus."""
        if self.state != 'holding':
            raise ValueError('Arm group is not holding')
        if len(targets) != len(self.names):
            raise ValueError('Exactly seven targets are required')
        snapshots = self.read_group_snapshot()
        plans = []
        for name, target in zip(self.names, targets):
            try:
                control = self.controls[name]
                snapshot = snapshots[control.entry['id']]
                plans.append(control.inspect_absolute_target(
                    target, control_snapshot=snapshot,
                ))
            except (OSError, ValueError, RuntimeError) as exc:
                raise ValueError(f'{name}: target inspection failed: {exc}') from exc
        return tuple(plans)

    def cancel_prepare(self):
        """Discard an uncommitted, read-only startup inspection."""
        if self.state != 'prepared':
            raise ValueError('No uncommitted startup inspection')
        self.prepared = None
        self.prepared_at = None
        self.state = 'idle'

    def protective_release(self, reason):
        """Best-effort release owned joints; never claim success for lost replies."""
        failed = []
        for name in reversed(self.names):
            control = self.controls[name]
            if control.owned:
                try:
                    control.release()
                except (OSError, ValueError, RuntimeError) as exc:
                    failed.append(f'{name}: {exc}')
        self.prepared = None
        self.prepared_at = None
        self.state = 'fault'
        self.reason = str(reason)
        if failed:
            self.reason += '; torque-off unconfirmed: ' + '; '.join(failed)
            raise RuntimeError(self.reason)

    def apply_targets(self, targets, *, deadline=None):
        """Preflight seven targets, then dispatch changed goals as one batch.

        If dispatch fails, best-effort release all owned joints and latch the
        group fault; a failed or interrupted write has an unknown outcome.
        """
        targets = tuple(targets)
        plans = self.inspect_targets(targets)
        def check_deadline():
            if deadline is not None:
                now = self.clock()
                if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
                    raise ValueError('Invalid transaction clock')
                if now > deadline:
                    raise ValueError(TARGET_COMMAND_EXPIRED)
        check_deadline()  # Reject an expired transaction before the first write.
        attempted = []
        try:
            changed = [(name, plan) for name, plan in zip(self.names, plans)
                       if plan.target_ticks != self.controls[name].target_ticks]
            if changed:
                goals = {self.controls[name].driver.joint_id: plan.target_ticks
                         for name, plan in changed}
                check_deadline()
                attempted.extend(name for name, _ in changed)
                self.write_group_goals(goals, deadline=deadline)
                for name, plan in changed:
                    control = self.controls[name]
                    measured_tick = denormalize_tick(control.entry, plan.current_normalized)
                    control._record_goal(plan.target_ticks, measured_tick)
        except BaseException as exc:
            if isinstance(exc, MotionSessionEnded):
                raise
            if isinstance(exc, CommandExpired):
                raise ValueError(TARGET_COMMAND_EXPIRED) from exc
            if not attempted and isinstance(exc, ValueError) and str(exc) == TARGET_COMMAND_EXPIRED:
                raise  # The deadline passed during inspection, before a goal write.
            failed = []
            for name in reversed(self.names):
                control = self.controls[name]
                if control.owned:
                    try:
                        control.release()
                    except (OSError, ValueError, RuntimeError) as release_exc:
                        failed.append(f'{name}: {release_exc}')
            self.state = 'fault'
            self.reason = (f'target dispatch failed; attempted={attempted}; '
                           f'outcome of last attempt may be unknown: {exc}')
            if failed:
                self.reason += '; torque-off unconfirmed: ' + '; '.join(failed)
            if not isinstance(exc, Exception):
                if hasattr(exc, 'add_note'):
                    exc.add_note(self.reason)
                raise
            raise RuntimeError(self.reason) from exc
        return plans


class ArmBusTransaction:
    """A single-use, bounded transaction on one bus; transport independent."""

    def __init__(self, group, *, timeout_s, clock=time.monotonic):
        if not isinstance(group, ArmBusHold):
            raise TypeError('ArmBusHold is required')
        if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError('Transaction timeout must be positive and finite')
        self.group = group
        self.timeout_s = float(timeout_s)
        self.clock = clock
        self.pending = None
        self.last_token = 0

    def _now(self):
        now = self.clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise ValueError('Invalid monotonic clock')
        return float(now)

    def _token(self, token):
        if (not isinstance(token, str) or not 1 <= len(token) <= 20
                or not token.isascii() or not token.isdecimal()
                or token[0] == '0' or int(token) >= 2**64):
            raise ValueError('Invalid transaction ID')

    def _new_token(self, token):
        self._token(token)
        if int(token) <= self.last_token:
            raise ValueError('Transaction ID was already used or went backwards')

    def _discard_pending(self):
        if self.pending and self.group.state == 'prepared':
            self.group.cancel_prepare()
        self.pending = None

    def _expire(self):
        if self.pending:
            _, started, deadline = self.pending
            now = self._now()
            if now < started or now > deadline:
                self._discard_pending()

    def prepare_hold(self, token):
        self._new_token(token)
        self._expire()
        if self.pending:
            raise ValueError('Another transaction is pending')
        started = self._now()
        self.group.prepare()  # no writes
        now = self._now()
        if not started <= now <= started + self.timeout_s:
            self.group.cancel_prepare()
            raise ValueError('Hold inspection expired')
        self.pending = (token, started, started + self.timeout_s)
        self.last_token = int(token)

    def commit_hold(self, token):
        self._token(token)
        self._expire()
        if not self.pending or self.pending[0] != token:
            raise ValueError('No matching fresh hold transaction')
        _, _, deadline = self.pending
        self.pending = None  # A timed-out reply cannot repeat an enable.
        return self.group.start(deadline=deadline)

    def apply_stream_targets(self, token, targets, *, deadline):
        """Validate fresh feedback and write one new streaming request once.

        The group must already own all seven joints. Startup still uses the
        explicit prepare/commit transaction; continuous commands need one fresh
        inspection, not two inspections separated by another ROS round trip.
        """
        self._new_token(token)
        self._expire()
        if self.pending:
            raise ValueError('Another transaction is pending')
        now = self._now()
        if (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                or not math.isfinite(deadline) or deadline <= now):
            raise ValueError(TARGET_COMMAND_EXPIRED)
        self.last_token = int(token)  # Consume even a rejected request; never replay.
        return self.group.apply_targets(tuple(targets), deadline=min(deadline, now + self.timeout_s))

    def abort(self, token):
        self._token(token)
        if not self.pending or self.pending[0] != token:
            self._expire()
            raise ValueError('No matching fresh transaction')
        self._discard_pending()

    def protective_release(self, reason):
        self._discard_pending()
        self.group.protective_release(reason)
