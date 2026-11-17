"""Exclusive-owner motion replay protocol; no ROS imports or direct motor access."""

import json
import math

PROTOCOL = 'xyc.motion_replay.v1'
PORT = 17886
MAX_FRAME = 65536
SIDES = ('left', 'right')
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper')
ARM_NAMES = tuple(f'arm_{side}_{joint}' for side in SIDES for joint in JOINTS)
ACTION_NAMES = ARM_NAMES + ('lift.command', 'base.vx_command', 'base.vy_command', 'base.wz_command')
STATE_NAMES = ARM_NAMES + ('lift.height_m', 'base.vx_m_s', 'base.vy_m_s', 'base.wz_rad_s')
ACTION_UNITS = ('normalized',) * 14 + ('ternary',) * 4
STATE_UNITS = ('normalized',) * 14 + ('m', 'm/s', 'm/s', 'rad/s')
MOBILE_ACTION_SCALE = (0.015, 0.15, 0.15, 0.60)


def send(channel, message):
    encoded = json.dumps(message, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode() + b'\n'
    if len(encoded) > MAX_FRAME:
        raise ValueError('Replay message is too large')
    channel.write(encoded)
    channel.flush()


def receive(channel):
    raw = channel.readline(MAX_FRAME + 1)
    if not raw:
        raise EOFError('Replay connection closed')
    if len(raw) > MAX_FRAME or not raw.endswith(b'\n'):
        raise ValueError('Replay message is incomplete or too large')
    result = json.loads(raw)
    if not isinstance(result, dict) or type(result.get('sequence')) is not int:
        raise ValueError('Replay message needs an integer sequence')
    return result


def checked_action(values):
    if (not isinstance(values, (list, tuple)) or len(values) != 18
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError('Robot control requires 18 finite canonical actions')
    values = [float(v) for v in values]
    for index, value in enumerate(values[:14]):
        if not (0 if index % 7 == 6 else -100) <= value <= 100:
            raise ValueError(f'Arm action {index} is outside its calibrated range')
    for index, value in enumerate(values[14:], 14):
        command = min((-1., 0., 1.), key=lambda item: abs(item - value))
        if abs(value - command) > 1e-6:
            raise ValueError(f'Mobile action {index} must be exactly -1, 0 or +1')
        values[index] = command
    return values


def physical_mobile_action(values):
    result = []
    for value, scale in zip(values, MOBILE_ACTION_SCALE):
        value = float(value)
        command = min((-1., 0., 1.), key=lambda item: abs(item - value))
        if not math.isfinite(value) or abs(value - command) > 1e-6:
            raise ValueError('Expected four fixed -1/0/+1 lift/base commands')
        result.append(command * scale)
    return result


class ReplaySession:
    """Exactly one client owns one replay; unknown outcomes are never retried.

    The injected robot implements snapshot, start, home, target, renew and stop.
    status is read-only. All writes require a successful prepare handshake.
    """
    def __init__(self, robot):
        self.robot = robot
        self.sequence = 0
        self.prepared = False
        self.started = False
        self.calibration_ids = None

    def process(self, message):
        sequence = message.get('sequence')
        if type(sequence) is not int or not self.sequence < sequence < 2**63:
            raise ValueError('Replay sequence repeated or went backwards')
        self.sequence = sequence
        operation = message.get('operation')
        if operation == 'status':
            return self.robot.snapshot()
        if operation == 'prepare':
            if self.prepared or message.get('protocol') != PROTOCOL:
                raise ValueError('Replay protocol mismatch or repeated preparation')
            if (tuple(message.get('action_names', [])) != ACTION_NAMES
                    or tuple(message.get('action_units', [])) != ACTION_UNITS
                    or tuple(message.get('state_names', [])) != STATE_NAMES
                    or tuple(message.get('state_units', [])) != STATE_UNITS):
                raise ValueError('Dataset state/action names, order or units differ from the robot')
            state = self.robot.snapshot()
            ids = message.get('calibration_ids')
            if not state.get('ready') or ids != state.get('calibration_ids'):
                raise ValueError('Robot feedback unavailable or follower calibration differs from the dataset')
            self.prepared, self.calibration_ids = True, ids
            return state
        if not self.prepared:
            raise ValueError('Prepare the replay before issuing commands')
        if operation in ('stop', 'finish'):
            self.robot.stop(freeze_arms=operation != 'finish')
            self.started = False
            return self.robot.snapshot()
        if operation == 'start':
            if self.started:
                raise ValueError('Replay already started')
            self.started = True
            self.robot.start(self.calibration_ids)
            return self.robot.snapshot()
        if not self.started:
            raise ValueError('Replay is not started')
        state = self.robot.snapshot()
        if state.get('calibration_ids') != self.calibration_ids:
            raise ValueError('Follower calibration changed during replay')
        if not state.get('ready'):
            raise ValueError('Replay feedback not ready: ' + str(state.get('reason') or 'missing feedback'))
        if operation == 'home':
            self.robot.home()
        elif operation == 'heartbeat':
            self.robot.renew()
        elif operation == 'target':
            self.robot.target(checked_action(message.get('action')), self.calibration_ids)
        else:
            raise ValueError('Unknown replay operation')
        return self.robot.snapshot()

    def disconnect(self):
        try:
            if self.started:
                self.robot.stop()
        finally:
            self.started = False
