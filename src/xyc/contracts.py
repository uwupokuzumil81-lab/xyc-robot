"""Canonical robot contracts shared by learning, recording and ROS transport."""

from dataclasses import asdict, dataclass
import math
from numbers import Real

ACTION = 'action'
OBS_STATE = 'observation.state'
OBS_IMAGES = 'observation.images'
OBS_ENV_STATE = 'observation.environment_state'
CAMERAS = ('forward', 'chest', 'wrist_left', 'wrist_right')
SIDES = ('left', 'right')
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper')
ARM_NAMES = tuple(f'arm_{side}_{joint}' for side in SIDES for joint in JOINTS)
STATE_NAMES = ARM_NAMES + ('lift.height_m', 'base.vx_m_s', 'base.vy_m_s', 'base.wz_rad_s')
ACTION_NAMES = ARM_NAMES + ('lift.command', 'base.vx_command', 'base.vy_command', 'base.wz_command')

# The last four actions are direction commands, not ratios or speed gears.
# The robot backend owns the only command -> fixed physical velocity mapping.
MOBILE_ACTION_SCALE = (0.015, 0.15, 0.15, 0.60)
COMMAND_VALIDITY_S = .25
LIFT_SOFT_STOP_MARGIN_M = .005
ACTION_SEMANTICS = 'observation_then_new_ternary_command_v4'
ACTION_UNITS = ('normalized',) * 14 + ('ternary',) * 4
STATE_UNITS = ('normalized',) * 14 + ('m', 'm/s', 'm/s', 'rad/s')


class CommandExpired(TimeoutError):
    """The command deadline passed before a hardware write."""


def _mobile_values(values):
    try:
        values = list(values)
    except TypeError as exc:
        raise ValueError('Mobile actions need four finite numeric commands') from exc
    if (len(values) != 4 or any(isinstance(v, bool) or not isinstance(v, Real)
                              or not math.isfinite(v) for v in values)):
        raise ValueError('Mobile actions need four finite numeric commands')
    return values


def checked_mobile_action(values):
    """Validate four canonical direction labels without quantizing predictions."""
    values = _mobile_values(values)
    commands = []
    for value in values:
        command = min((-1., 0., 1.), key=lambda item: abs(item - value))
        if abs(value - command) > 1e-6:
            raise ValueError('Mobile actions must be exactly -1, 0 or +1')
        commands.append(command)
    return commands


def mobile_action_metadata():
    return dict(zip(('lift_m_s', 'base_x_m_s', 'base_y_m_s', 'base_yaw_rad_s'), MOBILE_ACTION_SCALE))


def check_mobile_action_scale(assets):
    if assets.get('mobile_action_scale') != mobile_action_metadata():
        raise ValueError('Recorded mobile action speeds are missing or differ from the current robot')


def ternary_mobile_action(values, *, threshold=.5):
    """Quantize model outputs to the only legal mobile commands: -1, 0, +1."""
    values = _mobile_values(values)
    if not 0 < threshold <= 1:
        raise ValueError('Ternary threshold must be in (0,1]')
    return [1. if value >= threshold else -1. if value <= -threshold else 0. for value in values]


def checked_action(values):
    """Validate a strict 18-D command; do not quantize continuous model output."""
    if (not isinstance(values, (list, tuple)) or len(values) != 18
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError('Robot control requires 18 finite canonical actions')
    values = [float(v) for v in values]
    for index, value in enumerate(values[:14]):
        if not (0 if index % 7 == 6 else -100) <= value <= 100:
            raise ValueError(f'Arm action {index} is outside its calibrated range')
    values[14:] = checked_mobile_action(values[14:])
    return values


def physical_mobile_action(values):
    """Map strict ternary commands to fixed physical robot velocities."""
    return [command * scale for command, scale in zip(checked_mobile_action(values), MOBILE_ACTION_SCALE)]


@dataclass(frozen=True)
class PolicySchema:
    cameras: tuple[str, ...]
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    state_units: tuple[str, ...]
    action_units: tuple[str, ...]
    fps: float

    def __post_init__(self):
        if (not self.cameras or len(set(self.cameras)) != len(self.cameras)
                or not self.state_names or not self.action_names or not math.isfinite(self.fps) or self.fps <= 0
                or len(set(self.state_names)) != len(self.state_names)
                or len(set(self.action_names)) != len(self.action_names)
                or len(self.state_units) != len(self.state_names)
                or len(self.action_units) != len(self.action_names)):
            raise ValueError('Invalid policy input/output schema')

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{k: tuple(v) if isinstance(v, (tuple, list)) else v for k, v in value.items()})


def robot_action(values, schema, *, bound_arms=False):
    """Map one canonical model output; this function performs no I/O."""
    expected_units = ACTION_UNITS
    if (tuple(schema.action_names) != ACTION_NAMES or len(values) != 18
            or tuple(schema.action_units) != expected_units
            or not all(math.isfinite(float(v)) for v in values)):
        raise ValueError('Expected the XYC 18-dimensional arm-plus-ternary action contract')
    arms = [float(v) for v in values[:14]]
    for index, value in enumerate(arms):
        lower = 0. if index % 7 == 6 else -100.
        if bound_arms:
            arms[index] = min(100., max(lower, value))
            continue
        if not lower <= value <= 100.:
            raise ValueError(f'Arm action {index} is outside its calibrated range')
    mobile = ternary_mobile_action(values[14:])
    return {'arm_position_normalized': arms,
            'mobile_command': {'lift': mobile[0], 'x': mobile[1], 'y': mobile[2], 'yaw': mobile[3]},
            'action_period_s': 1. / schema.fps}
