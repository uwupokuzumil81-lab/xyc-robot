"""Names and robot data contracts shared across learning backends."""

from dataclasses import asdict, dataclass
import math

ACTION = 'action'
OBS_STATE = 'observation.state'
OBS_IMAGES = 'observation.images'
OBS_ENV_STATE = 'observation.environment_state'
CAMERAS = ('forward', 'chest', 'wrist_left', 'wrist_right')
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_yaw', 'wrist_roll', 'gripper')
ARM_NAMES = tuple(f'arm_{side}_{joint}' for side in ('left', 'right') for joint in JOINTS)
STATE_NAMES = ARM_NAMES + ('lift.height_m', 'base.vx_m_s', 'base.vy_m_s', 'base.wz_rad_s')
ACTION_NAMES = ARM_NAMES + ('lift.command', 'base.vx_command', 'base.vy_command', 'base.wz_command')

# The last four actions are direction commands, not ratios or speed gears.
# The robot backend owns the only command -> fixed physical velocity mapping.
MOBILE_ACTION_SCALE = (0.015, 0.15, 0.15, 0.60)
ACTION_UNITS = ('normalized',) * 14 + ('ternary',) * 4
STATE_UNITS = ('normalized',) * 14 + ('m', 'm/s', 'm/s', 'rad/s')


def ternary_mobile_action(values, *, threshold=.5):
    """Quantize model outputs to the only legal mobile commands: -1, 0, +1."""
    if len(values) != 4:
        raise ValueError('Expected lift, base-x, base-y and base-yaw commands')
    values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Mobile action contains a non-finite value')
    if not 0 < threshold <= 1:
        raise ValueError('Ternary threshold must be in (0,1]')
    return [1. if value >= threshold else -1. if value <= -threshold else 0. for value in values]


def normalized_mobile_action(values):
    """Convert legacy physical velocities to direction-only ternary commands."""
    if len(values) != 4:
        raise ValueError('Expected lift, base-x, base-y and base-yaw velocities')
    result = []
    for value, scale in zip(values, MOBILE_ACTION_SCALE):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Mobile action contains a non-finite value')
        result.append(0. if abs(value) <= scale * 1e-4 else 1. if value > 0 else -1.)
    return result


def physical_mobile_action(values):
    """Map strict ternary commands to fixed physical robot velocities."""
    result = []
    for value, scale in zip(values, MOBILE_ACTION_SCALE):
        value = float(value)
        command = min((-1., 0., 1.), key=lambda item: abs(item - value))
        if not math.isfinite(value) or abs(value - command) > 1e-6:
            raise ValueError('Mobile controls must be exactly -1, 0 or +1')
        result.append(command * scale)
    return result


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


def robot_action(values, schema):
    """Map one canonical model output; this function performs no I/O."""
    expected_units = ACTION_UNITS
    if (tuple(schema.action_names) != ACTION_NAMES or len(values) != 18
            or tuple(schema.action_units) != expected_units
            or not all(math.isfinite(float(v)) for v in values)):
        raise ValueError('Expected the XYC 18-dimensional arm-plus-ternary action contract')
    arms = [float(v) for v in values[:14]]
    for index, value in enumerate(arms):
        lower = 0. if index % 7 == 6 else -100.
        if not lower <= value <= 100.:
            raise ValueError(f'Arm action {index} is outside its calibrated range')
    mobile = ternary_mobile_action(values[14:])
    return {'arm_position_normalized': [float(v) for v in values[:14]],
            'mobile_command': {'lift': mobile[0], 'x': mobile[1], 'y': mobile[2], 'yaw': mobile[3]},
            'action_period_s': 1. / schema.fps}
