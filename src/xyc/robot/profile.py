"""The single robot profile; override its path with XYC_ROBOT_CONFIG."""

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import os
import math
from pathlib import Path

import yaml

from xyc.contracts import CAMERAS, LIFT_SOFT_STOP_MARGIN_M, SIDES


@dataclass(frozen=True)
class RobotProfile:
    connection: dict
    ports: dict
    project_directory: str
    leaders: dict
    buses: dict
    cameras: dict
    geometry: dict
    preview_fps: float

    @property
    def ssh_host(self):
        return os.environ.get('XYC_ROBOT_HOST', self.connection['host'])

    @property
    def ssh_user(self):
        return os.environ.get('XYC_ROBOT_USER', self.connection['user'])

    @property
    def robot_root(self):
        path = Path(self.project_directory).expanduser()
        return path.resolve() if path.is_absolute() else (Path.home() / path).resolve()

    @property
    def image_size(self):
        settings = self.cameras[CAMERAS[0]]
        return settings['height'], settings['width']

    @property
    def lift_max_height_m(self):
        return self.geometry['lift_max_height_m']

    def motor_parameters(self, side):
        return {**self.buses[side], 'bus': side}


@lru_cache(maxsize=8)
def load_profile(path):
    value = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    expected = {'connection', 'ports', 'project_directory', 'leaders', 'buses',
                'cameras', 'geometry', 'preview_fps'}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError('Invalid robot profile fields')
    if (set(value['leaders']) != set(SIDES) or set(value['buses']) != set(SIDES)
            or set(value['cameras']) != set(CAMERAS)):
        raise ValueError('Robot profile must define both arms and all canonical cameras')
    sizes = {(camera['height'], camera['width']) for camera in value['cameras'].values()}
    if len(sizes) != 1 or any(type(v) is not int or v < 2 or v % 2 for v in next(iter(sizes))):
        raise ValueError('Recording requires a shared, positive even camera resolution')
    ports = value['ports']
    if (set(ports) != {'teleop', 'data', 'motion', 'recording'}
            or any(type(p) is not int or not 1024 <= p <= 65535 for p in ports.values())
            or len(set(ports.values())) != len(ports)):
        raise ValueError('Robot ports must be distinct integers in 1024..65535')
    if not isinstance(value['preview_fps'], (int, float)) or not 0 < value['preview_fps'] <= 60:
        raise ValueError('Invalid browser preview FPS')
    height = value['geometry']['lift_max_height_m']
    if type(height) not in (int, float) or not math.isfinite(height) or height <= 2 * LIFT_SOFT_STOP_MARGIN_M:
        raise ValueError('Lift travel must be finite and wider than the two soft-stop margins')
    return RobotProfile(**value)


def robot_profile():
    path = os.environ.get('XYC_ROBOT_CONFIG')
    return load_profile(str(Path(path).expanduser().resolve()) if path else
                        str(files('xyc.robot').joinpath('profile.yaml')))
