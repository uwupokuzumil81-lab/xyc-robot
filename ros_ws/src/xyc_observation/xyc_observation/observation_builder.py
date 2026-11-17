# SPDX-License-Identifier: Apache-2.0
"""Schema, asset identity, freshness and sample validation. No hardware access."""

from dataclasses import dataclass
import hashlib
import json
import math

from builtin_interfaces.msg import Time
from xyc_interfaces.msg import Observation


CAMERAS = ('forward', 'chest', 'wrist_left', 'wrist_right')
JOINT_SUFFIXES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex',
                  'wrist_yaw', 'wrist_roll', 'gripper')
JOINTS = tuple(f'arm_{arm}_{joint}' for arm in ('left', 'right') for joint in JOINT_SUFFIXES)
MOBILE_NAMES = ('lift.height_m', 'base.vx_m_s', 'base.vy_m_s', 'base.wz_rad_s')
MOBILE_UNITS = ('m', 'm/s', 'm/s', 'rad/s')
STATE_NAMES = JOINTS + MOBILE_NAMES
STATE_UNITS = ('normalized',) * 14 + MOBILE_UNITS


def content_id(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def stamp_ns(stamp):
    if stamp.sec < 0 or not 0 <= stamp.nanosec < 10**9:
        raise ValueError('Malformed timestamp')
    return stamp.sec * 10**9 + stamp.nanosec


def time_message(nanoseconds):
    return Time(sec=int(nanoseconds // 10**9), nanosec=int(nanoseconds % 10**9))


class ObservationError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__('; '.join(self.errors))


@dataclass(frozen=True)
class ObservationConfig:
    camera_names: tuple = CAMERAS
    camera_topics: tuple = tuple(f'/camera/{name}/image_raw' for name in CAMERAS)
    camera_frame_ids: tuple = tuple(f'camera_{name}_optical_frame' for name in CAMERAS)
    joint_names: tuple = JOINTS
    mobile_state_topic: str = '/robot/chassis/state'
    mobile_frame_id: str = 'base_link'
    width: int = 640
    height: int = 480
    encoding: str = 'bgr8'
    publish_hz: float = 30.0
    sync_queue_size: int = 8
    sync_slop_s: float = 0.04
    max_time_skew_s: float = 0.05
    max_age_s: float = 0.25
    future_tolerance_s: float = 0.02

    def validate(self):
        if tuple(self.camera_names) != CAMERAS:
            raise ValueError('Schema v2 camera order is forward, chest, wrist_left, wrist_right')
        if (len(self.camera_topics) != 4 or len(set(self.camera_topics)) != 4
                or len(self.camera_frame_ids) != 4 or len(set(self.camera_frame_ids)) != 4):
            raise ValueError('Provide four unique camera topics/frame IDs')
        if any(not name for name in self.camera_names + self.camera_topics + self.camera_frame_ids):
            raise ValueError('Camera identities must not be empty')
        if tuple(self.joint_names) != JOINTS:
            raise ValueError('Require 14 joints: left seven, then right seven')
        if not self.mobile_state_topic or not self.mobile_frame_id:
            raise ValueError('Mobile state topic and frame ID must not be empty')
        if (type(self.width) is not int or type(self.height) is not int
                or self.width <= 0 or self.height <= 0 or self.encoding not in ('bgr8', 'rgb8')):
            raise ValueError('Expected positive image dimensions and bgr8/rgb8 encoding')
        if type(self.sync_queue_size) is not int or not 2 <= self.sync_queue_size <= 30:
            raise ValueError('sync_queue_size must be in 2..30 (bounded image memory)')
        for value in (self.publish_hz, self.sync_slop_s, self.max_time_skew_s,
                      self.max_age_s, self.future_tolerance_s):
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError('Frequency and timing limits must be finite and positive')
        if not self.sync_slop_s <= self.max_time_skew_s < self.max_age_s:
            raise ValueError('Require sync_slop <= max_time_skew < max_age')
        return self


class ObservationBuilder:
    def __init__(self, config, calibrations):
        self.config = config.validate()
        if set(calibrations) != {'left', 'right'}:
            raise ValueError('Require independent left/right calibration assets')
        self.calibrations = calibrations
        self.calibration_ids = [content_id(calibrations[arm]) for arm in ('left', 'right')]
        self.state_units = list(STATE_UNITS)
        self.bounds = {}
        for index, arm in enumerate(('left', 'right')):
            document = calibrations[arm]
            expected_names = self.config.joint_names[index * 7:(index + 1) * 7]
            if (document.get('schema_version') != 1 or document.get('arm') != arm
                    or set(document.get('motors', {})) != set(expected_names)):
                raise ValueError(f'{arm}: calibration asset identity/schema mismatch')
            for name in expected_names:
                entry = document['motors'][name]
                lower = 0 if name.endswith('_gripper') else -100
                if entry['normalized_min'] != lower or entry['normalized_max'] != 100:
                    raise ValueError(f'{name}: unsupported state normalization')
                self.bounds[name] = (lower, 100)
        self.schema = {
            'version': 2, 'camera_names': list(config.camera_names),
            'camera_topics': list(config.camera_topics),
            'camera_frame_ids': list(config.camera_frame_ids),
            'image_size': [config.width, config.height], 'encoding': config.encoding,
            'state_names': list(STATE_NAMES), 'state_units': self.state_units,
            'mobile_state_topic': config.mobile_state_topic,
            'mobile_frame_id': config.mobile_frame_id,
            'calibration_ids': self.calibration_ids, 'publish_hz': config.publish_hz,
            'max_time_skew_s': config.max_time_skew_s, 'max_age_s': config.max_age_s,
        }
        self.schema_id = content_id(self.schema)

    def _time_errors(self, stamp, now_ns, label):
        try:
            value = stamp_ns(stamp)
        except ValueError as exc:
            return [f'{label}: {exc}']
        if value <= 0:
            return [f'{label}: missing timestamp']
        age = (now_ns - value) / 1e9
        if age > self.config.max_age_s:
            return [f'{label}: stale ({age:.3f}s)']
        if age < -self.config.future_tolerance_s:
            return [f'{label}: timestamp is in the future ({-age:.3f}s)']
        return []

    def inspect_source(self, index, message, now_ns):
        """Indices 0..3 are cameras, 4..5 arms and 6 is measured mobile state."""
        if message is None:
            return ['no message received']
        errors = self._time_errors(message.header.stamp, now_ns, 'header')
        if index < 4:
            if message.header.frame_id != self.config.camera_frame_ids[index]:
                errors.append('camera frame_id mismatch')
            if (message.width != self.config.width or message.height != self.config.height
                    or message.encoding != self.config.encoding):
                errors.append('image dimensions/encoding mismatch')
            if message.step < message.width * 3 or len(message.data) != message.height * message.step:
                errors.append('image byte length/step mismatch')
            return errors
        if index == 6:
            if message.header.frame_id != self.config.mobile_frame_id:
                errors.append('mobile frame_id mismatch')
            if (len(message.names) != 4 or len(message.state) != 4 or len(message.units) != 4
                    or set(message.names) != set(MOBILE_NAMES)):
                return errors + ['mobile arrays/names must contain the four expected states']
            if len(set(message.names)) != 4:
                return errors + ['mobile state names must be unique']
            mapping = {name: position for position, name in enumerate(message.names)}
            for name, unit in zip(MOBILE_NAMES, MOBILE_UNITS):
                position = mapping[name]
                value = message.state[position]
                if message.units[position] != unit:
                    errors.append(f'{name}: unit mismatch')
                if not math.isfinite(value):
                    errors.append(f'{name}: nonfinite state')
                if name == 'lift.height_m' and math.isfinite(value) and not 0. <= value <= .6:
                    errors.append(f'{name}: outside verified 0..0.6 m range')
            if not message.valid or message.error:
                errors.append('mobile state invalid' + (f' ({message.error})' if message.error else ''))
            if message.homing:
                errors.append('lift homing is active')
            return errors
        if index not in (4, 5):
            return errors + ['unknown observation source index']
        arm = ('left', 'right')[index - 4]
        expected = self.config.joint_names[(index - 4) * 7:(index - 3) * 7]
        if message.arm != arm:
            errors.append('arm identity mismatch')
        if message.calibration_id != self.calibration_ids[index - 4]:
            errors.append('calibration_id mismatch')
        lengths = (len(message.names), len(message.position_normalized), len(message.valid),
                   len(message.errors), len(message.sample_stamps))
        if any(length != 7 for length in lengths) or set(message.names) != set(expected):
            return errors + ['joint arrays/names must contain the seven expected joints']
        if not message.complete:
            errors.append('arm complete=false')
        for name, value, valid, error, stamp in zip(
            message.names, message.position_normalized, message.valid,
            message.errors, message.sample_stamps,
        ):
            if not valid or error:
                errors.append(f'{name}: invalid ({error})')
            lower, upper = self.bounds[name]
            if not math.isfinite(value) or not lower <= value <= upper:
                errors.append(f'{name}: nonfinite/out-of-range state')
            errors.extend(self._time_errors(stamp, now_ns, name))
        return errors

    def build(self, images, arms, mobile, *, now_ns, sequence):
        if len(images) != 4 or len(arms) != 2 or mobile is None:
            raise ObservationError(['Require four images, two arm messages and one mobile state'])
        messages = tuple(images) + tuple(arms) + (mobile,)
        errors = []
        for index, message in enumerate(messages):
            label = (self.config.camera_names[index] if index < 4 else
                     ('left', 'right')[index - 4] if index < 6 else 'mobile')
            errors.extend(f'{label}: {error}' for error in self.inspect_source(index, message, now_ns))
        if errors:
            raise ObservationError(errors)
        all_stamps = [stamp_ns(message.header.stamp) for message in messages]
        all_stamps += [stamp_ns(stamp) for message in arms for stamp in message.sample_stamps]
        skew = (max(all_stamps) - min(all_stamps)) / 1e9
        if skew > self.config.max_time_skew_s:
            raise ObservationError([f'source time skew {skew:.6f}s exceeds limit'])
        state, state_stamps = [], []
        for index, message in enumerate(arms):
            mapping = {name: position for position, name in enumerate(message.names)}
            for name in self.config.joint_names[index * 7:(index + 1) * 7]:
                position = mapping[name]
                state.append(message.position_normalized[position])
                state_stamps.append(message.sample_stamps[position])
        mobile_mapping = {name: position for position, name in enumerate(mobile.names)}
        for name in MOBILE_NAMES:
            state.append(mobile.state[mobile_mapping[name]])
            state_stamps.append(mobile.header.stamp)
        observation = Observation()
        observation.header.stamp = images[0].header.stamp
        observation.published_at = time_message(now_ns)
        observation.sequence = sequence
        observation.schema_version = 2
        observation.schema_id = self.schema_id
        observation.camera_names = list(self.config.camera_names)
        observation.camera_topics = list(self.config.camera_topics)
        observation.images = list(images)
        observation.state_names = list(STATE_NAMES)
        observation.state = state
        observation.state_units = list(self.state_units)
        observation.state_stamps = state_stamps
        observation.arm_batch_stamps = [message.header.stamp for message in arms]
        observation.mobile_batch_stamp = mobile.header.stamp
        observation.calibration_ids = list(self.calibration_ids)
        observation.max_time_skew_s = skew
        observation.oldest_source_age_s = max(0., (now_ns - min(all_stamps)) / 1e9)
        observation.valid = True
        return observation
