"""Render the platform's shared aligned observation as a ROS message."""

from xyc.io import content_id
from xyc.robot.profile import robot_profile
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import cv2
import numpy as np
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Image
from xyc.contracts import ARM_NAMES, CAMERAS, STATE_NAMES, STATE_UNITS
from xyc.observation import MAX_CAMERA_AGE_NS
from xyc_interfaces.msg import Observation




def time_message(nanoseconds):
    return Time(sec=int(nanoseconds // 10**9), nanosec=int(nanoseconds % 10**9))


@dataclass(frozen=True)
class ObservationConfig:
    camera_names: tuple = CAMERAS
    camera_topics: tuple = tuple(f'/camera/{name}/image_raw/compressed' for name in CAMERAS)
    camera_frame_ids: tuple = tuple(robot_profile().cameras[name]['frame_id'] for name in CAMERAS)
    state_topics: tuple = ('/motors/left/recording_sample', '/motors/right/recording_sample')
    width: int = robot_profile().image_size[1]
    height: int = robot_profile().image_size[0]
    encoding: str = 'bgr8'
    publish_hz: float = 30.0

    def validate(self):
        if tuple(self.camera_names) != CAMERAS:
            raise ValueError('Camera order must match the common platform contract')
        for values, count in ((self.camera_topics, 4), (self.camera_frame_ids, 4), (self.state_topics, 2)):
            if len(values) != count or len(set(values)) != count or not all(values):
                raise ValueError('Provide unique, nonempty camera and state identities')
        if (type(self.width) is not int or type(self.height) is not int
                or self.width < 1 or self.height < 1 or self.encoding not in ('bgr8', 'rgb8')):
            raise ValueError('Require positive image dimensions and bgr8/rgb8 encoding')
        if (type(self.publish_hz) not in (int, float) or not 1 <= self.publish_hz <= 30
                or self.publish_hz != int(self.publish_hz)):
            raise ValueError('Observation frequency must be an integer in 1..30 Hz')
        return self


class ObservationBuilder:
    """Own ROS serialization only; pairing and validity belong to ObservationBuffer."""

    def __init__(self, config, calibrations):
        self.config = config.validate()
        if set(calibrations) != {'left', 'right'}:
            raise ValueError('Require independent left/right calibration assets')
        for index, side in enumerate(('left', 'right')):
            document = calibrations[side]
            if (document.get('schema_version') != 1 or document.get('arm') != side
                    or set(document.get('motors', {})) != set(ARM_NAMES[index * 7:(index + 1) * 7])):
                raise ValueError('Calibration identity differs from the platform contract')
        self.calibration_ids = [content_id(calibrations[side]) for side in ('left', 'right')]
        self.schema_id = content_id({
            'version': 3, 'cameras': CAMERAS, 'state_names': STATE_NAMES,
            'state_units': STATE_UNITS, 'width': config.width, 'height': config.height,
            'encoding': config.encoding, 'calibration_ids': self.calibration_ids,
            'alignment': 'xyc.observation.ObservationBuffer',
        })

    def build(self, aligned, *, now_ns, sequence):
        if list(aligned.calibration_ids) != self.calibration_ids:
            raise ValueError('Aligned observation calibration differs from the saved assets')
        if not 0 <= now_ns - aligned.stamp_ns <= MAX_CAMERA_AGE_NS:
            raise ValueError('Aligned observation is stale or from the future')
        output = Observation()
        output.header.stamp = time_message(aligned.stamp_ns)
        output.published_at = time_message(now_ns)
        output.sequence, output.schema_version, output.schema_id = sequence, 3, self.schema_id
        output.camera_names, output.camera_topics = list(CAMERAS), list(self.config.camera_topics)
        for index, frame in enumerate(aligned.images):
            pixels = cv2.imdecode(np.frombuffer(frame.jpeg, np.uint8), cv2.IMREAD_COLOR)
            if pixels is None or pixels.shape != (self.config.height, self.config.width, 3):
                raise ValueError('Camera JPEG dimensions differ from the observation configuration')
            if self.config.encoding == 'rgb8':
                pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            message = Image()
            message.header.stamp = time_message(frame.stamp_ns)
            message.header.frame_id = self.config.camera_frame_ids[index]
            message.height, message.width = self.config.height, self.config.width
            message.encoding, message.step = self.config.encoding, self.config.width * 3
            message.data = pixels.tobytes()
            output.images.append(message)
        output.state_names, output.state_units = list(STATE_NAMES), list(STATE_UNITS)
        output.state = list(aligned.state)
        output.state_stamps = [time_message(stamp) for stamp in aligned.state_stamps_ns]
        output.arm_batch_stamps = [time_message(stamp) for stamp in aligned.arm_batch_stamps_ns]
        output.mobile_batch_stamp = time_message(aligned.mobile_batch_stamp_ns)
        output.calibration_ids = self.calibration_ids
        stamps = [*(frame.stamp_ns for frame in aligned.images), *aligned.state_stamps_ns]
        output.max_time_skew_s = (max(stamps) - min(stamps)) / 1e9
        oldest = min(stamps)
        output.oldest_source_age_s = (now_ns - oldest) / 1e9
        output.valid = True
        return output
