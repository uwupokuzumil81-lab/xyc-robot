# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from uuid import uuid4

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import Image
from xyc_interfaces.msg import ArmState, MobileState, Observation

from xyc_observation.observation_builder import (
    CAMERAS, JOINTS, MOBILE_NAMES, MOBILE_UNITS, STATE_NAMES,
    ObservationBuilder, ObservationConfig, ObservationError,
    content_id, time_message,
)
from xyc_observation.observation_node import ObservationNode


def calibration(arm):
    return {
        'schema_version': 1, 'arm': arm, 'port': f'/test/{arm}',
        'ticks_per_revolution': 4096, 'source': 'test_fixture',
        'motors': {
            name: {'id': index + 1, 'model_number': 2569 if index in (1, 2) else 2825,
                   'operating_mode': 0, 'hardware_homing_offset': 0,
                   'reference_tick': 2000, 'min_delta_ticks': -1000, 'max_delta_ticks': 1000,
                   'direction': 1, 'normalized_min': 0 if name.endswith('_gripper') else -100,
                   'normalized_max': 100}
            for index, name in enumerate(JOINTS[:7] if arm == 'left' else JOINTS[7:])
        },
    }


def inputs(now_ns, config, docs):
    images = []
    for index, frame_id in enumerate(config.camera_frame_ids):
        image = Image()
        image.header.stamp = time_message(now_ns - 5_000_000 - index * 1_000_000)
        image.header.frame_id = frame_id
        image.width, image.height, image.encoding = config.width, config.height, config.encoding
        image.step = config.width * 3
        image.data = bytes([index + 1]) * (image.step * image.height)
        images.append(image)
    arms = []
    for index, arm in enumerate(('left', 'right')):
        state = ArmState()
        state.arm = arm
        state.header.stamp = time_message(now_ns - 3_000_000)
        state.calibration_id = content_id(docs[arm])
        state.names = list(JOINTS[index * 7:(index + 1) * 7])
        state.position_normalized = [float(index * 10 + i) for i in range(6)] + [50. + index]
        state.valid = [True] * 7
        state.errors = [''] * 7
        state.sample_stamps = [time_message(now_ns - 4_000_000 - i * 100_000) for i in range(7)]
        state.complete = True
        arms.append(state)
    mobile = MobileState()
    mobile.header.stamp = time_message(now_ns - 2_000_000)
    mobile.header.frame_id = config.mobile_frame_id
    mobile.names = list(MOBILE_NAMES)
    mobile.state = [.12, .10, -.05, .20]
    mobile.units = list(MOBILE_UNITS)
    mobile.valid = True
    return images, arms, mobile


class TestObservationBuilder(unittest.TestCase):
    def setUp(self):
        self.config = replace(ObservationConfig(), width=2, height=2)
        self.docs = {arm: calibration(arm) for arm in ('left', 'right')}
        self.builder = ObservationBuilder(self.config, self.docs)
        self.now = 100 * 10**9
        self.images, self.arms, self.mobile = inputs(self.now, self.config, self.docs)

    def build(self):
        return self.builder.build(
            self.images, self.arms, self.mobile, now_ns=self.now, sequence=7)

    def test_fixed_order_preserves_pixels_metadata_and_wire_roundtrip(self):
        expected = list(self.arms[0].position_normalized) + list(self.arms[1].position_normalized)
        for arm in self.arms:
            for name in ('names', 'position_normalized', 'valid', 'errors', 'sample_stamps'):
                setattr(arm, name, list(reversed(getattr(arm, name))))
        frame = self.build()
        self.assertEqual(frame.state_names, list(STATE_NAMES))
        self.assertEqual(list(frame.state), expected + list(self.mobile.state))
        self.assertEqual(frame.camera_names, list(CAMERAS))
        self.assertEqual(frame.header.stamp, self.images[0].header.stamp)
        self.assertEqual(frame.sequence, 7)
        self.assertTrue(frame.valid)
        wire = deserialize_message(serialize_message(frame), Observation)
        for index, image in enumerate(wire.images):
            self.assertEqual(bytes(image.data), bytes(self.images[index].data))
            self.assertEqual(image.header, self.images[index].header)
        self.assertEqual(len(wire.state), 18)
        self.assertEqual(wire.calibration_ids, self.builder.calibration_ids)
        self.assertEqual(wire.state_units, ['normalized'] * 14 + list(MOBILE_UNITS))

    def test_missing_camera_never_becomes_a_placeholder_image(self):
        self.images[2] = None
        with self.assertRaisesRegex(ObservationError, 'no message'):
            self.build()
        with self.assertRaises(ObservationError):
            self.builder.build(
                self.images[:3], self.arms, self.mobile, now_ns=self.now, sequence=1)

    def test_wrong_camera_identity_dimensions_encoding_or_length_is_rejected(self):
        for field, value in (('width', 3), ('encoding', 'rgb8'), ('step', 1), ('data', b'bad')):
            images, _, _ = inputs(self.now, self.config, self.docs)
            self.images = images
            setattr(self.images[0], field, value)
            with self.assertRaises(ObservationError):
                self.build()
        self.images, _, _ = inputs(self.now, self.config, self.docs)
        self.images[0].header.frame_id = self.images[1].header.frame_id
        with self.assertRaisesRegex(ObservationError, 'frame_id'):
            self.build()

    def test_calibration_version_and_arm_identity_are_checked(self):
        self.arms[0].calibration_id = 'different_calibration'
        with self.assertRaisesRegex(ObservationError, 'calibration_id'):
            self.build()
        self.arms[0].calibration_id = self.builder.calibration_ids[0]
        self.arms[0].arm = 'right'
        with self.assertRaisesRegex(ObservationError, 'arm identity'):
            self.build()

    def test_missing_duplicate_or_inconsistent_joint_arrays_are_rejected(self):
        for mutation in ('missing', 'duplicate', 'short_flags'):
            _, self.arms, _ = inputs(self.now, self.config, self.docs)
            if mutation == 'missing':
                self.arms[0].names.pop()
            elif mutation == 'duplicate':
                self.arms[0].names[0] = self.arms[0].names[1]
            else:
                self.arms[0].valid.pop()
            with self.assertRaises(ObservationError):
                self.build()

    def test_invalid_nan_and_outside_range_states_are_rejected(self):
        for value in (float('nan'), float('inf'), 101.):
            _, self.arms, _ = inputs(self.now, self.config, self.docs)
            self.arms[0].position_normalized[0] = value
            with self.assertRaises(ObservationError):
                self.build()
        _, self.arms, _ = inputs(self.now, self.config, self.docs)
        self.arms[0].valid[0] = False
        self.arms[0].complete = False
        with self.assertRaises(ObservationError):
            self.build()

    def test_mobile_state_is_measured_reorderable_and_not_tied_to_torque(self):
        self.mobile.enabled = False
        order = [3, 0, 2, 1]
        self.mobile.names = [self.mobile.names[index] for index in order]
        self.mobile.state = [self.mobile.state[index] for index in order]
        self.mobile.units = [self.mobile.units[index] for index in order]
        frame = self.build()
        self.assertEqual(list(frame.state[-4:]), [.12, .10, -.05, .20])
        self.assertEqual(frame.state_units[-4:], list(MOBILE_UNITS))
        self.assertEqual(frame.mobile_batch_stamp, self.mobile.header.stamp)

    def test_invalid_mobile_state_is_rejected_without_relaxing_feedback_checks(self):
        for mutation in ('invalid', 'homing', 'bad_unit', 'nan', 'height', 'name'):
            _, _, self.mobile = inputs(self.now, self.config, self.docs)
            if mutation == 'invalid':
                self.mobile.valid = False
                self.mobile.error = 'feedback missing'
            elif mutation == 'homing':
                self.mobile.homing = True
            elif mutation == 'bad_unit':
                self.mobile.units[1] = 'ticks/s'
            elif mutation == 'nan':
                self.mobile.state[2] = float('nan')
            elif mutation == 'height':
                self.mobile.state[0] = .61
            else:
                self.mobile.names[0] = 'unknown'
            with self.assertRaises(ObservationError, msg=mutation):
                self.build()

    def test_stale_future_and_zero_timestamps_are_rejected(self):
        for delta in (-10**9, 10**9):
            self.images, _, _ = inputs(self.now, self.config, self.docs)
            self.images[0].header.stamp = time_message(self.now + delta)
            with self.assertRaises(ObservationError):
                self.build()
        self.images[0].header.stamp = time_message(0)
        with self.assertRaises(ObservationError):
            self.build()

    def test_time_skew_includes_individual_motor_samples(self):
        self.arms[0].sample_stamps[0] = time_message(self.now - 60_000_000)
        with self.assertRaisesRegex(ObservationError, 'time skew'):
            self.build()
        self.arms[0].sample_stamps[0] = time_message(self.now - 52_000_000)
        frame = self.build()  # Newest mobile stamp is now-2ms; skew exactly 50ms.
        self.assertAlmostEqual(frame.max_time_skew_s, .05, places=6)

    def test_schema_identity_tracks_order_units_rate_and_calibration(self):
        config = replace(self.config, publish_hz=5.)
        changed_rate = ObservationBuilder(config, self.docs)
        self.assertNotEqual(changed_rate.schema_id, self.builder.schema_id)
        docs = deepcopy(self.docs)
        docs['left']['source'] = 'another_asset'
        changed_asset = ObservationBuilder(self.config, docs)
        self.assertNotEqual(changed_asset.schema_id, self.builder.schema_id)
        self.assertNotEqual(changed_asset.calibration_ids, self.builder.calibration_ids)
        reordered = {arm: dict(reversed(list(document.items()))) for arm, document in self.docs.items()}
        self.assertEqual(ObservationBuilder(self.config, reordered).schema_id, self.builder.schema_id)

    def test_bad_configuration_or_calibration_cannot_start_builder(self):
        with self.assertRaises(ValueError):
            ObservationBuilder(replace(self.config, camera_names=tuple(reversed(CAMERAS))), self.docs)
        with self.assertRaises(ValueError):
            ObservationBuilder(replace(self.config, sync_queue_size=100), self.docs)
        docs = deepcopy(self.docs)
        docs['left']['arm'] = 'right'
        with self.assertRaises(ValueError):
            ObservationBuilder(self.config, docs)


class TestObservationROS(unittest.TestCase):
    def setUp(self):
        self.context = Context()
        rclpy.init(context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.tag = 'xyc_obs_test_' + uuid4().hex[:8]
        self.namespace = '/' + self.tag
        self.config = replace(
            ObservationConfig(), width=2, height=2,
            camera_topics=tuple(f'{self.namespace}/camera/{name}' for name in CAMERAS),
        )
        self.docs = {arm: calibration(arm) for arm in ('left', 'right')}
        self.directory = tempfile.TemporaryDirectory()
        for arm, document in self.docs.items():
            (Path(self.directory.name) / f'{arm}.json').write_text(json.dumps(document))
        state_topics = [f'{self.namespace}/state/{arm}' for arm in ('left', 'right')]
        mobile_state_topic = f'{self.namespace}/state/mobile'
        overrides = [
            Parameter('camera_topics', value=list(self.config.camera_topics)),
            Parameter('state_topics', value=state_topics),
            Parameter('mobile_state_topic', value=mobile_state_topic),
            Parameter('width', value=2), Parameter('height', value=2),
            Parameter('calibration_dir', value=self.directory.name),
        ]
        self.observer = ObservationNode(context=self.context, namespace=self.namespace,
                                        parameter_overrides=overrides)
        self.input_node = Node(self.tag + '_input', context=self.context)
        self.executor.add_node(self.observer)
        self.executor.add_node(self.input_node)
        self.publishers = [
            self.input_node.create_publisher(Image, topic, qos_profile_sensor_data)
            for topic in self.config.camera_topics
        ] + [self.input_node.create_publisher(ArmState, topic, qos_profile_sensor_data)
             for topic in state_topics] + [
                 self.input_node.create_publisher(
                     MobileState, mobile_state_topic, qos_profile_sensor_data)]
        self.skip_camera = False
        self.bad_calibration = False
        self.frames = []
        self.status = []
        self.subscriptions = [
            self.input_node.create_subscription(
                Observation, f'{self.namespace}/observation', self.frames.append, qos_profile_sensor_data),
            self.input_node.create_subscription(
                DiagnosticArray, f'{self.namespace}/observation_status', self.status.append, 10),
        ]
        self.timer = self.input_node.create_timer(1 / 30, self.publish)

    def publish(self):
        now = self.input_node.get_clock().now().nanoseconds
        images, arms, mobile = inputs(now, self.config, self.docs)
        if self.bad_calibration:
            arms[0].calibration_id = 'wrong_version'
        for index, (publisher, message) in enumerate(zip(self.publishers, images + arms + [mobile])):
            if self.skip_camera and index == 3:
                continue
            publisher.publish(message)

    def pump(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=.02)
        self.assertTrue(predicate(), 'Observation condition did not become true')

    def spin_for(self, duration):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=.02)

    def tearDown(self):
        self.observer.destroy_node()
        self.input_node.destroy_node()
        self.executor.shutdown(timeout_sec=1.)
        self.context.try_shutdown()
        self.directory.cleanup()

    def test_synchronization_missing_camera_version_failure_and_recovery(self):
        self.pump(lambda: len(self.frames) >= 3)
        self.assertTrue(all(frame.valid and len(frame.images) == 4 and len(frame.state) == 18
                            for frame in self.frames))
        self.assertTrue(all(frame.max_time_skew_s <= .05 for frame in self.frames))
        self.assertEqual(self.frames[-1].calibration_ids,
                         [content_id(self.docs['left']), content_id(self.docs['right'])])
        self.skip_camera = True
        self.spin_for(.6)
        count = len(self.frames)
        self.spin_for(.3)
        self.assertEqual(len(self.frames), count)
        self.pump(lambda: any('camera/wrist_right' in status.name and status.level != DiagnosticStatus.OK
                             for array in self.status for status in array.status))
        self.skip_camera = False
        self.pump(lambda: len(self.frames) >= count + 2)
        self.bad_calibration = True
        self.spin_for(.4)
        count = len(self.frames)
        self.spin_for(.3)
        self.assertEqual(len(self.frames), count)
        self.assertGreater(self.observer.rejected, 0)
        self.assertTrue(any('calibration_id' in error for error in self.observer.last_errors))
        self.bad_calibration = False
        self.pump(lambda: len(self.frames) >= count + 2)
        self.assertEqual(self.frames[-1].schema_id, self.observer.builder.schema_id)
        sequences = [frame.sequence for frame in self.frames]
        self.assertEqual(sequences, sorted(set(sequences)))


if __name__ == '__main__':
    unittest.main()
