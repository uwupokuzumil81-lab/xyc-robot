# SPDX-License-Identifier: Apache-2.0
"""Shared observation pairing and its ROS serialization; no devices are opened."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import unittest
from uuid import uuid4

import cv2
import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message, serialize_message
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header, String
from xyc.contracts import ARM_NAMES, CAMERAS, STATE_NAMES, STATE_UNITS
from xyc.observation import ObservationBuffer
from xyc_interfaces.msg import Observation
from xyc_observation.observation_builder import ObservationBuilder, ObservationConfig, content_id, time_message
from xyc_observation.observation_node import ObservationNode


def calibration(side):
    names = ARM_NAMES[:7] if side == 'left' else ARM_NAMES[7:]
    return {'schema_version': 1, 'arm': side, 'motors': {
        name: {'id': index + 1} for index, name in enumerate(names)}}


def sources(now, docs):
    pixels = np.full((16, 16, 3), (10, 50, 200), np.uint8)
    ok, encoded = cv2.imencode('.jpg', pixels)
    assert ok
    images = [CompressedImage(
        header=Header(
            stamp=time_message(now - 10_000_000 + index * 1_000_000)),
        format='jpeg', data=encoded.tobytes()) for index in range(4)]
    samples = []
    for index, side in enumerate(('left', 'right')):
        stamp = now - 5_000_000
        samples.append({'stamp_ns': stamp, 'names': list(ARM_NAMES[index * 7:(index + 1) * 7]),
                        'state': [float(index)] * 6 + [50.], 'valid': True,
                        'state_stamps_ns': [stamp - 1_000_000] * 7,
                        'calibration_id': content_id(docs[side]), 'session': 'session-' + side,
                        'mobile': {'valid': True, 'state': [.1, .15, 0., 0.]}})
    return images, samples


def align(images, samples, now):
    buffer = ObservationBuffer()
    for name, image in zip(CAMERAS, images):
        buffer.add_camera(name, image.header.stamp.sec * 10**9 + image.header.stamp.nanosec, bytes(image.data))
    for side, sample in zip(('left', 'right'), samples):
        buffer.add_arm(side, sample)
    return buffer, buffer.pick(now)


class TestObservationBuilder(unittest.TestCase):
    def setUp(self):
        self.config = ObservationConfig(width=16, height=16)
        self.docs = {side: calibration(side) for side in ('left', 'right')}
        self.builder = ObservationBuilder(self.config, self.docs)
        self.now = 10**12
        self.images, self.samples = sources(self.now, self.docs)

    def test_same_core_sample_is_preserved_by_ros_roundtrip(self):
        buffer, aligned = align(self.images, self.samples, self.now)
        frame = self.builder.build(aligned, now_ns=self.now, sequence=7)
        wire = deserialize_message(serialize_message(frame), Observation)
        self.assertEqual(list(wire.state), list(aligned.state))
        self.assertEqual(wire.state_names, list(STATE_NAMES))
        self.assertEqual(wire.state_units, list(STATE_UNITS))
        self.assertEqual(wire.camera_names, list(CAMERAS))
        self.assertEqual(wire.header.stamp, time_message(aligned.stamp_ns))
        self.assertEqual(wire.calibration_ids, list(aligned.calibration_ids))
        self.assertEqual(wire.sequence, 7)
        self.assertEqual(wire.schema_version, 3)
        self.assertEqual(len(wire.state_stamps), 18)
        self.assertEqual(len(wire.arm_batch_stamps), 2)
        self.assertEqual(wire.mobile_batch_stamp, time_message(self.samples[0]['stamp_ns']))
        self.assertEqual(len(wire.images), 4)
        self.assertIsNone(buffer.pick(self.now))  # ROS cannot duplicate the same camera frames.

    def test_missing_or_invalid_controlled_sources_use_common_rejection(self):
        for failure in ('missing', 'invalid_state', 'mobile_missing', 'old_state'):
            images, samples = sources(self.now, self.docs)
            if failure == 'missing': images = images[:3]
            if failure == 'invalid_state': samples[1]['valid'] = False
            if failure == 'mobile_missing': samples[0]['mobile']['valid'] = False
            if failure == 'old_state': samples[0]['stamp_ns'] -= 200_000_000
            buffer, aligned = align(images, samples, self.now)
            self.assertIsNone(aligned, failure)
            self.assertTrue(buffer.reason)

    def test_rgb_pixels_are_decoded_from_the_shared_jpeg(self):
        _, aligned = align(self.images, self.samples, self.now)
        renderer = ObservationBuilder(replace(self.config, encoding='rgb8'), self.docs)
        frame = renderer.build(aligned, now_ns=self.now, sequence=0)
        expected = cv2.imdecode(np.frombuffer(aligned.images[0].jpeg, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]
        np.testing.assert_array_equal(np.frombuffer(frame.images[0].data, np.uint8).reshape(16,16,3), expected)

    def test_invalid_jpeg_dimensions_calibration_and_age_are_rejected(self):
        _, aligned = align(self.images, self.samples, self.now)
        for changed in (replace(aligned, calibration_ids=('wrong', 'wrong')),
                        replace(aligned, stamp_ns=self.now - 200_000_000)):
            with self.assertRaises(ValueError):
                self.builder.build(changed, now_ns=self.now, sequence=0)
        wrong = ObservationBuilder(replace(self.config, width=32), self.docs)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            wrong.build(aligned, now_ns=self.now, sequence=0)

    def test_configuration_matches_canonical_sources_and_supported_rate(self):
        for config in (replace(self.config, camera_names=tuple(reversed(CAMERAS))),
                       replace(self.config, publish_hz=60.), replace(self.config, width=0),
                       replace(self.config, state_topics=('/same', '/same'))):
            with self.assertRaises(ValueError): config.validate()


class TestObservationNode(unittest.TestCase):
    def test_ros_consumes_the_same_compressed_images_and_control_samples(self):
        context = Context()
        rclpy.init(context=context)
        executor = SingleThreadedExecutor(context=context)
        name = '/observation_audit_' + uuid4().hex[:8]
        docs = {side: calibration(side) for side in ('left', 'right')}
        with tempfile.TemporaryDirectory() as directory:
            for side, document in docs.items():
                (Path(directory) / (side + '.json')).write_text(json.dumps(document))
            topics = [name + '/camera/' + camera for camera in CAMERAS]
            state_topics = [name + '/state/' + side for side in ('left', 'right')]
            node = ObservationNode(context=context, namespace=name, parameter_overrides=[
                Parameter('calibration_dir', value=directory), Parameter('width', value=16),
                Parameter('height', value=16), Parameter('camera_topics', value=topics),
                Parameter('state_topics', value=state_topics)])
            source = Node('source', namespace=name, context=context)
            executor.add_node(node)
            executor.add_node(source)
            camera_pubs = [source.create_publisher(CompressedImage, topic, qos_profile_sensor_data) for topic in topics]
            arm_pubs = [source.create_publisher(String, topic, qos_profile_sensor_data) for topic in state_topics]
            received, diagnostics = [], []
            source.create_subscription(Observation, name + '/observation', received.append, qos_profile_sensor_data)
            source.create_subscription(DiagnosticArray, name + '/observation_status', diagnostics.append, 1)
            try:
                until = time.monotonic() + 4.
                while time.monotonic() < until and (not received or not diagnostics):
                    images, samples = sources(source.get_clock().now().nanoseconds, docs)
                    for publisher, image in zip(camera_pubs, images): publisher.publish(image)
                    for publisher, sample in zip(arm_pubs, samples): publisher.publish(String(data=json.dumps(sample)))
                    executor.spin_once(timeout_sec=.01)
                self.assertTrue(received)
                self.assertTrue(diagnostics)
                self.assertEqual(list(received[-1].state), [0.]*6+[50.]+[1.]*6+[50.,.1,.15,0.,0.])
                self.assertEqual(len(diagnostics[-1].status), 7)
            finally:
                executor.shutdown()
                source.destroy_node()
                node.destroy_node()
                context.shutdown()
