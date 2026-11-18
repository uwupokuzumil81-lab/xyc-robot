"""Publish the same aligned samples used by recording and live inference."""

from xyc.robot.profile import robot_profile
# SPDX-License-Identifier: Apache-2.0

from dataclasses import fields
import json
from pathlib import Path

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from xyc.observation import MAX_CAMERA_AGE_NS, ObservationBuffer
from xyc_interfaces.msg import Observation
from .observation_builder import ObservationBuilder, ObservationConfig


class ObservationNode(Node):
    def __init__(self, **options):
        super().__init__('observation_node', **options)
        defaults = ObservationConfig()
        values = {}
        for field in fields(defaults):
            value = getattr(defaults, field.name)
            parameter = self.declare_parameter(
                field.name, list(value) if isinstance(value, tuple) else value,
                ParameterDescriptor(read_only=True)).value
            values[field.name] = tuple(parameter) if isinstance(value, tuple) else parameter
        config = ObservationConfig(**values)
        directory = Path(self.declare_parameter(
            'calibration_dir', str(robot_profile().robot_root / 'config/calibration'),
            ParameterDescriptor(read_only=True)).value).expanduser()
        documents = {side: json.loads((directory / f'{side}.json').read_text()) for side in ('left', 'right')}
        self.builder = ObservationBuilder(config, documents)
        self.buffer = ObservationBuffer()
        self.buffer.reset(0, int(config.publish_hz), 'mobile')
        self.topics = [*config.camera_topics, *config.state_topics]
        self.labels = [*[f'camera/{name}' for name in config.camera_names], 'arm/left', 'arm/right']
        self.latest = [None] * 6
        self.received = [0] * 6
        self.sequence = self.published = self.rejected = 0
        self.last_error = ''
        self.publisher = self.create_publisher(
            Observation, 'observation', QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.status_publisher = self.create_publisher(DiagnosticArray, 'observation_status', 1)
        self.inputs = [self.create_subscription(
            CompressedImage, topic, lambda message, index=index: self.camera(message, index),
            qos_profile_sensor_data) for index, topic in enumerate(config.camera_topics)]
        self.inputs += [self.create_subscription(
            String, topic, lambda message, index=index: self.arm(message, index),
            qos_profile_sensor_data) for index, topic in enumerate(config.state_topics, 4)]
        self.publish_timer = self.create_timer(1. / config.publish_hz, self.publish_observation)
        self.status_timer = self.create_timer(1., self.publish_status)
        self.get_logger().info('Shared ObservationBuffer ready: four JPEGs + 18-D controlled state')

    def camera(self, message, index):
        stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
        try:
            self.buffer.add_camera(self.builder.config.camera_names[index], stamp, bytes(message.data))
            self.latest[index] = stamp, ''
            self.received[index] += 1
        except ValueError as exc:
            self.last_error = str(exc)

    def arm(self, message, index):
        try:
            sample = json.loads(message.data)
            self.buffer.add_arm('left' if index == 4 else 'right', sample)
            self.latest[index] = sample['stamp_ns'], '' if sample.get('valid') else sample.get('reason', 'Invalid state')
            self.received[index] += 1
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self.last_error = str(exc)

    def publish_observation(self):
        try:
            now = self.get_clock().now().nanoseconds
            aligned = self.buffer.pick(now)
            if aligned is None:
                self.last_error = self.buffer.reason
                return
            self.publisher.publish(self.builder.build(aligned, now_ns=now, sequence=self.sequence))
            self.sequence = (self.sequence + 1) % 2**32
            self.published += 1
            self.last_error = ''
        except ValueError as exc:
            self.rejected += 1
            self.last_error = str(exc)

    def publish_status(self):
        now = self.get_clock().now().nanoseconds
        output = DiagnosticArray()
        output.header.stamp = self.get_clock().now().to_msg()
        for index, (topic, label) in enumerate(zip(self.topics, self.labels)):
            latest = self.latest[index]
            reason = 'Waiting for source' if latest is None else latest[1]
            if latest is not None and not 0 <= now - latest[0] <= MAX_CAMERA_AGE_NS:
                reason = 'Source is stale or from the future'
            status = DiagnosticStatus(
                name='xyc_observation/' + label, hardware_id=topic,
                level=DiagnosticStatus.WARN if reason else DiagnosticStatus.OK,
                message=reason or 'OK', values=[KeyValue(key='received', value=str(self.received[index]))])
            output.status.append(status)
        output.status.insert(0, DiagnosticStatus(
            name='xyc_observation/bundle', hardware_id=self.builder.schema_id,
            level=DiagnosticStatus.WARN if self.last_error or not self.published else DiagnosticStatus.OK,
            message=self.last_error or ('OK' if self.published else 'Waiting for shared observation'),
            values=[KeyValue(key='published', value=str(self.published)),
                    KeyValue(key='rejected', value=str(self.rejected))]))
        self.status_publisher.publish(output)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ObservationNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
