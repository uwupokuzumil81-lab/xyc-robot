# SPDX-License-Identifier: Apache-2.0
"""Subscribe, approximately synchronize, validate and publish live observations."""

from dataclasses import fields
import json
from pathlib import Path
import time

import message_filters
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from xyc_interfaces.msg import ArmState, MobileState, Observation

from xyc_observation.observation_builder import (
    ObservationBuilder, ObservationConfig, ObservationError, stamp_ns,
)


class ObservationNode(Node):
    def __init__(self, **node_options):
        super().__init__('observation_node', **node_options)
        try:
            def declare(name, value):
                self.declare_parameter(name, value, ParameterDescriptor(read_only=True))
            defaults = ObservationConfig()
            for field in fields(defaults):
                value = getattr(defaults, field.name)
                declare(field.name, list(value) if isinstance(value, tuple) else value)
            declare('state_topics', ['/motors/left/state', '/motors/right/state'])
            declare('calibration_dir', str(Path.home() / 'pi05_from_zero/config/calibration'))
            values = {}
            for field in fields(defaults):
                value = self.get_parameter(field.name).value
                values[field.name] = tuple(value) if isinstance(getattr(defaults, field.name), tuple) else value
            config = ObservationConfig(**values)
            calibration_dir = Path(self.get_parameter('calibration_dir').value).expanduser()
            calibrations = {
                arm: json.loads((calibration_dir / f'{arm}.json').read_text())
                for arm in ('left', 'right')
            }
            self.builder = ObservationBuilder(config, calibrations)
            state_topics = list(self.get_parameter('state_topics').value)
            if len(state_topics) != 2 or len(set(state_topics)) != 2 or not all(state_topics):
                raise ValueError('Provide separate left/right state topics')
            self.topics = list(config.camera_topics) + state_topics + [config.mobile_state_topic]
            self.labels = ([f'camera/{name}' for name in config.camera_names]
                           + ['arm/left', 'arm/right', 'mobile'])
            self.latest = [None] * 7
            self.received = [0] * 7
            self.pending = None
            self.sequence = 0
            self.matched = 0
            self.published = 0
            self.rejected = 0
            self.last_errors = []
            self.last_anchor_ns = None
            self.last_valid_ns = None
            self.next_warning = 0.0
            self.publisher = self.create_publisher(
                Observation, 'observation', QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            )
            self.status_publisher = self.create_publisher(DiagnosticArray, 'observation_status', 10)
            qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.filters = []
            for index, topic in enumerate(self.topics):
                message_type = Image if index < 4 else ArmState if index < 6 else MobileState
                subscriber = message_filters.Subscriber(self, message_type, topic, qos_profile=qos)
                subscriber.registerCallback(self.on_input, index)
                self.filters.append(subscriber)
            self.synchronizer = message_filters.ApproximateTimeSynchronizer(
                self.filters, config.sync_queue_size, config.sync_slop_s,
            )
            self.synchronizer.registerCallback(self.on_match)
            self.publish_timer = self.create_timer(1.0 / config.publish_hz, self.publish_observation)
            self.status_timer = self.create_timer(1.0, self.publish_status)
            self.get_logger().info(
                f'Read-only observation: 4 cameras + 18 robot states; target {config.publish_hz:g} Hz; '
                f'publishing {self.publisher.topic_name}; schema_id={self.builder.schema_id}'
            )
        except BaseException:
            self.destroy_node()
            raise

    def on_input(self, message, index):
        self.latest[index] = message
        self.received[index] += 1

    def on_match(self, *messages):
        self.matched += 1
        # Keep only one latest synchronized bundle. Never accumulate image batches.
        self.pending = messages

    def publish_observation(self):
        messages, self.pending = self.pending, None
        if messages is None:
            return
        try:
            now_ns = self.get_clock().now().nanoseconds
            observation = self.builder.build(
                messages[:4], messages[4:6], messages[6], now_ns=now_ns, sequence=self.sequence,
            )
            anchor = stamp_ns(observation.header.stamp)
            if self.last_anchor_ns is not None and anchor <= self.last_anchor_ns:
                raise ObservationError(['reference timestamp repeated or moved backwards'])
        except ObservationError as exc:
            self.rejected += 1
            self.last_errors = exc.errors
            return
        self.publisher.publish(observation)
        self.sequence = (self.sequence + 1) % (2**32)
        self.published += 1
        self.last_anchor_ns = anchor
        self.last_valid_ns = now_ns
        self.last_errors = []

    def publish_status(self):
        now_ns = self.get_clock().now().nanoseconds
        output = DiagnosticArray()
        output.header.stamp = self.get_clock().now().to_msg()
        problems = []
        for index, (topic, label, message) in enumerate(zip(self.topics, self.labels, self.latest)):
            errors = self.builder.inspect_source(index, message, now_ns)
            status = DiagnosticStatus()
            status.name = f'xyc_observation/{label}'
            status.hardware_id = topic
            status.level = DiagnosticStatus.WARN if errors else DiagnosticStatus.OK
            status.message = '; '.join(errors) if errors else 'OK'
            status.values = [
                KeyValue(key='topic', value=topic),
                KeyValue(key='received', value=str(self.received[index])),
            ]
            output.status.append(status)
            if errors:
                problems.append(f'{label}: {status.message}')
        if self.last_valid_ns is None:
            problems.append('No validated synchronized observation has been published')
        elif (now_ns - self.last_valid_ns) / 1e9 > self.builder.config.max_age_s:
            problems.append('No recent validated observation; do not reuse an old frame')
        if self.last_errors:
            problems.extend(self.last_errors)
        summary = DiagnosticStatus()
        summary.name = 'xyc_observation/bundle'
        summary.hardware_id = self.builder.schema_id
        summary.level = DiagnosticStatus.WARN if problems else DiagnosticStatus.OK
        summary.message = '; '.join(problems) if problems else 'OK'
        summary.values = [
            KeyValue(key='matched', value=str(self.matched)),
            KeyValue(key='published', value=str(self.published)),
            KeyValue(key='rejected', value=str(self.rejected)),
            KeyValue(key='schema_id', value=self.builder.schema_id),
            KeyValue(key='last_rejection', value='; '.join(self.last_errors)),
        ]
        output.status.insert(0, summary)
        self.status_publisher.publish(output)
        if problems and time.monotonic() >= self.next_warning:
            self.get_logger().warning('Observation unavailable: ' + '; '.join(problems[:3]))
            self.next_warning = time.monotonic() + 5.0


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ObservationNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
        finally:
            rclpy.try_shutdown()


if __name__ == '__main__':
    main()
