# SPDX-License-Identifier: Apache-2.0
"""Real ROS/TCP replay wiring with fake buses, isolated from every physical driver."""

import json
import os
import socket
import threading
import time
import unittest

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger
from xyc_interfaces.msg import ArmState

from xyc_hardware.arm_group_coordinator import ArmGroupCoordinator
from xyc_hardware.arm_calibration import denormalize_tick, normalize_tick
from xyc_hardware.chassis_control import COMMAND_QOS
from xyc_hardware.motion_replay import MotionReplay
from xyc_hardware.replay_transport import ACTION_NAMES, ACTION_UNITS, STATE_NAMES, STATE_UNITS, PROTOCOL, receive, send
from test_arm_group_coordinator import FakeBus


class ReplayFakeBus(FakeBus):
    def __init__(self, side):
        super().__init__(side)
        self.control_publisher = self.create_publisher(String, f'/motors/{side}/recording_sample', qos_profile_sensor_data)

    def publish_state(self):
        msg = ArmState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.arm = self.side
        msg.calibration_id = 'calibration' + self.side
        msg.names = list(self.arm_transaction.group.names)
        msg.position_normalized = [normalize_tick(control.entry, self.motors[name].position)
                                   for name, control in self.arm_transaction.group.controls.items()]
        msg.valid = [True] * 7
        msg.complete = True
        self.pub.publish(msg)
        valid = all(c.owned and not c.fault for c in self.arm_transaction.group.controls.values())
        self.control_publisher.publish(String(data=json.dumps({
            'calibration_id': 'calibration' + self.side, 'valid': valid, 'reason': '' if valid else 'not held',
            'stamp_ns': self.get_clock().now().nanoseconds,
            'driver_targets': [normalize_tick(c.entry, c.target_ticks) for c in self.arm_transaction.group.controls.values()]
                              if valid else [],
            'driver_target_ticks': [c.target_ticks for c in self.arm_transaction.group.controls.values()]
                                   if valid else []})))


class FakeChassis(Node):
    def __init__(self):
        super().__init__('fake_replay_chassis')
        self.enabled = False
        self.fault = ''
        self.stops = 0
        self.requests = {}
        self.accepted = []
        self.lock = threading.Lock()
        self.status = self.create_publisher(String, '/robot/chassis/status', 1)
        self.commands = self.create_publisher(String, '/robot/chassis/recording_command', COMMAND_QOS)
        self.subs = [self.create_subscription(TwistStamped, '/robot/base/cmd_vel',
                                             lambda msg: self.on_velocity('base', msg), COMMAND_QOS),
                     self.create_subscription(TwistStamped, '/robot/lift/cmd_vel',
                                             lambda msg: self.on_velocity('lift', msg), COMMAND_QOS)]
        self.chassis_services = [self.create_service(Trigger, '/robot/chassis/' + name, callback)
                         for name, callback in [('enable', self.enable), ('stop', self.stop), ('home_lift', self.home)]]
        self.timer = self.create_timer(.03, self.publish_status)

    def publish_status(self):
        self.status.publish(String(data=json.dumps({'enabled': self.enabled, 'fault': self.fault,
                            'feedback_paused': False, 'height_m': 0., 'base_measured': [0., 0., 0.], 'homing': ''})))

    def enable(self, _, response):
        self.enabled = response.success = True
        return response

    def stop(self, _, response):
        self.stops += 1
        response.success = True
        return response

    def home(self, _, response):
        response.success = True
        return response

    def on_velocity(self, kind, message):
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        with self.lock:
            self.requests[kind] = stamp, message
            if len(self.requests) != 2 or self.requests['base'][0] != self.requests['lift'][0]:
                return
            base, lift = self.requests['base'][1], self.requests['lift'][1]
            values = [lift.twist.linear.z, base.twist.linear.x, base.twist.linear.y, base.twist.angular.z]
            self.accepted.append(values)
            self.commands.publish(String(data=json.dumps({'sequence': stamp, 'targets': values})))


@unittest.skipUnless(os.environ.get('XYC_ISOLATED_TEST') == '1' and os.environ.get('ROS_DOMAIN_ID') == '183',
                     'requires isolated ROS domain 183; never run against robot nodes')
class TestReplayROS(unittest.TestCase):
    def test_source_owned_stream_acceptance_and_disconnect_stop(self):
        rclpy.init()
        nodes, thread, executor, connection = [], None, None, None
        replay = None
        try:
            left, right = ReplayFakeBus('left'), ReplayFakeBus('right')
            elbow_name = 'arm_left_elbow_flex'
            elbow = left.arm_transaction.group.controls[elbow_name]
            left.motors[elbow_name].position = denormalize_tick(elbow.entry, 97.80018)
            chassis = FakeChassis()
            coordinator = ArmGroupCoordinator(parameter_overrides=[
                Parameter('command_source', value='replay'), Parameter('streaming_targets', value=True)])
            replay = MotionReplay(parameter_overrides=[Parameter('port', value=0)])
            nodes = [left, right, chassis, coordinator, replay]
            executor = MultiThreadedExecutor(num_threads=8)
            for node in nodes: executor.add_node(node)
            thread = threading.Thread(target=executor.spin, daemon=True)
            thread.start()
            deadline = time.monotonic() + 5.
            while not replay.snapshot()['ready'] and time.monotonic() < deadline: time.sleep(.02)
            self.assertTrue(replay.snapshot()['ready'])
            connection = socket.create_connection(('127.0.0.1', replay.port), timeout=3.)
            connection.settimeout(4.)
            channel = connection.makefile('rwb')
            sequence = 0
            def call(operation, **values):
                nonlocal sequence
                sequence += 1
                send(channel, {'sequence': sequence, 'operation': operation, **values})
                response = receive(channel)
                self.assertTrue(response['accepted'], response.get('message'))
                return response['snapshot']
            state = call('status')
            call('prepare', protocol=PROTOCOL, calibration_ids=state['calibration_ids'],
                 action_names=list(ACTION_NAMES), action_units=list(ACTION_UNITS),
                 state_names=list(STATE_NAMES), state_units=list(STATE_UNITS))
            call('start')
            deadline = time.monotonic() + 2.
            while not replay.snapshot()['chassis'].get('enabled') and time.monotonic() < deadline:
                call('heartbeat')
                time.sleep(.02)
            action = [0.] * 6 + [50.] + [0.] * 6 + [50.] + [0., 1., 0., 0.]
            action[2] = 99.35838
            action[5] = 80.
            action[6] = 100.
            call('target', action=action)
            self.assertIn('stream_targets', left.operations)
            self.assertIn('stream_targets', right.operations)
            self.assertIn([0., .15, 0., 0.], chassis.accepted)
            self.assertIn(('goal', denormalize_tick(elbow.entry, 99.35838)), left.motors[elbow_name].writes)
            for name, target in (('arm_left_wrist_roll', 80.), ('arm_left_gripper', 100.)):
                control = left.arm_transaction.group.controls[name]
                self.assertIn(('goal', denormalize_tick(control.entry, target)), left.motors[name].writes)
            deadline = time.monotonic() + 2.
            while replay.snapshot()['driver_targets'][5:7] != [80., 100.] and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertEqual(replay.snapshot()['driver_targets'][5:7], [80., 100.])
            # A chassis protection fault is reported with its actual snapshot;
            # it must not be misreported as a calibration change or prevent
            # freezing independently healthy arm feedback on disconnect.
            chassis.fault = 'ID 11 持续过流：5.23 A'
            deadline = time.monotonic() + 2.
            while replay.snapshot()['ready'] and time.monotonic() < deadline: time.sleep(.01)
            sequence += 1
            send(channel, {'sequence': sequence, 'operation': 'heartbeat'})
            rejected = receive(channel)
            self.assertFalse(rejected['accepted'])
            self.assertTrue(rejected['stop_confirmed'], rejected.get('stop_error'))
            self.assertIn('ID 11', rejected['message'])
            self.assertNotIn('calibration changed', rejected['message'])
            self.assertEqual(rejected['snapshot']['chassis']['fault'], chassis.fault)
            self.assertTrue(rejected['snapshot']['arms_ready'])
            channel.close()
            connection.close()
            connection = None
            deadline = time.monotonic() + 3.
            while chassis.stops < 1 and time.monotonic() < deadline: time.sleep(.02)
            self.assertGreaterEqual(chassis.stops, 1)
            self.assertEqual(replay.last_mobile, [0., 0., 0., 0.])
        finally:
            if connection is not None: connection.close()
            if replay is not None: replay.close()
            if executor is not None: executor.shutdown()
            if thread is not None: thread.join(3.)
            for node in reversed(nodes): node.destroy_node()
            rclpy.shutdown()
