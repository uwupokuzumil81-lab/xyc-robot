# SPDX-License-Identifier: Apache-2.0
"""Relay PC chassis intent into ROS; no separate keyboard controller."""

import json
import threading
import time

from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from xyc_hardware.chassis_control import CHASSIS_NOT_READY, COMMAND_QOS
from xyc_hardware.teleop_transport import checked_chassis


class PCChassisBridge(Node):
    """Use the existing left-bus owner for base and lift commands."""

    def __init__(self):
        self.packet_lock = threading.Lock()
        self.session = None
        self.packet = None
        self.operation = None
        self.operation_since = 0.
        self.last_operation_id = 0
        self.last_received_operation = 0
        self.operation_deadline = 0.
        self.ack = None
        self.stop_needed = False
        self._zero_sent = False
        super().__init__('pc_chassis_bridge', use_global_arguments=False)
        self.base_pub = self.create_publisher(TwistStamped, '/robot/base/cmd_vel', COMMAND_QOS)
        self.lift_pub = self.create_publisher(TwistStamped, '/robot/lift/cmd_vel', COMMAND_QOS)
        self.status_sub = self.create_subscription(String, '/robot/chassis/status', self.on_status, 1)
        self.control_clients = {name: self.create_client(Trigger, '/robot/chassis/' + name)
                                for name in ('enable', 'stop', 'home_lift')}
        self.pending = None
        self.state = {}
        self.received = None
        self.base, self.lift = (0., 0., 0.), 0.
        self.base_until = self.lift_until = 0.
        self.timer = self.create_timer(1 / 30, self.publish_commands)

    def on_status(self, message):
        try:
            state = json.loads(message.data)
            if not isinstance(state, dict):
                return
            self.state, self.received = state, time.monotonic()
        except (TypeError, ValueError):
            return

    def publish_velocity(self):
        now = time.monotonic()
        # A delayed key stream never keeps a previous velocity alive.
        base = self.base if now < self.base_until else (0., 0., 0.)
        lift = self.lift if now < self.lift_until else 0.
        if (self.received is None or now - self.received > .5
                or not self.state.get('enabled')):
            base, lift = (0., 0., 0.), 0.
        stamp = self.get_clock().now().to_msg()
        command = TwistStamped()
        command.header.stamp, command.header.frame_id = stamp, 'base_link'
        command.twist.linear.x, command.twist.linear.y, command.twist.angular.z = base
        self.base_pub.publish(command)
        command = TwistStamped()
        command.header.stamp, command.header.frame_id = stamp, 'base_link'
        command.twist.linear.z = lift
        self.lift_pub.publish(command)

    def begin_session(self, session):
        with self.packet_lock:
            self.session, self.packet = session, None
            self.last_operation_id = 0
            self.last_received_operation = 0
            self.operation_deadline = 0.
            self.ack = None

    def receive_packet(self, session, payload):
        base, lift, operation = checked_chassis(payload)
        with self.packet_lock:
            if session != self.session:
                raise ValueError('底盘控制连接已变化')
            if operation is not None:
                if operation['id'] < self.last_received_operation:
                    raise ValueError('底盘操作序号倒退')
                if operation['id'] > self.last_received_operation:
                    self.operation_deadline = time.monotonic() + 3.
                self.last_received_operation = operation['id']
            self.packet = (time.monotonic(), base, lift, operation)

    def disconnect(self, session):
        with self.packet_lock:
            if session == self.session:
                if self.packet is not None:
                    self.stop_needed = True
                self.packet = None

    def snapshot(self):
        with self.packet_lock:
            fresh = self.received is not None and time.monotonic() - self.received <= .5
            status = dict(self.state) if fresh else {}
            for field in ('fault', 'reason'):
                if field in status:
                    status[field] = str(status[field])[:80]
            ack = dict(self.ack) if self.ack is not None else None
            return {'chassis_status': status, 'chassis_ack': ack}

    def publish_commands(self):
        now = time.monotonic()
        with self.packet_lock:
            if self.operation is not None:
                session, operation_id, name = self.operation
                if self.pending.done():
                    try:
                        answer = self.pending.result()
                        accepted, message = answer.success, answer.message
                    except Exception as exc:
                        accepted, message = False, str(exc)
                    retry = (session == self.session and name == 'enable' and not accepted
                             and message.startswith(CHASSIS_NOT_READY)
                             and now < self.operation_deadline)
                    if retry:
                        self.last_operation_id = operation_id - 1
                        self.ack = {'id': operation_id, 'pending': True,
                                    'message': message[len(CHASSIS_NOT_READY):]}
                    elif session == self.session and operation_id:
                        self.ack = {'id': operation_id, 'accepted': accepted, 'message': message[:80]}
                        if not accepted:
                            self.get_logger().error(f'底盘操作 {name} 失败：{message}')
                    self.pending = self.operation = None
                elif now - self.operation_since > 3.:
                    self.control_clients[name].remove_pending_request(self.pending)
                    if session == self.session and operation_id:
                        self.ack = {'id': operation_id, 'accepted': False, 'message': '操作响应超时，执行状态待核实'}
                    self.pending = self.operation = None
                    self.stop_needed = True
                    self.packet = None
            packet = self.packet
            active = packet is not None and 0 <= now - packet[0] <= .15
            if packet is not None and not active:
                self.packet = None
                self.stop_needed = True
            self.base, self.lift = (packet[1], packet[2]) if active else ((0., 0., 0.), 0.)
            self.base_until = self.lift_until = now + .05 if active else 0.
            requested = packet[3] if active else None
            if self.operation is None:
                if self.stop_needed:
                    operation_id, name, session = 0, 'stop', self.session
                elif requested is not None and requested['id'] > self.last_operation_id:
                    operation_id, name, session = requested['id'], requested['name'], self.session
                else:
                    operation_id, name, session = 0, None, self.session
                if name is not None:
                    client = self.control_clients[name]
                    if client.service_is_ready():
                        self.pending = client.call_async(Trigger.Request())
                        self.operation = (session, operation_id, name)
                        self.operation_since = now
                        if operation_id:
                            self.last_operation_id = operation_id
                        if name == 'stop':
                            self.stop_needed = False
                    elif operation_id:
                        if now < self.operation_deadline:
                            self.ack = {'id': operation_id, 'pending': True, 'message': '正在等待底盘控制服务'}
                        else:
                            self.last_operation_id = operation_id
                            self.ack = {'id': operation_id, 'accepted': False,
                                        'message': '底盘服务未就绪，请用 chassis_mode:=true 启动'}
            # No heartbeat after disconnection: an in-flight homing request
            # also loses its lease if its stop response is unavailable.
            if active:
                self.publish_velocity()
            elif self.stop_needed or self.operation is not None:
                self.base_until = self.lift_until = 0.
                # The stop service cancels homing. Publish zero once while
                # waiting; repeated zero heartbeats would keep homing alive.
                if not self._zero_sent:
                    self.publish_velocity()
                    self._zero_sent = True
            if active:
                self._zero_sent = False
