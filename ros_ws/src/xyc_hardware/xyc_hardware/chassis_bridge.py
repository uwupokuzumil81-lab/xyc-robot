# SPDX-License-Identifier: Apache-2.0
"""Relay PC chassis intent into ROS; no separate keyboard controller."""

import json
import threading
import time

from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from xyc_interfaces.msg import MobileCommand

from xyc_hardware.chassis_control import (
    CHASSIS_NOT_READY, COMMAND_QOS, TELEOP_FAULT_QOS, TELEOP_FAULT_TOPIC, mobile_message,
)
from xyc.contracts import COMMAND_VALIDITY_S
from xyc.transport.teleop import MAX_SAMPLE_AGE, checked_chassis


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
        self.teleop_fault = ''
        self.stop_needed = False
        self._zero_sent = False
        super().__init__('pc_chassis_bridge', use_global_arguments=False)
        self.mobile_pub = self.create_publisher(MobileCommand, '/robot/mobile/command', COMMAND_QOS)
        self.status_sub = self.create_subscription(String, '/robot/chassis/status', self.on_status, 1)
        self.fault_sub = self.create_subscription(String, TELEOP_FAULT_TOPIC, self.on_teleop_fault, TELEOP_FAULT_QOS)
        self.control_clients = {name: self.create_client(Trigger, '/robot/chassis/' + name)
                                for name in ('enable', 'stop', 'home_lift')}
        self.pending = None
        self.state = {}
        self.received = None
        self.commands = [0.] * 4
        self.command_until = 0.
        self.timer = self.create_timer(1 / 30, self.publish_commands)

    def on_status(self, message):
        try:
            state = json.loads(message.data)
            if not isinstance(state, dict):
                return
            self.state, self.received = state, time.monotonic()
        except (TypeError, ValueError):
            return

    def on_teleop_fault(self, message):
        with self.packet_lock:
            if self.teleop_fault or not message.data:
                return
            self.teleop_fault = message.data
            self.packet = None
            self.ack = None

    def publish_velocity(self):
        stamp = self.get_clock().now().to_msg()
        now = time.monotonic()
        # A delayed key stream never keeps a previous velocity alive.
        commands = self.commands if now < self.command_until else [0.] * 4
        if (self.received is None or now - self.received > .5
                or not self.state.get('enabled')):
            commands = [0.] * 4
        generated = stamp.sec * 10**9 + stamp.nanosec
        lifetime = min(COMMAND_VALIDITY_S, self.command_until - now) if any(commands) else COMMAND_VALIDITY_S
        self.mobile_pub.publish(mobile_message(
            self, commands, stamp=stamp, expires_ns=generated + int(lifetime * 1e9)))

    def begin_session(self, session):
        with self.packet_lock:
            self.session, self.packet = session, None
            self.last_operation_id = 0
            self.last_received_operation = 0
            self.operation_deadline = 0.
            self.ack = None

    def receive_packet(self, session, payload):
        commands, operation = checked_chassis(payload)
        with self.packet_lock:
            if session != self.session:
                raise ValueError('底盘控制连接已变化')
            if self.teleop_fault:
                if operation is not None:
                    self.ack = {'id': operation['id'], 'accepted': False, 'message': self.teleop_fault[:80]}
                return
            if operation is not None:
                if operation['id'] < self.last_received_operation:
                    raise ValueError('底盘操作序号倒退')
                if operation['id'] > self.last_received_operation:
                    self.operation_deadline = time.monotonic() + 3.
                self.last_received_operation = operation['id']
            self.packet = (time.monotonic(), commands, operation)

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
            if self.teleop_fault:
                status['fault'] = self.teleop_fault
            for field in ('fault', 'reason'):
                if field in status:
                    status[field] = str(status[field])[:80]
            ack = dict(self.ack) if self.ack is not None else None
            return {'chassis_status': status, 'chassis_ack': ack}

    def publish_commands(self):
        now = time.monotonic()
        with self.packet_lock:
            if self.teleop_fault:
                return  # The motor owner executes the same stop operation used by replay and policy.
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
            active = packet is not None and 0 <= now - packet[0] <= MAX_SAMPLE_AGE
            if packet is not None and not active:
                self.packet = None
                self.stop_needed = True
            self.command_until = min(now + .05, packet[0] + MAX_SAMPLE_AGE) if active else 0.
            requested = packet[2] if active else None
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
            stopping = self.stop_needed or (self.operation is not None and self.operation[2] == 'stop')
            stopping |= requested is not None and requested['name'] == 'stop'
            self.commands = packet[1] if active and not stopping else [0.] * 4
            # No heartbeat after disconnection: an in-flight homing request
            # also loses its lease if its stop response is unavailable.
            if active:
                self.publish_velocity()
            elif self.stop_needed or self.operation is not None:
                self.command_until = 0.
                # The stop service cancels homing. Publish zero once while
                # waiting; repeated zero heartbeats would keep homing alive.
                if not self._zero_sent:
                    self.publish_velocity()
                    self._zero_sent = True
            if active:
                self._zero_sent = False
