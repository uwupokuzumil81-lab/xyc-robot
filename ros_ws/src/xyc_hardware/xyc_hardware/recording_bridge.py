# SPDX-License-Identifier: Apache-2.0
"""Recording RPCs share the PC link but never delay its motor-control replies."""

from collections import deque
import json
import threading
import time

from rclpy.node import Node
from std_msgs.msg import String
from xyc_interfaces.srv import RecordingControl


class RecordingBridge(Node):
    def __init__(self):
        super().__init__('recording_bridge', use_global_arguments=False)
        self.lock = threading.Lock()
        self.session = None
        self.received = 0.
        self.last_id = 0
        self.packet = None
        self.disconnects = deque(maxlen=4)
        self.pending = None
        self.ack = None
        self.status = {}
        self.status_at = 0.
        self.client = self.create_client(RecordingControl, '/robot/recording/control')
        self.subscription = self.create_subscription(String, '/robot/recording/status', self.on_status, 1)
        self.lease = self.create_publisher(String, '/robot/recording/operator_lease', 1)
        self.timer = self.create_timer(.02, self.poll)

    def on_status(self, msg):
        try:
            status = json.loads(msg.data)
            if isinstance(status, dict):
                with self.lock:
                    self.status, self.status_at = status, time.monotonic()
        except ValueError:
            pass

    def begin_session(self, session):
        with self.lock:
            self.session, self.received, self.last_id = session, 0., 0
            self.packet, self.ack = None, None
            self.status, self.status_at = {}, 0.

    def receive(self, session, packet):
        if packet is not None:
            if (not isinstance(packet, dict) or set(packet) - {'id', 'operation', 'config'}
                    or type(packet.get('id')) is not int or not 0 < packet['id'] < 2**53
                    or packet.get('operation') not in ('start', 'save', 'discard')
                    or not isinstance(packet.get('config', {}), dict)
                    or len(json.dumps(packet).encode()) > 4096):
                return  # Bad recording input cannot tear down a working arm link.
        with self.lock:
            if session != self.session:
                return
            self.received = time.monotonic()
            if packet is not None and packet['id'] > self.last_id and self.packet is None:
                self.packet = (session, dict(packet))
                self.last_id = packet['id']

    def disconnect(self, session):
        with self.lock:
            if self.session == session:
                self.session, self.received = None, 0.
                self.disconnects.append(session)

    def snapshot(self):
        with self.lock:
            status = dict(self.status) if time.monotonic() - self.status_at < 2. else {}
            if status and status.get('recording_session') != self.session:
                status = {'phase': 'idle', 'frames': 0, 'saved_episodes': 0,
                          'storage': status.get('storage'), 'buffering': status.get('buffering'),
                          'data_channel_ready': status.get('data_channel_ready', False),
                          'message': '采集待机；准备好后按 R 开录'}
            return {'recording_supported': True, 'recording_ack': self.ack,
                    'recording_status': status}

    def poll(self):
        with self.lock:
            if self.session is not None and time.monotonic() - self.received <= .25:
                self.lease.publish(String(data=self.session))
            if self.pending is not None:
                session, packet, future, started = self.pending
                if future.done():
                    try:
                        result = future.result()
                        accepted, message = result.accepted, result.message
                    except Exception as exc:
                        accepted, message = False, str(exc)
                    if session == self.session:
                        self.ack = {'id': packet['id'], 'accepted': accepted, 'message': message}
                    self.pending = None
                elif time.monotonic() - started > 2.:
                    self.client.remove_pending_request(future)
                    if session == self.session:
                        self.ack = {'id': packet['id'], 'accepted': False,
                                    'message': '采集回执超时，请查看采集状态；遥操继续'}
                    self.pending = None
                return
            if not self.client.service_is_ready():
                return
            if self.disconnects:
                session = self.disconnects.popleft()
                packet = {'id': 0, 'operation': 'disconnect'}
            elif self.packet is not None:
                session, packet = self.packet
                self.packet = None
            else:
                return
            request = RecordingControl.Request()
            request.request_id = f'{session}:{packet["id"]}'
            request.operation = packet['operation']
            request.config_json = json.dumps(packet.get('config', {}), ensure_ascii=False)
            self.pending = (session, packet, self.client.call_async(request), time.monotonic())
