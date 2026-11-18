"""Bounded, acknowledged recording transport. No filesystem or motor access."""

from xyc.robot.profile import robot_profile
# SPDX-License-Identifier: Apache-2.0

from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import struct
import threading
import time
from uuid import uuid4

from xyc.contracts import CAMERAS
from xyc.observation import CameraFrame
from .alignment import Capture


STREAM_PORT = robot_profile().ports['recording']  # Loopback only; desktop accesses it through the configured data port.
MAX_HEADER = 65536
MAX_PACKET = 4 * 2 * 1024 * 1024 + MAX_HEADER + 4


class RecordingBackpressure(ConnectionError):
    pass


def pack_event(sequence, operation, payload=None, capture=None, *, epoch=None):
    event = {'version': 1, 'sequence': sequence, 'operation': operation, 'payload': payload or {}}
    if epoch is not None:
        event['epoch'] = epoch
    images = ()
    if capture is not None:
        event['capture'] = {key: value for key, value in vars(capture).items() if key != 'images'}
        event['images'] = [{'stamp_ns': frame.stamp_ns, 'length': len(frame.jpeg)} for frame in capture.images]
        images = tuple(frame.jpeg for frame in capture.images)
    header = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()
    if len(header) > MAX_HEADER or any(len(data) > 2 * 1024 * 1024 for data in images):
        raise ValueError('Recording packet exceeds its size limit')
    packet = struct.pack('!I', len(header)) + header + b''.join(images)
    if len(packet) > MAX_PACKET:
        raise ValueError('Recording packet exceeds its size limit')
    return packet


def unpack_event(packet):
    if not 4 <= len(packet) <= MAX_PACKET:
        raise ValueError('Invalid recording packet size')
    size = struct.unpack('!I', packet[:4])[0]
    if not 0 < size <= MAX_HEADER or size + 4 > len(packet):
        raise ValueError('Invalid recording packet header')
    event = json.loads(packet[4:4 + size])
    if (not isinstance(event, dict) or event.get('version') != 1
            or type(event.get('sequence')) is not int or event['sequence'] < 0
            or event.get('operation') not in ('idle', 'open', 'begin', 'capture', 'save', 'discard', 'close', 'batch')
            or not isinstance(event.get('payload'), dict)):
        raise ValueError('Unsupported recording event')
    offset = 4 + size
    if event['operation'] == 'batch':
        lengths = event['payload'].get('lengths')
        if (not isinstance(lengths, list) or not 2 <= len(lengths) <= 4
                or any(type(n) is not int or n < 4 for n in lengths)
                or sum(lengths) != len(packet) - offset):
            raise ValueError('Invalid recording batch lengths')
        events = []
        for length in lengths:
            item = unpack_event(packet[offset:offset + length])
            if item['operation'] != 'capture' or item.get('epoch') != event.get('epoch'):
                raise ValueError('Only captures from one source may be batched')
            events.append(item)
            offset += length
        event['events'] = events
    if event['operation'] == 'capture':
        images = event.get('images')
        if not isinstance(images, list) or len(images) != len(CAMERAS):
            raise ValueError('Recording needs four JPEG images')
        frames = []
        for image in images:
            length = image.get('length')
            if (type(length) is not int or not 4 <= length <= 2 * 1024 * 1024
                    or offset + length > len(packet) or type(image.get('stamp_ns')) is not int):
                raise ValueError('Invalid recording JPEG length or timestamp')
            jpeg = packet[offset:offset + length]
            if not jpeg.startswith(b'\xff\xd8'):
                raise ValueError('Recording image is not JPEG')
            frames.append(CameraFrame(image['stamp_ns'], jpeg))
            offset += length
        values = dict(event['capture'])
        for key in ('state', 'action', 'state_stamps_ns', 'action_stamps_ns', 'calibration_ids',
                    'sessions', 'sent_times_ns', 'accepted_times_ns'):
            values[key] = tuple(values[key])
        event['capture'] = Capture(images=tuple(frames), **values)
    if offset != len(packet):
        raise ValueError('Recording packet contains trailing bytes')
    return event


@dataclass
class Pending:
    sequence: int
    operation: str
    packet: bytes
    done: threading.Event
    result: object = None
    error: str = ''


class CaptureRelay:
    """Keep at most 32 samples/32MiB; retain each event until desktop ACK."""

    def __init__(self, *, max_events=32, max_bytes=32 * 1024 * 1024):
        self.condition = threading.Condition()
        self.queue = deque()
        self.max_events, self.max_bytes = max_events, max_bytes
        self.bytes = self.sequence = self.last_ack = 0
        self.client = None
        self.client_at = 0.
        self.closed = False
        self.write_error = ''
        self.epoch = uuid4().hex

    def alive(self):
        return self.client is not None and time.monotonic() - self.client_at < 2.

    def heartbeat(self, client):
        if not isinstance(client, str) or not 1 <= len(client) <= 64:
            raise ValueError('Invalid desktop recording identity')
        with self.condition:
            if self.closed:
                raise ConnectionError('Recording source closed')
            if self.alive() and self.client != client:
                raise ConnectionError('Another desktop already owns the recording connection')
            self.client, self.client_at = client, time.monotonic()
            self.condition.notify_all()
        return {'version': 1, 'storage': 'pc'}

    def exchange(self, client, ack=None):
        self.heartbeat(client)
        with self.condition:
            acknowledgments = ack if isinstance(ack, list) else ([] if ack is None else [ack])
            if len(acknowledgments) > 4:
                raise ValueError('Too many recording acknowledgments')
            for ack in acknowledgments:
                if (not isinstance(ack, dict) or type(ack.get('sequence')) is not int
                        or type(ack.get('accepted')) is not bool or ack.get('epoch', self.epoch) != self.epoch):
                    raise ValueError('Invalid desktop recording acknowledgment')
                sequence = ack['sequence']
                if sequence > self.last_ack:
                    if not self.queue or sequence != self.queue[0].sequence:
                        raise ValueError('Recording acknowledgment is out of order')
                    pending = self.queue.popleft()
                    self.bytes -= len(pending.packet)
                    pending.result = ack.get('result', {})
                    pending.error = '' if ack['accepted'] else str(ack.get('message', 'Desktop write failed'))
                    if pending.error:
                        self.write_error = pending.error
                    elif pending.operation in ('discard', 'open'):
                        self.write_error = ''
                    self.last_ack = sequence
                    pending.done.set()
                    if pending.error:
                        self.cancel(pending.error)
                    self.condition.notify_all()
            self.condition.wait_for(lambda: self.closed or self.queue, timeout=.2)
            if not self.queue:
                return pack_event(0, 'idle', epoch=self.epoch)
            # Piggyback several queued samples on one response, so a delayed
            # network round trip cannot cap capture throughput at 1/RTT.
            packets, size = [], 1024
            for pending in self.queue:
                if pending.operation != 'capture' or len(packets) == 4 or size + len(pending.packet) > MAX_PACKET:
                    break
                packets.append(pending.packet)
                size += len(pending.packet)
            if len(packets) < 2:
                return self.queue[0].packet
            return (pack_event(0, 'batch', {'lengths': [len(p) for p in packets]}, epoch=self.epoch)
                    + b''.join(packets))

    def put(self, operation, payload=None, *, capture=None, wait_for_space=False):
        with self.condition:
            if self.write_error and operation not in ('discard', 'close', 'open'):
                raise OSError('电脑写入失败：' + self.write_error)
            self.sequence += 1
            packet = pack_event(self.sequence, operation, payload, capture, epoch=self.epoch)
            def space():
                return len(self.queue) < self.max_events and self.bytes + len(packet) <= self.max_bytes
            while not space():
                if not wait_for_space or not self.alive() or self.closed:
                    raise RecordingBackpressure('电脑数据接收跟不上，传输缓存已满；保存已接收前段')
                self.condition.wait(.05)
            if self.closed or not self.alive():
                raise ConnectionError(f"电脑采集数据通道未连接，请保留 SSH 的{robot_profile().ports['data']}端口转发")
            pending = Pending(self.sequence, operation, packet, threading.Event())
            self.queue.append(pending)
            self.bytes += len(packet)
            self.condition.notify_all()
            return pending

    def call(self, operation, payload=None, *, timeout=15.):
        pending = self.put(operation, payload, wait_for_space=True)
        deadline = time.monotonic() + timeout
        while not pending.done.wait(.05):
            if self.closed or not self.alive() or time.monotonic() >= deadline:
                self.cancel('电脑采集连接中断或保存回执超时；请检查电脑本地数据')
                raise ConnectionError('电脑采集连接中断或保存回执超时；请检查电脑本地数据')
        if pending.error:
            raise OSError(pending.error)
        return pending.result

    def cancel(self, reason):
        with self.condition:
            for pending in self.queue:
                pending.error = reason
                pending.done.set()
            self.queue.clear()
            self.bytes = 0
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.cancel('Recording source closed')


class RelayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, relay):
        self.relay = relay
        super().__init__(address, RelayHandler)


class RelayHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *_):
        pass

    def do_POST(self):
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_HEADER:
                raise ValueError('Invalid recording request size')
            self.connection.settimeout(3.)
            request = json.loads(self.rfile.read(length))
            if self.path == '/exchange':
                data = self.server.relay.exchange(request.get('client'), request.get('ack'))
                content_type = 'application/x-xyc-capture-v1'
            elif self.path == '/heartbeat':
                data = json.dumps(self.server.relay.heartbeat(request.get('client'))).encode()
                content_type = 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            try:
                self.send_error(409, str(exc).encode('ascii', 'replace').decode())
            except OSError:
                pass
