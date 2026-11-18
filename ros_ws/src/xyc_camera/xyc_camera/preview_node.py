"""Read-only four-camera browser preview, accessed through an SSH forward."""

from collections import deque
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import socket
import struct
import threading
import time
from urllib.parse import urlsplit

from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from xyc.robot.profile import robot_profile


CAMERAS = {'forward': '前方', 'chest': '胸前',
           'wrist_left': '左腕', 'wrist_right': '右腕'}
PREVIEW_STREAM_FPS = robot_profile().preview_fps


class FrameHub:
    """Bounded latest-frame storage shared by ROS and HTTP threads."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.condition = threading.Condition()
        self.frames = {}
        self.arrivals = {name: deque(maxlen=90) for name in CAMERAS}
        self.closed = False

    def push(self, name, data, stamp_ns=None):
        if (name not in CAMERAS or not data.startswith(b'\xff\xd8')
                or stamp_ns is not None and (type(stamp_ns) is not int or stamp_ns <= 0)):
            raise ValueError('Unsupported camera or JPEG data')
        with self.condition:
            now = self.clock()
            previous = self.frames.get(name)
            sequence = 1 if previous is None else previous[0] + 1
            self.frames[name] = (sequence, data, now, stamp_ns)
            self.arrivals[name].append(now)
            self.condition.notify_all()

    def wait(self, name, previous, timeout=1.0):
        frame = self.wait_full(name, previous, timeout)
        return None if frame is None else frame[:2]

    def wait_full(self, name, previous, timeout=1.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.closed or (name in self.frames and self.frames[name][0] != previous),
                timeout=timeout,
            )
            frame = self.frames.get(name)
            if self.closed or frame is None or frame[0] == previous:
                return None
            return frame

    def latest(self, name):
        with self.condition:
            return self.frames.get(name)

    def status(self):
        with self.condition:
            now = self.clock()
            result = {}
            for name, label in CAMERAS.items():
                frame = self.frames.get(name)
                age = None if frame is None else max(0.0, now - frame[2])
                arrivals = [t for t in self.arrivals[name] if now - t <= 2.0]
                fps = ((len(arrivals) - 1) / (arrivals[-1] - arrivals[0])
                       if len(arrivals) > 1 and arrivals[-1] > arrivals[0] else 0.0)
                result[name] = {'label': label, 'age_s': age, 'fps': round(fps, 1),
                                'live': age is not None and age < 1.0}
            return result

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class PreviewServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, hub, page, recording_address=None):
        self.hub, self.page = hub, page
        self.recording_address = recording_address or (
            '127.0.0.1', robot_profile().ports['recording'])
        super().__init__(address, PreviewHandler)


class PreviewHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def setup(self):
        super().setup()
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def log_message(self, *_):
        pass

    def respond(self, content_type, data, headers=None):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        for name, value in (headers or {}).items():
            self.send_header(name, str(value))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path == '/':
                self.respond('text/html; charset=utf-8', self.server.page)
            elif path == '/api/status':
                data = json.dumps(self.server.hub.status(), ensure_ascii=False).encode('utf-8')
                self.respond('application/json; charset=utf-8', data)
            elif path.startswith('/stream/') and path[len('/stream/'):] in CAMERAS:
                self.stream(path[len('/stream/'):])
            elif path == '/stream/all':
                self.stream_all()
            elif path.startswith('/snapshot/') and path[len('/snapshot/'):] in CAMERAS:
                self.snapshot(path[len('/snapshot/'):])
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    def do_POST(self):
        """Relay recording bytes over a separate HTTP connection."""
        path = urlsplit(self.path).path
        if path not in ('/api/recording/exchange', '/api/recording/heartbeat'):
            self.send_error(404)
            return
        upstream = http.client.HTTPConnection(*self.server.recording_address, timeout=3.)
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 65536:
                self.send_error(413)
                return
            self.connection.settimeout(3.)
            body = self.rfile.read(length)
            if len(body) != length:
                raise ValueError('Incomplete recording request')
            upstream.request('POST', '/' + path.rsplit('/', 1)[-1], body,
                             {'Content-Type': 'application/json'})
            response = upstream.getresponse()
            size = int(response.getheader('Content-Length', '0'))
            if not 0 < size <= 4 * 2 * 1024 * 1024 + 65540:
                raise ValueError('Invalid recording response size')
            data = response.read(size)
            if len(data) != size:
                raise ValueError('Incomplete recording response')
            self.send_response(response.status)
            self.send_header(
                'Content-Type', response.getheader('Content-Type', 'application/octet-stream'))
            self.send_header('Content-Length', str(size))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)
        except (OSError, ValueError, http.client.HTTPException):
            try:
                self.send_error(502, 'Recording source unavailable; use recording:=true')
            except OSError:
                pass
        finally:
            upstream.close()

    def stream(self, name):
        self.close_connection = True  # Multipart body ends only when this socket closes.
        self.connection.settimeout(2.0)
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Connection', 'close')
        self.end_headers()
        previous = 0
        next_frame_at = 0.0
        hub = self.server.hub
        while not hub.closed:
            delay = next_frame_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            frame = hub.wait(name, previous)
            if frame is None:
                # MIME whitespace detects closed clients without presenting
                # an invalid image part to a browser waiting for its first JPEG.
                self.wfile.write(b'\r\n')
            else:
                previous, data = frame
                next_frame_at = time.monotonic() + 1.0 / PREVIEW_STREAM_FPS
                self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                 + str(len(data)).encode() + b'\r\n\r\n' + data + b'\r\n')
            self.wfile.flush()

    def stream_all(self):
        """
        Multiplex four JPEG feeds over one browser connection.

        Each packet is camera index (uint8), JPEG length (uint32, network byte
        order), followed by the JPEG bytes.  A single connection avoids browser
        per-host limits that otherwise leave the third and fourth MJPEG feeds
        permanently queued.
        """
        self.close_connection = True
        self.connection.settimeout(2.0)
        self.send_response(200)
        self.send_header('Content-Type', 'application/x-xyc-camera-stream')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        self.send_header('Connection', 'close')
        self.end_headers()
        hub = self.server.hub
        names = tuple(CAMERAS)
        previous = dict.fromkeys(names, 0)
        while not hub.closed:
            started = time.monotonic()
            wrote = False
            for index, name in enumerate(names):
                frame = hub.latest(name)
                if frame is None or frame[0] == previous[name]:
                    continue
                sequence, data = frame[:2]
                previous[name] = sequence
                self.wfile.write(struct.pack('!BI', index, len(data)) + data)
                wrote = True
            if wrote:
                self.wfile.flush()
            delay = 1.0 / PREVIEW_STREAM_FPS - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)

    def snapshot(self, name):
        """Return one new stamped JPEG for model inference, never a cached old frame."""
        hub = self.server.hub
        existing = hub.latest(name)
        previous = existing[0] if existing is not None else 0
        frame = hub.wait_full(name, previous, timeout=1.0)
        if frame is None:
            self.send_error(504, 'No fresh camera frame')
            return
        sequence, data, _, stamp_ns = frame
        if stamp_ns is None:
            self.send_error(503, 'Camera frame has no source timestamp')
            return
        self.respond('image/jpeg', data, {
            'X-XYC-Camera': name,
            'X-XYC-Sequence': sequence,
            'X-XYC-Source-Time-Ns': stamp_ns,
        })


class CameraPreview(Node):
    def __init__(self, **kwargs):
        super().__init__('camera_preview', **kwargs)
        fixed = ParameterDescriptor(read_only=True)
        port = self.declare_parameter('port', robot_profile().ports['data'], fixed).value
        self.hub = FrameHub()
        # The preview is a four-camera service, so its ROS inputs are fixed for
        # the lifetime of the node.  Creating and destroying subscriptions from
        # HTTP viewer state introduced a discovery race where one stream could
        # remain detached even though its camera was publishing normally.
        self.inputs = {
            name: self.create_subscription(
                CompressedImage, f'/camera/{name}/image_raw/compressed',
                lambda msg, name=name: self.receive(name, msg), qos_profile_sensor_data,
            )
            for name in CAMERAS
        }
        page = Path(get_package_share_directory('xyc_camera')) / 'web' / 'cameras.html'
        self.server = PreviewServer(('127.0.0.1', port), self.hub, page.read_bytes())
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.get_logger().info(f'四路只读预览就绪：http://127.0.0.1:{port}；通过 SSH 转发访问')

    def receive(self, name, message):
        if 'jpeg' in message.format.lower() or message.format.lower() == 'jpg':
            try:
                stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
                self.hub.push(name, bytes(message.data), stamp_ns)
            except ValueError as exc:
                self.get_logger().warning(str(exc), throttle_duration_sec=2.0)

    def destroy_node(self):
        self.hub.close()
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2.0)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = CameraPreview()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
