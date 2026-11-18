"""Preview transport checks use synthetic images, never physical cameras."""
import http.client
import json
import os
import struct
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from xyc_camera.preview_node import CameraPreview, CAMERAS, FrameHub, PreviewServer


JPEG = b'\xff\xd8synthetic-jpeg\xff\xd9'


class TestPreviewCadence(unittest.TestCase):
    def test_compressed_acquisition_keeps_every_frame_independent_of_browser_rate(self):
        from xyc_camera.camera_node import CameraNode
        message = SimpleNamespace(header=SimpleNamespace())
        camera = SimpleNamespace(
            camera=MagicMock(), bridge=MagicMock(), publisher=MagicMock(),
            compressed_publisher=MagicMock(), encode_pending=None,
            encode_condition=threading.Condition(), get_clock=MagicMock(),
            frame_id='fake_optical', encode_stopped=False)
        camera.camera.read.return_value = (True, object())
        camera.bridge.cv2_to_imgmsg.return_value = message
        camera.compressed_publisher.get_subscription_count.return_value = 1
        with patch('xyc_camera.camera_node.rclpy.ok', return_value=True):
            for _ in range(100):
                camera.encode_pending = None
                CameraNode.publish_frame(camera)
                self.assertIsNotNone(camera.encode_pending)
        self.assertEqual(camera.publisher.publish.call_count, 100)

    def test_shutdown_during_blocking_camera_read_is_not_a_capture_failure(self):
        from xyc_camera.camera_node import CameraNode
        camera = SimpleNamespace(
            camera=MagicMock(), publisher=MagicMock(), encode_stopped=False,
        )
        camera.camera.read.return_value = (False, None)
        with patch('xyc_camera.camera_node.rclpy.ok', return_value=False):
            CameraNode.publish_frame(camera)
        camera.publisher.publish.assert_not_called()


class TestFrameHub(unittest.TestCase):
    def test_latest_frame_replaces_old_frame_and_reports_staleness(self):
        now = [1.0]
        hub = FrameHub(lambda: now[0])
        self.assertFalse(hub.status()['forward']['live'])
        hub.push('forward', JPEG)
        now[0] = 1.1
        hub.push('forward', JPEG + b'new')
        self.assertEqual(hub.wait('forward', 0), (2, JPEG + b'new'))
        self.assertAlmostEqual(hub.status()['forward']['fps'], 10.0)
        now[0] = 2.2
        self.assertFalse(hub.status()['forward']['live'])
        self.assertEqual(len(hub.frames), 1)

    def test_close_releases_waiters(self):
        hub = FrameHub()
        hub.close()
        self.assertIsNone(hub.wait('forward', 0, timeout=.01))

    def test_invalid_camera_and_non_jpeg_data_are_rejected(self):
        for name, data in (('unknown', JPEG), ('forward', b'not-a-jpeg')):
            with self.subTest(name=name), self.assertRaises(ValueError):
                FrameHub().push(name, data)


class TestPreviewHTTP(unittest.TestCase):
    def setUp(self):
        self.hub = FrameHub()
        self.server = PreviewServer(('127.0.0.1', 0), self.hub, b'<html>four cameras</html>')
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()

    def tearDown(self):
        self.hub.close()
        self.server.shutdown()
        self.server.server_close()
        self.worker.join(2.)

    def request(self, path):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3.)
        self.addCleanup(connection.close)
        connection.request('GET', path)
        response = connection.getresponse()
        self.addCleanup(response.close)
        return response

    def test_dashboard_and_status_have_expected_content_and_do_not_cache(self):
        response = self.request('/')
        self.assertEqual(response.status, 200)
        self.assertIn(b'four cameras', response.read())
        self.assertEqual(response.getheader('Cache-Control'), 'no-store')
        status = json.loads(self.request('/api/status').read())
        self.assertEqual(set(status), set(CAMERAS))
        self.assertTrue(all(not camera['live'] for camera in status.values()))
        self.assertEqual(self.request('/../../README.md').status, 404)

    def test_stream_has_correct_jpeg_boundaries_and_frame_lengths(self):
        self.hub.push('wrist_left', JPEG)
        response = self.request('/stream/wrist_left')
        self.assertEqual(response.status, 200)
        self.assertIn('multipart/x-mixed-replace', response.getheader('Content-Type'))
        self.assertEqual(response.readline(), b'--frame\r\n')
        self.assertEqual(response.readline(), b'Content-Type: image/jpeg\r\n')
        self.assertEqual(response.readline(), f'Content-Length: {len(JPEG)}\r\n'.encode())
        self.assertEqual(response.readline(), b'\r\n')
        self.assertEqual(response.read(len(JPEG)), JPEG)

    def test_waiting_stream_does_not_send_a_non_image_part_before_the_first_jpeg(self):
        response = self.request('/stream/forward')
        self.assertEqual(response.readline(), b'\r\n')
        self.hub.push('forward', JPEG)
        self.assertEqual(response.readline(), b'--frame\r\n')

    def test_one_binary_stream_multiplexes_all_camera_names(self):
        self.hub.push('wrist_left', JPEG)
        response = self.request('/stream/all')
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader('Content-Type'), 'application/x-xyc-camera-stream')
        header = response.read(5)
        index, length = struct.unpack('!BI', header)
        self.assertEqual(tuple(CAMERAS)[index], 'wrist_left')
        self.assertEqual(length, len(JPEG))
        self.assertEqual(response.read(length), JPEG)

    def test_recording_uses_existing_preview_port_and_preserves_binary_packets(self):
        from xyc.recording.stream import CaptureRelay, RelayServer
        relay = CaptureRelay()
        upstream = RelayServer(('127.0.0.1', 0), relay)
        worker = threading.Thread(target=upstream.serve_forever, daemon=True)
        worker.start()
        self.server.recording_address = upstream.server_address
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3.)
        try:
            body = json.dumps({'client': 'desktop'}).encode()
            connection.request('POST', '/api/recording/heartbeat', body)
            response = connection.getresponse()
            self.assertEqual(json.loads(response.read()), {'version': 1, 'storage': 'pc'})
            pending = relay.put('begin')
            connection.request('POST', '/api/recording/exchange', body)
            response = connection.getresponse()
            self.assertEqual(response.read(), pending.packet)
        finally:
            connection.close()
            relay.close()
            upstream.shutdown()
            upstream.server_close()
            worker.join()


@unittest.skipUnless(os.environ.get('XYC_ISOLATED_TEST') == '1' and
                     os.environ.get('ROS_DOMAIN_ID') == '181', 'requires isolated ROS domain 181')
class TestPreviewROS(unittest.TestCase):
    def setUp(self):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        rclpy.init(args=[])
        self.rclpy = rclpy
        self.executor = SingleThreadedExecutor()
        self.nodes = []
        self.responses = []
        self.connections = []

    def tearDown(self):
        for response in self.responses:
            response.close()
        for connection in self.connections:
            connection.close()
        self.executor.shutdown()
        for node in reversed(self.nodes):
            node.destroy_node()
        self.rclpy.try_shutdown()

    def add(self, node):
        self.nodes.append(node)
        self.executor.add_node(node)
        return node

    def spin_until(self, predicate, timeout=5.):
        until = time.monotonic() + timeout
        while time.monotonic() < until and not predicate():
            self.executor.spin_once(timeout_sec=.02)
        self.assertTrue(predicate())

    def test_preview_keeps_four_ros_inputs_for_its_lifetime(self):
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage
        preview = self.add(CameraPreview(parameter_overrides=[Parameter('port', value=0)]))
        source = self.add(Node('fake_four_cameras'))
        publishers = {name: source.create_publisher(
            CompressedImage, f'/camera/{name}/image_raw/compressed', qos_profile_sensor_data)
            for name in CAMERAS}
        self.assertEqual(set(preview.inputs), set(CAMERAS))
        self.spin_until(lambda: all(pub.get_subscription_count() for pub in publishers.values()))
        for name in CAMERAS:
            connection = http.client.HTTPConnection(*preview.server.server_address, timeout=3.)
            self.connections.append(connection)
            connection.request('GET', '/stream/' + name)
            self.responses.append(connection.getresponse())
        for name, publisher in publishers.items():
            message = CompressedImage(format='jpeg', data=JPEG)
            message.header.stamp.sec = 1
            publisher.publish(message)
        self.spin_until(lambda: all(item['live'] for item in preview.hub.status().values()))
        for response in self.responses:
            response.close()
        for connection in self.connections:
            connection.close()
        self.assertEqual(set(preview.inputs), set(CAMERAS))

    def test_capture_still_publishes_raw_image_alongside_worker_encoded_preview(self):
        import cv2
        import numpy as np
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, Image
        from xyc_camera.camera_node import CameraNode

        class FakeCamera:
            def isOpened(self):
                return True

            def set(self, *_):  # noqa: A003 - OpenCV VideoCapture API
                return True

            def read(self):
                return True, np.full((30, 40, 3), 100, dtype=np.uint8)

            def release(self):
                pass
        parameters = [Parameter('device', value='/fake/camera'),
                      Parameter('frame_id', value='fake_optical'),
                      Parameter('width', value=40), Parameter('height', value=30)]
        with patch('xyc_camera.camera_node.cv2.VideoCapture', return_value=FakeCamera()):
            self.add(CameraNode(namespace='/camera/synthetic', parameter_overrides=parameters))
        sink = self.add(Node('fake_image_observer'))
        raw, compressed = [], []
        sink.create_subscription(
            Image, '/camera/synthetic/image_raw', raw.append, qos_profile_sensor_data)
        sink.create_subscription(CompressedImage, '/camera/synthetic/image_raw/compressed',
                                 compressed.append, qos_profile_sensor_data)
        self.spin_until(lambda: raw and compressed)
        decoded = cv2.imdecode(
            np.frombuffer(bytes(compressed[-1].data), dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (30, 40, 3))
        self.assertEqual(raw[-1].encoding, 'bgr8')
        self.assertEqual(raw[-1].header.frame_id, 'fake_optical')


if __name__ == '__main__':
    unittest.main()
