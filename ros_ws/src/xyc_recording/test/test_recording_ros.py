# SPDX-License-Identifier: Apache-2.0
"""Exercise PC bridge -> ROS recording service -> v3 files with fake inputs."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

if os.environ.get('XYC_ISOLATED_TEST') != '1':
    raise unittest.SkipTest('ROS integration test requires XYC_ISOLATED_TEST=1 and an isolated domain')

import cv2
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from xyc_hardware.recording_bridge import RecordingBridge
from xyc_recording.alignment import ARM_NAMES, CAMERAS
from xyc_recording.recorder_node import RecorderNode
from xyc_recording.local_recorder import LocalDatasetReceiver


class TestRecordingSessions(unittest.TestCase):
    def setUp(self):
        rclpy.init()
        self.directory = tempfile.TemporaryDirectory()
        self.recorder = RecorderNode(parameter_overrides=[
            Parameter('stream_port', value=0),
            Parameter('cache_dir', value=self.directory.name)])
        self.bridge = RecordingBridge()

    def tearDown(self):
        self.recorder.destroy_node()
        self.bridge.destroy_node()
        self.directory.cleanup()
        rclpy.try_shutdown()

    def test_new_client_cannot_display_a_previous_clients_222_frames(self):
        self.bridge.begin_session('new-client')
        previous = {'phase': 'idle', 'frames': 222, 'saved_episodes': 1,
                    'storage': 'pc', 'buffering': 'robot_episode_cache',
                    'recording_session': 'old-client', 'dataset': 'xyc/old'}
        self.bridge.on_status(String(data=json.dumps(previous)))
        status = self.bridge.snapshot()['recording_status']
        self.assertEqual(status['frames'], 0)
        self.assertNotIn('dataset', status)
        current = {**previous, 'frames': 7, 'recording_session': 'new-client', 'dataset': 'xyc/new'}
        self.bridge.on_status(String(data=json.dumps(current)))
        self.assertEqual(self.bridge.snapshot()['recording_status']['frames'], 7)

    def test_start_failure_resets_old_progress_and_reports_the_requested_dataset(self):
        self.recorder.state.update(frames=222, missing_input='old homing',
                                   dataset='xyc/old', recording_session='old-client')
        config = {'dataset': 'xyc/new', 'task': 'pick', 'fps': 30, 'episode_frames': 900}
        with patch.object(self.recorder, 'assets', return_value={}), \
                patch('xyc_recording.recorder_node.CachedDesktopWriter', side_effect=OSError('disk full')), \
                self.assertRaisesRegex(OSError, 'disk full'):
            self.recorder.command('new-client:1', 'start', json.dumps(config))
        self.assertEqual(self.recorder.state['frames'], 0)
        self.assertEqual(self.recorder.state['dataset'], 'xyc/new')
        self.assertEqual(self.recorder.state['missing_input'], '')

    def test_disconnect_releases_a_saved_writer_but_old_disconnect_cannot_close_new_one(self):
        writer = MagicMock(stage=None)
        self.recorder.writer = writer
        self.recorder.state['recording_session'] = 'current-client'
        self.recorder.command('old-client:0', 'disconnect', '{}')
        writer.close.assert_not_called()
        self.recorder.command('current-client:0', 'disconnect', '{}')
        writer.close.assert_called_once()
        self.assertIsNone(self.recorder.writer)


class TestROSRecording(unittest.TestCase):
    def test_remote_start_waits_for_homing_saves_then_discards_next_episode(self):
        project = Path(__file__).resolve().parents[4]
        ids = []
        for side in ('left', 'right'):
            document = json.loads((project / f'config/calibration/{side}.json').read_text())
            ids.append(hashlib.sha256(json.dumps(document, sort_keys=True, separators=(',', ':'),
                                                ensure_ascii=False, allow_nan=False).encode()).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            rclpy.init()
            recorder = RecorderNode(parameter_overrides=[
                Parameter('project_dir', value=str(project)), Parameter('stream_port', value=0),
                Parameter('cache_dir', value=str(Path(directory) / 'robot-cache'))])
            bridge = RecordingBridge()
            source = Node('synthetic_recording_sources')
            executor = SingleThreadedExecutor()
            for node in (recorder, bridge, source):
                executor.add_node(node)
            camera_pubs = [source.create_publisher(CompressedImage,
                f'/camera/{name}/image_raw/compressed', qos_profile_sensor_data) for name in CAMERAS]
            arm_pubs = [source.create_publisher(String, f'/motors/{side}/recording_sample',
                                               qos_profile_sensor_data) for side in ('left', 'right')]
            command_pubs = [source.create_publisher(String, topic, qos_profile_sensor_data) for topic in
                            ('/robot/arms/recording_command', '/robot/chassis/recording_command')]
            ok, data = cv2.imencode('.jpg', np.full((480, 640, 3), (30, 50, 180), dtype=np.uint8))
            self.assertTrue(ok)
            homed = False
            commands_enabled = True
            ticks = 0
            session = 'synthetic-pc-session'
            bridge.begin_session(session)

            def publish():
                nonlocal ticks
                ticks += 1
                bridge.receive(session, None)
                stamp = source.get_clock().now()
                for index, side in enumerate(('left', 'right')):
                    sample = {'stamp_ns': stamp.nanoseconds - 1_000_000, 'names': list(ARM_NAMES[index * 7:index * 7 + 7]),
                              'valid': True, 'state': [1.] * 7, 'target': [2.] * 7,
                              'state_stamps_ns': [stamp.nanoseconds - 1_000_000] * 7,
                              'calibration_id': ids[index], 'session': side + '-synthetic',
                              'mobile': {'valid': homed, 'state': [.1, .02, .03, .04],
                                         'action': [.015, .15, 0., .60]}}
                    arm_pubs[index].publish(String(data=json.dumps(sample)))
                if ticks % 2 == 0:
                    image = CompressedImage()
                    image.header.stamp = stamp.to_msg()
                    image.format, image.data = 'jpeg', data.tobytes()
                    for publisher in camera_pubs:
                        publisher.publish(image)
                if not commands_enabled:
                    return
                generated = source.get_clock().now().nanoseconds
                event = {'kind': 'arms', 'sequence': ticks, 'generated_ns': generated,
                         'sent_ns': generated, 'accepted_ns': generated, 'targets': [2.] * 14,
                         'calibration_ids': ids, 'sessions': ['left-synthetic', 'right-synthetic']}
                command_pubs[0].publish(String(data=json.dumps(event)))
                if ticks % 2 == 0:
                    event.update(kind='mobile', targets=[.015, .15, 0., .60], session='left-synthetic')
                    command_pubs[1].publish(String(data=json.dumps(event)))

            timer = source.create_timer(1 / 60, publish)
            def wait(predicate, timeout=12.):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    executor.spin_once(timeout_sec=.01)
                    if predicate():
                        return
                self.fail('recording state timeout: ' + repr(bridge.status))

            config = {'dataset': 'xyc/ros_test', 'task': 'synthetic recording test', 'fps': 30,
                      'episode_frames': 30, 'scope': 'mobile', 'resume': False}
            local = LocalDatasetReceiver(directory, config,
                url=f'http://127.0.0.1:{recorder.stream_server.server_address[1]}').start()
            try:
                wait(lambda: local.ready.is_set())
                wait(lambda: bridge.client.service_is_ready())
                bridge.receive(session, {'id': 1, 'operation': 'start', 'config': config})
                wait(lambda: bridge.status.get('phase') == 'waiting')
                for _ in range(30):
                    executor.spin_once(timeout_sec=.01)
                self.assertEqual(bridge.status.get('frames'), 0)
                homed = True
                wait(lambda: bridge.status.get('saved_episodes') == 1 and bridge.status.get('phase') == 'idle')
                info_path = Path(directory) / 'xyc/ros_test/meta/info.json'
                self.assertEqual(json.loads(info_path.read_text())['total_frames'], 30)
                import pyarrow.parquet as pq
                rows = pq.read_table(info_path.parents[1] / 'data/chunk-000/file-000.parquet').to_pylist()
                for row in rows:
                    self.assertGreaterEqual(min(row['action.sample_times']),
                                            max(*row['observation.camera_times'], *row['observation.state_times']))
                    self.assertEqual(row['action'][:14], [2.] * 14)
                    self.assertEqual(row['observation.state'][:14], [1.] * 14)
                self.assertEqual(len({row['action.arm_command_id'] for row in rows}), 30)
                bridge.receive(session, {'id': 2, 'operation': 'start', 'config': config})
                wait(lambda: bridge.status.get('phase') == 'recording' and bridge.status.get('frames', 0) >= 3)
                commands_enabled = False
                until = time.monotonic() + .15
                while time.monotonic() < until:
                    executor.spin_once(timeout_sec=.01)
                self.assertEqual(bridge.status.get('saved_episodes'), 1)
                self.assertEqual(bridge.status.get('phase'), 'recording')
                commands_enabled = True
                bridge.receive(session, {'id': 3, 'operation': 'discard', 'config': {}})
                wait(lambda: bridge.status.get('phase') == 'idle' and bridge.status.get('frames') == 0)
                self.assertEqual(json.loads(info_path.read_text())['total_episodes'], 1)
                bridge.receive(session, {'id': 4, 'operation': 'start', 'config': config})
                wait(lambda: bridge.status.get('phase') == 'recording' and bridge.status.get('frames', 0) >= 3)
                commands_enabled = False
                wait(lambda: bridge.status.get('phase') == 'idle' and bridge.status.get('saved_episodes') == 2)
                quality = json.loads((info_path.parent / 'quality/episode-000001.json').read_text())
                self.assertIn('动作', quality['interruption']['reason'])
                self.assertGreaterEqual(quality['interruption']['gap_s'], .5)
                self.assertTrue(2 <= bridge.status['frames'] < 30)
                self.assertTrue(bridge.status.get('missing_input'))
                self.assertGreater(ticks, 60)
            finally:
                source.destroy_timer(timer)
                executor.shutdown()
                for node in (bridge, recorder, source):
                    node.destroy_node()
                local.close()
                rclpy.try_shutdown()
