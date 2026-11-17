# SPDX-License-Identifier: Apache-2.0
"""Synthetic cameras/control samples only. No test opens a motor or camera."""

from dataclasses import replace
import json
import math
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from xyc_recording.alignment import ACTION_SEMANTICS, ARM_NAMES, CAMERAS, CameraFrame, Capture, CaptureBuffer
from xyc_recording.writer import LeRobotV3Writer, acquire_dataset_lock, checked_config


def arm(side, stamp, *, target=12., mobile=True):
    offset = 0 if side == 'left' else 7
    return {'stamp_ns': stamp, 'valid': True, 'names': list(ARM_NAMES[offset:offset + 7]),
            'state': [10.] * 7, 'target': [target] * 7, 'state_stamps_ns': [stamp] * 7,
            'calibration_id': ('a' if side == 'left' else 'b') * 64, 'session': side + '-session',
            'mobile': {'valid': mobile, 'state': [.1, .02, .03, .04], 'action': [.015, .15, 0., .60]}}


def jpeg(value):
    image = np.zeros((16, 16, 3), np.uint8)
    image[:] = (value, 30, 220)
    ok, data = cv2.imencode('.jpg', image)
    assert ok
    return data.tobytes()


def command(kind, stamp, sequence=None, *, target=30., accepted=None):
    event = {'kind': kind, 'sequence': sequence or stamp, 'generated_ns': stamp,
             'sent_ns': stamp + 500_000, 'accepted_ns': accepted or stamp + 1_000_000,
             'targets': [target] * 14 if kind == 'arms' else [.015, .15, 0., .60]}
    if kind == 'arms':
        event.update(calibration_ids=['a' * 64, 'b' * 64], sessions=['left-session', 'right-session'])
    else:
        event['session'] = 'left-session'
    return event


def capture(index, origin=1_000_000_000):
    stamp = origin + round(index * 1e9 / 29.5)
    images = tuple(CameraFrame(stamp + i * 1_000_000, jpeg(index * 10)) for i in range(4))
    return Capture(stamp, images, tuple([float(index)] * 14 + [.1, .02, .03, .04]),
                   tuple([float(index) + .5] * 14 + [1., 1., 0., 1.]),
                   (stamp,) * 18, (stamp + 4_000_000,) * 18,
                   ('a' * 64, 'b' * 64), ('left-session', 'right-session'), .003,
                   (stamp + 5_000_000,) * 18, (stamp + 6_000_000,) * 18, index + 1, index + 1)


class TestAlignment(unittest.TestCase):
    def ready(self, mobile=True):
        buffer = CaptureBuffer()
        for side in ('left', 'right'):
            buffer.add_arm(side, arm(side, 995_000_000, target=11., mobile=mobile))
            buffer.add_arm(side, arm(side, 1_010_000_000, target=99., mobile=mobile))
        for i, name in enumerate(CAMERAS):
            buffer.add_camera(name, 1_000_000_000 + i * 2_000_000, b'jpeg')
        buffer.add_command(command('arms', 990_000_000, 1, target=11., accepted=1_019_000_000))
        buffer.add_command(command('arms', 1_012_000_000, 2))
        buffer.add_command(command('mobile', 1_013_000_000, 2))
        return buffer

    def test_observation_is_paired_with_new_requested_action_not_held_or_next_measured_state(self):
        buffer = self.ready()
        sample = buffer.pick(1_020_000_000)
        self.assertEqual(len(sample.state), 18)
        self.assertEqual(sample.action[:14], (30.,) * 14)
        self.assertEqual(sample.state[:14], (10.,) * 14)
        self.assertEqual(sample.arm_command_id, 2)
        self.assertGreaterEqual(min(sample.action_stamps_ns), max(sample.state_stamps_ns))
        self.assertEqual(sample.state[-4:], (.1, .02, .03, .04))
        self.assertEqual(sample.action[-4:], (1., 1., 0., 1.))
        self.assertIsNone(buffer.pick(1_030_000_000))  # no duplicated camera frame

    def test_unknown_lift_height_is_not_silently_recorded_as_zero(self):
        buffer = self.ready(mobile=False)
        self.assertIsNone(buffer.pick(1_020_000_000))
        self.assertIn('升降回零', buffer.reason)
        buffer.reset(0, 30, 'arms')
        self.assertEqual(len(buffer.pick(1_020_000_000).state), 14)

    def test_mobile_feedback_fault_is_reported_without_filling_a_recording_row(self):
        buffer = self.ready(mobile=False)
        buffer.arms['left'][-1]['mobile']['reason'] = 'ID [8,9,10,11] feedback missing'
        self.assertIsNone(buffer.pick(1_020_000_000))
        self.assertIn('8,9,10,11', buffer.reason)
        self.assertEqual(buffer.command_cursors['arms'], 0)

    def test_stale_and_widely_separated_cameras_are_rejected(self):
        buffer = self.ready()
        self.assertIsNone(buffer.pick(1_500_000_000))
        buffer = self.ready()
        buffer.cameras['wrist_right'].clear()
        buffer.add_camera('wrist_right', 1_070_000_000, b'jpeg')
        self.assertIsNone(buffer.pick(1_080_000_000))

    def test_invalid_control_and_session_change_are_not_recorded(self):
        buffer = self.ready()
        buffer.arms['left'][1]['valid'] = False
        self.assertIsNone(buffer.pick(1_020_000_000))
        buffer = self.ready()
        buffer.pick(1_020_000_000)
        for i, name in enumerate(CAMERAS):
            buffer.add_camera(name, 1_100_000_000 + i * 2_000_000, b'jpeg')
        for side in ('left', 'right'):
            sample = arm(side, 1_099_000_000)
            sample['session'] = 'restarted'
            buffer.add_arm(side, sample)
        buffer.add_command(command('arms', 1_110_000_000, 3))
        buffer.add_command(command('mobile', 1_110_000_000, 3))
        with self.assertRaisesRegex(ValueError, '会话/标定'):
            buffer.pick(1_120_000_000)

    def test_near_thirty_hz_jitter_does_not_halve_recording_rate(self):
        for fps, minimum in ((30, 97), (15, 47)):
            buffer = CaptureBuffer()
            buffer.reset(0, fps, 'mobile')
            count = 0
            for i in range(100):
                stamp = 1_000_000_000 + round(i * 1e9 / 29.5)
                for side in ('left', 'right'):
                    buffer.add_arm(side, arm(side, stamp - 1_000_000))
                for j, name in enumerate(CAMERAS):
                    buffer.add_camera(name, stamp + j * 500_000, b'jpeg')
                buffer.add_command(command('arms', stamp + 2_000_000, i + 1))
                buffer.add_command(command('mobile', stamp + 2_000_000, i + 1))
                count += buffer.pick(stamp + 4_000_000) is not None
            self.assertGreaterEqual(count, minimum)
            self.assertLessEqual(count, 100 if fps == 30 else 54)

    def test_independent_camera_phases_and_delivery_jitter_preserve_new_groups(self):
        events = []
        for i, name in enumerate(CAMERAS):
            for j in range(180):
                stamp = 1_000_000_000 + round((j / (29.4 + i * .1)
                    + i * .007 + .0015 * math.sin(j * 1.7 + i)) * 1e9)
                events.append((stamp + (2 + i) * 1_000_000, 'camera', name, stamp))
        for i in range(400):
            stamp = 990_000_000 + round(i * 1e9 / 60)
            for side in ('left', 'right'):
                events.append((stamp + 1_000_000, 'arm', side, stamp))
            events.append((stamp + 3_000_000, 'command', 'arms', stamp + 1_000_000))
            if i % 2 == 0:
                events.append((stamp + 3_000_000, 'command', 'mobile', stamp + 1_000_000))
        buffer = CaptureBuffer()
        count = 0
        for now, kind, name, stamp in sorted(events):
            if kind == 'camera':
                buffer.add_camera(name, stamp, b'jpeg')
            elif kind == 'arm':
                buffer.add_arm(name, arm(name, stamp))
            else:
                buffer.add_command(command(name, stamp))
            count += buffer.pick(now) is not None
        self.assertGreaterEqual(count, 172)

    def test_old_generation_late_acknowledgment_and_missing_new_command_are_not_labels(self):
        buffer = self.ready()
        buffer.commands['arms'].pop()  # only the old command, acknowledged after the picture
        self.assertIsNone(buffer.pick(1_020_000_000))
        self.assertIn('新生成', buffer.reason)
        buffer.add_command(command('arms', 1_012_000_000, 2))
        self.assertIsNotNone(buffer.pick(1_020_000_000))
        for i, name in enumerate(CAMERAS):
            buffer.add_camera(name, 1_025_000_000 + i * 500_000, b'jpeg')
        self.assertIsNone(buffer.pick(1_040_000_000))  # never reuse the last action event

    def test_invalid_or_unsent_command_timestamps_are_rejected(self):
        buffer = self.ready()
        bad = command('arms', 1_030_000_000, 3)
        bad['accepted_ns'] = bad['generated_ns'] - 1
        with self.assertRaises(ValueError):
            buffer.add_command(bad)


class TestV3Writer(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.config = {'dataset': 'xyc/unit_test', 'task': '拿起玩具', 'fps': 30, 'scope': 'mobile'}

    def tearDown(self):
        self.temporary.cleanup()

    def writer(self, **overrides):
        return LeRobotV3Writer(self.base, {**self.config, **overrides}, width=16, height=16)

    def record(self, writer, count=5):
        writer.begin()
        for index in range(count):
            writer.add(capture(index))
        return writer.save()

    def test_v3_layout_video_frame_counts_stats_resume_and_tasks(self):
        writer = self.writer()
        self.record(writer)
        root = writer.root
        writer.close()
        writer = self.writer(resume=True)
        self.record(writer)
        writer.close()
        info = json.loads((root / 'meta/info.json').read_text())
        self.assertEqual(info['codebase_version'], 'v3.0')
        self.assertEqual(info['total_episodes'], 2)
        self.assertEqual(info['total_frames'], 10)
        self.assertEqual(info['features']['observation.state']['shape'], [18])
        episodes = [pq.read_table(path).to_pylist()[0] for path in sorted((root / 'meta/episodes').rglob('*.parquet'))]
        self.assertEqual([e['dataset_from_index'] for e in episodes], [0, 5])
        self.assertEqual([e['dataset_to_index'] for e in episodes], [5, 10])
        for path in (root / 'videos').rglob('*.mp4'):
            with av.open(str(path)) as video:
                frames = list(video.decode(video=0))
                self.assertEqual(len(frames), 5)
                self.assertEqual(frames[0].pts, 0)
        stats = json.loads((root / 'meta/stats.json').read_text())
        self.assertEqual(stats['action']['count'], [10])
        self.assertAlmostEqual(stats['action']['mean'][0], 2.5)
        self.assertEqual(len(stats['action']['q01']), 18)
        self.assertGreater(stats['observation.images.forward']['mean'][0][0][0], .7)
        data = pq.read_table(root / 'data/chunk-000/file-000.parquet').to_pylist()
        self.assertAlmostEqual(data[4]['timestamp'], 4 / 30, places=6)
        self.assertAlmostEqual(data[4]['observation.source_time'], 4 / 29.5, places=6)

    def test_discard_and_concurrent_writer_do_not_overwrite_saved_episodes(self):
        writer = self.writer()
        self.record(writer)
        writer.begin()
        writer.add(capture(0))
        writer.discard()
        with self.assertRaises(FileExistsError):
            self.writer(resume=True)
        self.assertTrue(writer.lock_path.exists())
        writer.close()
        with self.assertRaisesRegex(ValueError, 'exists'):
            self.writer()
        with self.assertRaisesRegex(ValueError, 'differ'):
            self.writer(resume=True, fps=15)
        writer = self.writer(resume=True)
        self.assertEqual(writer.info['total_episodes'], 1)
        writer.close()

    def test_crashed_writer_releases_ownership_without_removing_saved_data(self):
        writer = self.writer()
        self.record(writer)
        root = writer.root
        writer.close()
        script = ('import os, sys; from pathlib import Path; '
                  'from xyc_recording.writer import acquire_dataset_lock; '
                  'fd = acquire_dataset_lock(Path(sys.argv[1])); os._exit(0)')
        result = subprocess.run([sys.executable, '-c', script, str(root / '.writer.lock')],
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors='replace'))
        writer = self.writer(resume=True)
        self.assertEqual(writer.info['total_frames'], 5)
        with self.assertRaises(FileExistsError):
            acquire_dataset_lock(writer.lock_path)
        writer.close()

    def test_interrupted_episode_is_not_advertised_as_saved(self):
        writer = self.writer()
        root = writer.root
        writer.begin()
        writer.add(capture(0))
        writer.close()
        self.assertEqual(json.loads((root / 'meta/info.json').read_text())['total_episodes'], 0)
        self.assertEqual(len(list(root.glob('.pending-*'))), 1)
        with self.assertRaisesRegex(ValueError, 'Unfinished'):
            self.writer(resume=True)

    def test_unsafe_names_and_malformed_settings_are_rejected(self):
        for change in ({'dataset': '../escape'}, {'fps': True}, {'episode_frames': 1}, {'task': ''}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                checked_config({**self.config, **change})

    def test_interrupted_prefix_is_committed_with_its_reason_and_real_frame_count(self):
        writer = self.writer()
        writer.begin()
        for index in range(3):
            writer.add(capture(index))
        interruption = {'reason': 'missing accepted arm command', 'gap_s': .55}
        result = writer.save(interruption=interruption)
        quality = json.loads((writer.root / 'meta/quality/episode-000000.json').read_text())
        self.assertEqual(quality['interruption'], interruption)
        self.assertEqual(result['frames'], 3)
        self.assertEqual(writer.info['total_frames'], 3)
        writer.close()

    def test_encoded_video_preserves_rgb_colours_after_explicit_yuv_conversion(self):
        writer = self.writer()
        self.record(writer, count=3)
        path = writer.root / 'videos/observation.images.forward/chunk-000/file-000.mp4'
        with av.open(str(path)) as container:
            frames = list(container.decode(video=0))
        expected = cv2.imdecode(np.frombuffer(capture(0).images[0].jpeg, dtype=np.uint8),
                                cv2.IMREAD_COLOR)[0, 0, ::-1].astype(float)
        actual = frames[0].to_ndarray(format='rgb24')[0, 0].astype(float)
        self.assertEqual(len(frames), 3)
        self.assertLess(np.max(np.abs(expected - actual)), 12)
        writer.close()

    def test_capture_stages_jpeg_without_encoding_and_discard_skips_transcoding(self):
        writer = self.writer()
        writer.begin()
        with patch.object(writer, 'encode_image', wraps=writer.encode_image) as encode:
            for i in range(3):writer.add(capture(i))
            encode.assert_not_called()
            self.assertEqual(len(writer.raw_paths), 4)
            self.assertFalse(writer.videos)
            result = writer.save()
            self.assertEqual(encode.call_count, 12)
            self.assertEqual(result['frames'], 3)
            writer.begin()
            writer.add(capture(0))
            encode.reset_mock()
            writer.discard()
            encode.assert_not_called()
        writer.close()

    def test_staged_video_count_mismatch_cannot_commit_an_episode(self):
        writer = self.writer()
        writer.begin()
        for i in range(3):writer.add(capture(i))
        writer.close_videos(encode=False)
        writer.raw_paths['forward'].write_bytes(jpeg(0))
        with self.assertRaisesRegex(ValueError, 'staged JPEG count'):
            writer.save()
        self.assertEqual(writer.info['total_episodes'], 0)
        self.assertTrue(writer.stage.exists())
        writer.close()

    def test_writer_rejects_past_action_and_resume_into_older_action_semantics(self):
        writer = self.writer()
        writer.begin()
        with self.assertRaisesRegex(ValueError, 'after the observation'):
            writer.add(replace(capture(0), action_stamps_ns=(999_000_000,) * 18))
        writer.discard()
        self.record(writer)
        root = writer.root
        writer.close()
        path = root / 'meta/robot.json'
        assets = json.loads(path.read_text())
        self.assertEqual(assets.pop('action_semantics'), ACTION_SEMANTICS)
        path.write_text(json.dumps(assets))
        with self.assertRaisesRegex(ValueError, 'action时间约定'):
            self.writer(resume=True)

    def test_writer_rejects_physical_mobile_speed_instead_of_ternary_label(self):
        writer = self.writer()
        writer.begin()
        legacy = replace(capture(0), action=tuple([0.] * 14 + [.015, .15, 0., .60]))
        with self.assertRaisesRegex(ValueError, r'exactly -1, 0 or \+1'):
            writer.add(legacy)
        writer.discard()
        writer.close()

    @unittest.skipUnless(os.environ.get('XYC_OFFICIAL_READER') == '1', 'optional official LeRobot reader check')
    def test_official_lerobot_reads_both_episodes_all_videos_and_action_chunks(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        writer = self.writer()
        self.record(writer)
        self.record(writer)
        root = writer.root
        writer.close()
        dataset = LeRobotDataset('xyc/unit_test', root=root, video_backend='pyav',
                                delta_timestamps={'action': [0., 1 / 30, 2 / 30]})
        self.assertEqual(len(dataset), 10)
        for index in (0, 4, 5, 9):
            item = dataset[index]
            self.assertEqual(item['task'], '拿起玩具')
            self.assertEqual(tuple(item['observation.state'].shape), (18,))
            self.assertEqual(tuple(item['action'].shape), (3, 18))
            for name in CAMERAS:
                rgb = item['observation.images.' + name]
                self.assertEqual(tuple(rgb.shape), (3, 16, 16))
                self.assertGreater(float(rgb[0].mean()), .7)
        self.assertTrue(dataset[4]['action_is_pad'][1:].all())
