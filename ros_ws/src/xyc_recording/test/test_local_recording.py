# SPDX-License-Identifier: Apache-2.0
"""Real loopback transport and local v3 writes; no robot devices are opened."""

import json
from pathlib import Path
import struct
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import av
import pyarrow.parquet as pq

from test_recording import capture
from xyc_recording.local_recorder import LocalDatasetReceiver
from xyc_recording.cache import CachedDesktopWriter
from xyc_recording.stream import (
    CaptureRelay, RecordingBackpressure, RelayServer, pack_event, unpack_event,
)
from xyc_recording.alignment import (
    ACTION_SEMANTICS, ACTION_UNITS, RECORDING_CONTRACT, STATE_UNITS,
)


def robot_assets():
    return {'robot': 'synthetic', 'recording_contract': RECORDING_CONTRACT,
            'action_semantics': ACTION_SEMANTICS,
            'state_units': list(STATE_UNITS), 'action_units': list(ACTION_UNITS)}


class TestRecordingPackets(unittest.TestCase):
    def test_roundtrip_preserves_each_source_timestamp_command_id_and_original_jpeg(self):
        original = capture(3)
        event = unpack_event(pack_event(7, 'capture', capture=original))
        self.assertEqual(event['sequence'], 7)
        self.assertEqual(event['capture'], original)

    def test_truncated_oversized_headers_trailing_bytes_and_wrong_images_are_rejected(self):
        packet = pack_event(1, 'capture', capture=capture(0))
        for value in (b'', packet[:-10], packet + b'extra', struct.pack('!I', 90000) + b'{}'):
            with self.subTest(size=len(value)), self.assertRaises((ValueError, KeyError)):
                unpack_event(value)

    def test_queue_is_bounded_and_acknowledgments_are_idempotent(self):
        relay = CaptureRelay(max_events=1)
        relay.heartbeat('desktop')
        first = relay.put('capture', capture=capture(0))
        with self.assertRaises(RecordingBackpressure):
            relay.put('capture', capture=capture(1))
        ack = {'sequence': first.sequence, 'accepted': True, 'result': {}}
        self.assertEqual(unpack_event(relay.exchange('desktop', ack))['operation'], 'idle')
        self.assertEqual(unpack_event(relay.exchange('desktop', ack))['operation'], 'idle')
        self.assertTrue(first.done.is_set())
        self.assertEqual(relay.bytes, 0)
        with self.assertRaises(ConnectionError):
            relay.heartbeat('another-desktop')
        relay.close()

    def test_failed_desktop_write_blocks_further_frames_and_cannot_report_a_save(self):
        relay = CaptureRelay()
        relay.heartbeat('desktop')
        first = relay.put('capture', capture=capture(0))
        queued = relay.put('capture', capture=capture(1))
        relay.exchange('desktop', {'sequence': first.sequence, 'accepted': False, 'message': 'disk full'})
        self.assertTrue(queued.done.is_set())
        self.assertEqual(queued.error, 'disk full')
        for operation in ('capture', 'save'):
            with self.subTest(operation=operation), self.assertRaisesRegex(OSError, 'disk full'):
                relay.put(operation)
        relay.close()


class TestLocalRecording(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='xyc-local-recording-')
        self.addCleanup(self.directory.cleanup)
        self.config = {'dataset': 'xyc/local_test', 'task': 'test local recording', 'fps': 30,
                       'episode_frames': 5, 'scope': 'mobile', 'resume': False}
        self.relay = CaptureRelay()
        self.server = RelayServer(('127.0.0.1', 0), self.relay)
        self.server_worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_worker.start()
        self.desktop = Path(self.directory.name) / 'desktop'
        self.robot_cache = Path(self.directory.name) / 'robot-cache'
        self.receiver = LocalDatasetReceiver(self.desktop, self.config,
            url=f'http://127.0.0.1:{self.server.server_address[1]}', width=16, height=16).start()
        deadline = time.monotonic() + 5.
        while not self.receiver.ready.wait(.02) and time.monotonic() < deadline:
            pass
        self.assertTrue(self.receiver.ready.is_set())
        self.writer = CachedDesktopWriter(self.relay, self.config, robot_assets=robot_assets(),
                                           cache_dir=self.robot_cache)

    def tearDown(self):
        self.writer.close()
        self.receiver.close()
        self.relay.close()
        self.server.shutdown()
        self.server.server_close()
        self.server_worker.join()

    def record(self, count=5):
        self.writer.begin()
        for index in range(count):
            self.writer.add(capture(index))
        return self.writer.save()

    def test_complete_samples_are_saved_on_desktop_with_four_matching_videos(self):
        result = self.record()
        root = self.desktop / self.config['dataset']
        self.assertEqual(result['path'], str(root.resolve()))
        self.assertEqual(result['frames'], 5)
        self.assertEqual(self.writer.info['total_frames'], 5)
        rows = pq.read_table(root / 'data/chunk-000/file-000.parquet').to_pylist()
        self.assertEqual(len(rows), 5)
        self.assertEqual(len({row['action.arm_command_id'] for row in rows}), 5)
        self.assertAlmostEqual(rows[-1]['observation.source_time'], 4 / 29.5)
        for video in root.glob('videos/*/chunk-000/file-000.mp4'):
            with av.open(str(video)) as container:
                self.assertEqual(sum(1 for _ in container.decode(video=0)), 5)
        self.assertEqual(len(list(root.glob('videos/*/chunk-000/file-000.mp4'))), 4)
        self.assertFalse(list(root.glob('.pending-*')))
        self.assertFalse(list(self.robot_cache.glob('episode-*')))

    def test_recording_only_writes_robot_cache_until_episode_ends(self):
        self.writer.begin()
        for index in range(3):
            self.writer.add(capture(index))
        self.writer.output.flush()
        self.assertGreater((self.writer.stage / 'frames.bin').stat().st_size, 0)
        self.assertEqual(self.receiver.frames, 0)
        self.assertIsNone(self.receiver.writer.stage)
        self.assertEqual(self.writer.info['total_frames'], 0)
        self.writer.discard()

    def test_lost_ack_replays_do_not_duplicate_samples_or_commit_twice(self):
        original = self.relay.exchange
        replayed = set()
        def exchange(client, ack=None):
            sequence = ack[-1]['sequence'] if isinstance(ack, list) and ack else (ack or {}).get('sequence')
            if sequence is not None and sequence not in replayed:
                replayed.add(sequence)
                return original(client, None)  # Simulate an ACK lost before reaching the server.
            return original(client, ack)
        with patch.object(self.relay, 'exchange', side_effect=exchange):
            self.record()
        self.assertEqual(self.writer.info['total_episodes'], 1)
        self.assertEqual(self.writer.info['total_frames'], 5)

    def test_discard_and_second_episode_preserve_saved_data(self):
        self.record(3)
        self.writer.begin()
        self.writer.add(capture(0))
        self.writer.discard()
        self.record(4)
        self.assertEqual(self.writer.info['total_episodes'], 2)
        self.assertEqual(self.writer.info['total_frames'], 7)

    def test_failed_transfer_keeps_durable_cache_and_retry_saves_once(self):
        self.writer.begin()
        for index in range(3):
            self.writer.add(capture(index))
        original = self.relay.call
        def fail_save(operation, *args, **kwargs):
            if operation == 'save':raise ConnectionError('simulated transfer interruption')
            return original(operation, *args, **kwargs)
        with patch.object(self.relay, 'call', side_effect=fail_save), self.assertRaisesRegex(OSError, '缓存保留'):
            self.writer.save()
        self.assertTrue((self.writer.stage / 'frames.bin').is_file())
        self.assertEqual(json.loads((self.desktop / self.config['dataset'] / 'meta/info.json').read_text())['total_frames'], 0)
        result = self.writer.save()
        self.assertEqual(result['frames'], 3)
        self.assertEqual(self.writer.info['total_episodes'], 1)
        self.assertFalse(list(self.robot_cache.glob('episode-*')))

    def test_lost_final_ack_does_not_duplicate_a_committed_episode(self):
        self.writer.begin()
        for index in range(3):self.writer.add(capture(index))
        original = self.relay.call
        def lose_ack(operation, *args, **kwargs):
            result = original(operation, *args, **kwargs)
            if operation == 'save':raise ConnectionError('lost final commit receipt')
            return result
        with patch.object(self.relay, 'call', side_effect=lose_ack), self.assertRaises(OSError):
            self.writer.save()
        self.assertIsNotNone(self.writer.stage)
        self.writer.save()
        self.assertEqual(self.writer.info['total_episodes'], 1)
        self.assertEqual(self.writer.info['total_frames'], 3)

    def test_commit_before_receipt_file_is_recovered_from_episode_quality(self):
        self.writer.begin()
        for index in range(3):self.writer.add(capture(index))
        original = self.relay.call
        def lose_ack(operation, *args, **kwargs):
            result = original(operation, *args, **kwargs)
            if operation == 'save':raise ConnectionError('lost receipt')
            return result
        with patch.object(self.relay, 'call', side_effect=lose_ack), self.assertRaises(OSError):self.writer.save()
        (self.desktop / self.config['dataset'] / 'meta/transfers' / (self.writer.cache_id + '.json')).unlink()
        self.writer.save()
        self.assertEqual(self.writer.info['total_episodes'], 1)
        self.assertFalse(list(self.robot_cache.glob('episode-*')))

    def test_local_close_waits_for_delayed_robot_save_after_quit(self):
        self.writer.begin()
        for index in range(3):self.writer.add(capture(index))
        self.receiver.last_activity -= 10.
        errors = []
        def save():
            try:self.writer.save()
            except Exception as exc:errors.append(str(exc))
        request = threading.Timer(.15, save)
        request.start()
        self.receiver.close()
        request.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.writer.info['total_frames'], 3)
        self.relay.client_at = 0.

    def test_restart_retransmits_same_cache_and_recovers_matching_desktop_staging(self):
        self.writer.begin()
        for index in range(3):self.writer.add(capture(index))
        original = self.relay.call
        def fail_save(operation, *args, **kwargs):
            if operation == 'save':raise ConnectionError('interrupted before commit')
            return original(operation, *args, **kwargs)
        with patch.object(self.relay, 'call', side_effect=fail_save), self.assertRaises(OSError):self.writer.save()
        self.writer.close()
        self.receiver.close()
        self.relay.client_at = 0.
        resumed = {**self.config, 'resume': True}
        self.receiver = LocalDatasetReceiver(self.desktop, resumed,
            url=f'http://127.0.0.1:{self.server.server_address[1]}', width=16, height=16).start()
        self.assertTrue(self.receiver.ready.wait(5.))
        self.writer = CachedDesktopWriter(self.relay, resumed, robot_assets=robot_assets(),
                                           cache_dir=self.robot_cache)
        self.assertTrue(self.writer.recovered)
        self.writer.begin()
        self.writer.save()
        self.assertEqual(self.writer.info['total_frames'], 3)
        self.assertEqual(self.writer.info['total_episodes'], 1)
