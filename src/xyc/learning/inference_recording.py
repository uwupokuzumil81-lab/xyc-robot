"""Model-independent inference traces, written by a bounded background worker.

Predictions, selections, dispatch results and measured feedback are distinct records.
This module never sends commands, opens serial ports or imports ROS.
"""

from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
import logging
from pathlib import Path
from queue import Empty, Queue
import threading
import time
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from xyc.io import write_json


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


class InferenceRecorder:
    """One run directory. Producers never wait for the disk/encoder queue.

    Runtime disk/encoder errors disable logging, not the caller's control loop.
    The final metadata explicitly marks missing records and unfinished files.
    Source timestamps must name their own clock; recorder timestamps are local
    monotonic nanoseconds and must not be subtracted from another host's clock.
    """
    def __init__(self, directory, *, schema, metadata=None, record_video=False,
                 max_queue=256, max_bytes=64 * 1024 * 1024, flush_rows=64):
        if min(max_queue, max_bytes, flush_rows) < 1:
            raise ValueError('Recording queue and flush limits must be positive')
        if record_video and any(not name or name in ('.', '..') or '/' in name or '\\' in name
                                for name in schema.cameras):
            raise ValueError('Video camera names must be simple file names')
        self.path = Path(directory).expanduser().resolve()
        self.schema, self.record_video = schema, record_video
        self.max_bytes, self.flush_rows = max_bytes, flush_rows
        self._queue, self._lock = Queue(maxsize=max_queue), threading.Lock()
        self._closed = self._closing = False
        self._queued_bytes = self._prediction_id = self._step_id = 0
        self._submitted, self._written, self._dropped = Counter(), Counter(), Counter()
        self._error, self._video_count, self._video_shapes = None, 0, {}
        self._started_ns = time.perf_counter_ns()
        self._metadata = {
            'format': 'xyc.inference.record.v1', 'run_id': uuid4().hex,
            'status': 'running', 'started_at_utc': utc_now(),
            'started_monotonic_ns': self._started_ns,
            'clock': 'recorder_host_monotonic_ns; source timestamps retain their named clock',
            'schema': schema.to_dict(), 'context': json_copy(metadata or {}),
            'video': {'enabled': record_video, 'cameras': list(schema.cameras),
                      'fps': schema.fps, 'time_semantics': 'frame_index / fps; see steps for original observation time'},
            'queue': {'max_events': max_queue, 'max_bytes': max_bytes},
        }
        self.path.mkdir(parents=True, exist_ok=False)
        write_json(self.path / 'meta.json', self._metadata)
        if record_video:
            (self.path / 'videos').mkdir()
        self._thread = threading.Thread(target=self._work, name='xyc-inference-writer', daemon=True)
        self._thread.start()
        self.event('run_started')

    @classmethod
    def create(cls, root, **kwargs):
        name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
        return cls(Path(root) / name, **kwargs)

    def _vector(self, value, size, *, missing=False):
        if value is None:
            return None
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (size,) or not missing and not np.isfinite(array).all():
            raise ValueError(f'Expected a finite vector of length {size}')
        return [float(v) if np.isfinite(v) else None for v in array]

    def _put(self, kind, row, images=None):
        size = len(json.dumps(row, allow_nan=False).encode('utf-8'))
        size += sum(image.nbytes for image in (images or {}).values())
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError('Inference recording is already closed')
            self._submitted[kind] += 1
            if self._error or self._queue.full() or self._queued_bytes + size > self.max_bytes:
                self._dropped[kind] += 1
                if sum(self._dropped.values()) == 1:
                    logging.getLogger(__name__).warning('Inference trace cannot keep up; missing records will be reported in meta.json')
                return False
            self._queued_bytes += size
            self._queue.put_nowait((kind, row, images, size))
        return True

    def event(self, name, **details):
        return self._put('event', {'event': str(name), 'monotonic_ns': time.perf_counter_ns(),
                                   'at_utc': utc_now(), 'details': json_copy(details)})

    @property
    def closed(self):
        return self._closed

    def check_export_path(self, path):
        path = Path(path).expanduser().resolve()
        if path.is_relative_to(self.path):
            relative = path.relative_to(self.path)
            if relative.parts[0] == 'videos' or relative.name in {
                    'meta.json', 'meta.json.tmp', 'events.jsonl', 'steps.parquet',
                    'steps.partial.parquet', 'predictions.parquet', 'predictions.partial.parquet'}:
                raise ValueError('Prediction JSON cannot overwrite an inference trace asset')

    def prediction(self, actions, *, state, observation=None, latency_s=0.):
        if hasattr(actions, 'detach'):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float64)
        if (actions.ndim != 2 or not len(actions) or actions.shape[1] != len(self.schema.action_names)
                or not np.isfinite(actions).all() or not np.isfinite(latency_s) or latency_s < 0):
            raise ValueError('Invalid physical-unit prediction chunk or latency')
        with self._lock:
            self._prediction_id += 1
            identity = self._prediction_id
        self._put('prediction', {'prediction_id': identity, 'monotonic_ns': time.perf_counter_ns(),
                  'observation_json': json.dumps(json_copy(observation or {}), ensure_ascii=False),
                  'state': self._vector(state, len(self.schema.state_names)),
                  'actions': actions.tolist(), 'latency_s': float(latency_s)})
        return identity

    def _step_row(self, kind, step_id):
        if type(step_id) is not int or not 1 <= step_id <= self._step_id:
            raise ValueError('Execution/feedback must refer to an existing recorded selection step')
        return {name: None for name in (
            'prediction_id', 'chunk_index', 'selection', 'contributors_json', 'state',
            'selected_action', 'sent_action', 'driver_accepted', 'feedback_state',
            'observation_json', 'command_id', 'source_time_ns', 'source_clock',
            'message', 'video_frame_index', 'video_pts_s')} | {
                'record_kind': kind, 'step_id': int(step_id), 'monotonic_ns': time.perf_counter_ns()}

    def selection(self, action, *, state, prediction_id, chunk_index, observation=None,
                  selection='queue', contributors=None, images=None):
        if type(prediction_id) is not int or not 1 <= prediction_id <= self._prediction_id or chunk_index < 0:
            raise ValueError('Selection must refer to a prediction in this recording')
        with self._lock:
            self._step_id += 1
            identity = self._step_id
        row = self._step_row('selection', identity)
        row.update(prediction_id=int(prediction_id), chunk_index=int(chunk_index), selection=selection,
                   contributors_json=json.dumps(json_copy(contributors or [])),
                   observation_json=json.dumps(json_copy(observation or {}), ensure_ascii=False),
                   state=self._vector(state, len(self.schema.state_names)),
                   selected_action=self._vector(action, len(self.schema.action_names)))
        copied = None
        if self.record_video:
            if images is None or set(images) != set(self.schema.cameras):
                raise ValueError('Video recording requires one RGB uint8 HWC image per configured camera')
            copied = {}
            for name in self.schema.cameras:
                value = np.asarray(images[name])
                if value.dtype != np.uint8 or value.ndim != 3 or value.shape[2] != 3:
                    raise ValueError('Recorded images must be RGB uint8 HWC')
                copied[name] = np.ascontiguousarray(value).copy()
        self._put('selection', row, copied)
        return identity

    def execution(self, step_id, *, sent_action, accepted=None, command_id=None,
                  source_time_ns=None, source_clock=None, message=''):
        if accepted is not None and type(accepted) is not bool:
            raise ValueError('Driver acceptance must be true, false or unknown (None)')
        self._check_source_clock(source_time_ns, source_clock)
        row = self._step_row('execution', step_id)
        row.update(sent_action=self._vector(sent_action, len(self.schema.action_names)),
                   driver_accepted=accepted, command_id=str(command_id) if command_id is not None else None,
                   source_time_ns=source_time_ns, source_clock=source_clock, message=message)
        self._put('execution', row)

    def feedback(self, step_id, *, state, source_time_ns=None, source_clock=None, message=''):
        self._check_source_clock(source_time_ns, source_clock)
        row = self._step_row('feedback', step_id)
        row.update(feedback_state=self._vector(state, len(self.schema.state_names), missing=True),
                   source_time_ns=source_time_ns, source_clock=source_clock, message=message)
        self._put('feedback', row)

    @staticmethod
    def _check_source_clock(stamp, clock):
        if stamp is not None and (type(stamp) is not int or not clock):
            raise ValueError('Source timestamps require integer nanoseconds and a named source clock')

    def _arrow_schemas(self):
        vector = pa.list_(pa.float64())
        predictions = pa.schema([
            ('prediction_id', pa.int64()), ('monotonic_ns', pa.int64()), ('observation_json', pa.string()),
            ('state', vector), ('actions', pa.list_(vector)), ('latency_s', pa.float64())])
        steps = pa.schema([
            ('record_kind', pa.string()), ('step_id', pa.int64()), ('monotonic_ns', pa.int64()),
            ('prediction_id', pa.int64()), ('chunk_index', pa.int64()), ('selection', pa.string()),
            ('contributors_json', pa.string()), ('state', vector), ('selected_action', vector),
            ('sent_action', vector), ('driver_accepted', pa.bool_()), ('feedback_state', vector),
            ('observation_json', pa.string()), ('command_id', pa.string()),
            ('source_time_ns', pa.int64()), ('source_clock', pa.string()), ('message', pa.string()),
            ('video_frame_index', pa.int64()), ('video_pts_s', pa.float64())])
        return predictions, steps

    def _work(self):
        writers, buffers, videos, events = {}, {'predictions': [], 'steps': []}, {}, None
        try:
            for name, schema in zip(('predictions', 'steps'), self._arrow_schemas()):
                writers[name] = pq.ParquetWriter(self.path / (name + '.partial.parquet'), schema, compression='zstd')
            events = (self.path / 'events.jsonl').open('a', encoding='utf-8')
            last_flush = time.monotonic()
            while True:
                try:
                    kind, row, images, size = self._queue.get(timeout=.1)
                except Empty:
                    with self._lock:
                        if self._closing:
                            break
                    kind = None
                if kind is not None:
                    try:
                        if kind == 'event':
                            events.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                            events.flush()
                            self._written['event'] += 1
                        else:
                            if images is not None:
                                self._write_images(images, row, videos)
                            buffers['predictions' if kind == 'prediction' else 'steps'].append(row)
                    finally:
                        with self._lock:
                            self._queued_bytes -= size
                        self._queue.task_done()
                if (sum(map(len, buffers.values())) >= self.flush_rows
                        or time.monotonic() - last_flush >= .5):
                    self._flush(writers, buffers)
                    last_flush = time.monotonic()
            self._flush(writers, buffers)
        except Exception as exc:
            self._error = f'{type(exc).__name__}: {exc}'
            logging.getLogger(__name__).error('Inference trace writer failed: %s', self._error)
        finally:
            for writer in writers.values():
                try:
                    writer.close()
                except Exception as exc:
                    self._error = self._error or str(exc)
            for container, stream in videos.values():
                try:
                    for packet in stream.encode():
                        container.mux(packet)
                    container.close()
                except Exception as exc:
                    self._error = self._error or str(exc)
            if events is not None:
                try:
                    events.close()
                except Exception as exc:
                    self._error = self._error or str(exc)
            if not self._error:
                try:
                    for name in writers:
                        (self.path / (name + '.partial.parquet')).replace(self.path / (name + '.parquet'))
                    for name in videos:
                        (self.path / 'videos' / (name + '.partial.mp4')).replace(self.path / 'videos' / (name + '.mp4'))
                except Exception as exc:
                    self._error = str(exc)

    def _flush(self, writers, buffers):
        for name, rows in buffers.items():
            if rows:
                writers[name].write_table(pa.Table.from_pylist(rows, schema=writers[name].schema))
                for row in rows:
                    self._written['prediction' if name == 'predictions' else row['record_kind']] += 1
                rows.clear()

    def _write_images(self, images, row, videos):
        import av
        rate = Fraction(str(self.schema.fps)).limit_denominator(100000)
        for name, image in images.items():
            if name not in videos:
                container = av.open(str(self.path / 'videos' / (name + '.partial.mp4')), 'w', format='mp4')
                stream = container.add_stream('libx264', rate=rate)
                stream.height, stream.width = image.shape[:2]
                stream.pix_fmt = 'yuv420p' if not (stream.height % 2 or stream.width % 2) else 'yuv444p'
                stream.options = {'crf': '23', 'preset': 'veryfast'}
                videos[name] = container, stream
                self._video_shapes[name] = list(image.shape)
            if list(image.shape) != self._video_shapes[name]:
                raise ValueError('Camera shape changed during recording: ' + name)
            container, stream = videos[name]
            frame = av.VideoFrame.from_ndarray(image, format='rgb24')
            frame.pts, frame.time_base = self._video_count, 1 / rate
            for packet in stream.encode(frame):
                container.mux(packet)
        row['video_frame_index'] = self._video_count
        row['video_pts_s'] = self._video_count / self.schema.fps
        self._video_count += 1

    def close(self, *, outcome='completed', reason=None, task_success=None):
        if self._closed:
            return self.summary
        if task_success is not None and type(task_success) is not bool:
            raise ValueError('Task success must be explicitly true, false or unknown (None)')
        self.event('run_finished', outcome=outcome, reason=reason)
        with self._lock:
            self._closing = True
        self._thread.join()
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Empty:
                break
        with self._lock:
            self._closed = True
        missing = self._submitted - self._written
        status = 'logging_failed' if self._error else 'degraded' if missing and outcome == 'completed' else outcome
        self.summary = {**self._metadata, 'status': status, 'outcome': outcome, 'reason': reason,
                        'task_success': task_success, 'ended_at_utc': utc_now(),
                        'elapsed_s': (time.perf_counter_ns() - self._started_ns) / 1e9,
                        'submitted': dict(self._submitted), 'written': dict(self._written),
                        'missing': dict(missing), 'writer_error': self._error,
                        'video_frames_per_camera': self._video_count, 'video_shapes': self._video_shapes}
        try:
            write_json(self.path / 'meta.json', self.summary)
        except OSError as exc:
            self.summary.update(status='logging_failed', writer_error=str(exc))
            logging.getLogger(__name__).error('Could not finalize inference metadata: %s', exc)
        return self.summary

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        self.close(outcome='failed' if error else 'completed', reason=str(error) if error else None)
        return False
