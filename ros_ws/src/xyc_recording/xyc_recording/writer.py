# SPDX-License-Identifier: Apache-2.0
"""Portable LeRobot v3.0 writer used on the operator's computer.

Each committed episode occupies a v3 data/video file, with explicit episode
offsets. Actual source times are retained separately from the nominal FPS grid.
The upstream LeRobot reader is used by the compatibility test.
"""

from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time
from uuid import uuid4

import av
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .alignment import (
    ACTION_NAMES, ACTION_SEMANTICS, CAMERAS, STATE_NAMES, CameraFrame,
    checked_ternary_mobile_action,
)
from .config import checked_config


def json_write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open('w', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def acquire_dataset_lock(path):
    """The OS releases ownership on exit, including crashes and Ctrl+C.

    Keep the lock file in place: unlinking it permits POSIX writers to lock
    different inodes at the same path and write concurrently.
    """
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        os.close(descriptor)
        raise FileExistsError(f'数据集正在被另一个采集程序写入：{path.parent}') from exc
    try:
        os.ftruncate(descriptor, 0)
        os.write(descriptor, str(os.getpid()).encode())
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def scalar_stats(array):
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    out = {name: function(values, axis=0).tolist() for name, function in
           (('min', np.min), ('max', np.max), ('mean', np.mean), ('std', np.std))}
    out['count'] = [len(values)]
    for q in (.01, .10, .50, .90, .99):
        out[f'q{round(q * 100):02d}'] = np.quantile(values, q, axis=0).tolist()
    return out


def histogram_stats(histogram, frames):
    hist = histogram.astype(np.float64)
    counts = hist.sum(axis=1)
    levels = np.arange(256, dtype=np.float64) / 255.
    means = (hist * levels).sum(axis=1) / counts
    variance = (hist * levels ** 2).sum(axis=1) / counts - means ** 2
    cumulative = np.cumsum(hist, axis=1)
    def shaped(values):
        return np.asarray(values).reshape(3, 1, 1).tolist()
    out = {'mean': shaped(means), 'std': shaped(np.sqrt(np.maximum(0, variance))),
           'min': shaped([np.flatnonzero(h)[0] / 255. for h in hist]),
           'max': shaped([np.flatnonzero(h)[-1] / 255. for h in hist]), 'count': [frames]}
    for q in (.01, .10, .50, .90, .99):
        out[f'q{round(q * 100):02d}'] = shaped([
            min(255, np.searchsorted(cumulative[i], q * counts[i])) / 255. for i in range(3)])
    return out


class LeRobotV3Writer:
    def __init__(self, datasets_dir, config, *, robot_assets=None, width=640, height=480,
                 recover_transfer_id=None):
        self.config = checked_config(config)
        self.root = (Path(datasets_dir).expanduser().resolve() / self.config['dataset']).resolve()
        if not self.root.is_relative_to(Path(datasets_dir).expanduser().resolve()):
            raise ValueError('Dataset path escaped datasets_dir')
        self.width, self.height = width, height
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError('Video size must be positive and even')
        self.size = 18 if self.config['scope'] == 'mobile' else 14
        self.stage = None
        self.committing = False
        self.videos = {}
        self.raw_files = {}
        self.raw_paths = {}
        self.raw_transcoded = False
        self.rows = []
        self.hist = {name: np.zeros((3, 256), dtype=np.int64) for name in CAMERAS}
        self.lock_path = self.root / '.writer.lock'
        self.lock_fd = None
        self.encode_workers = None
        assets_expected = {**(robot_assets or {}), 'action_semantics': ACTION_SEMANTICS}
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            self.lock_fd = acquire_dataset_lock(self.lock_path)
            pending = list(self.root.glob('.pending-*'))
            if pending and (not recover_transfer_id or not self.config['resume']):
                raise ValueError('Unfinished .pending episode found; preserve it and recover before resuming')
            info_path = self.root / 'meta/info.json'
            if info_path.exists():
                if not self.config['resume']:
                    raise ValueError('Dataset exists; explicitly use resume to append')
                self.info = json.loads(info_path.read_text(encoding='utf-8'))
                assets = json.loads((self.root / 'meta/robot.json').read_text(encoding='utf-8'))
                if assets.get('action_semantics') != ACTION_SEMANTICS:
                    raise ValueError('旧数据使用不同的action时间约定，请换一个数据集名称；已有数据不会改写')
                if (self.info.get('codebase_version') != 'v3.0'
                        or self.info.get('fps') != self.config['fps']
                        or self.info.get('features', {}) != self.features()):
                    raise ValueError('Resume FPS, feature names, units or image sizes differ')
                self.tasks = pd.read_parquet(self.root / 'meta/tasks.parquet')
                if assets != assets_expected:
                    raise ValueError('Calibration/robot assets differ from the existing dataset')
            else:
                self.info = {'codebase_version': 'v3.0', 'robot_type': 'xyc_alohamini',
                             'total_episodes': 0, 'total_frames': 0, 'total_tasks': 0,
                             'chunks_size': 1000, 'data_files_size_in_mb': 100,
                             'video_files_size_in_mb': 200, 'fps': self.config['fps'], 'splits': {},
                             'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                             'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
                             'features': self.features()}
                self.tasks = pd.DataFrame({'task_index': pd.Series(dtype='int64')})
                json_write(self.root / 'meta/robot.json', assets_expected)
                json_write(self.root / 'meta/info.json', self.info)
                self.tasks.to_parquet(self.root / 'meta/tasks.parquet')
            for stage in pending:
                marker = stage / 'transfer.json'
                if (stage.is_symlink() or not marker.is_file()
                        or json.loads(marker.read_text(encoding='utf-8')).get('cache_id') != recover_transfer_id):
                    raise ValueError('Unfinished episode belongs to a different transfer; refusing to remove it')
            for stage in pending:
                # Explicit resume of the SAME durable Pi cache. Final dataset
                # files are untouched; the source can retransmit the full episode.
                shutil.rmtree(stage)
            self.task = self.config['task']
            self.task_index = int(self.tasks.loc[self.task].task_index) if self.task in self.tasks.index else len(self.tasks)
            cv2.setNumThreads(1)
            self.encode_workers = ThreadPoolExecutor(max_workers=4, thread_name_prefix='camera-video')
        except BaseException:
            self.close()
            raise

    def features(self):
        def vector(dtype, length, names=None):
            return {'dtype': dtype, 'shape': [length], 'names': names}
        result = {'observation.state': vector('float32', self.size, list(STATE_NAMES[:self.size])),
                  'action': vector('float32', self.size, list(ACTION_NAMES[:self.size])),
                  'observation.source_time': vector('float64', 1),
                  'observation.camera_times': vector('float64', 4, list(CAMERAS)),
                  'observation.state_times': vector('float64', self.size),
                  'action.sample_times': vector('float64', self.size),
                  'action.sent_times': vector('float64', self.size),
                  'action.accepted_times': vector('float64', self.size),
                  'action.arm_command_id': vector('int64', 1),
                  'observation.camera_skew_s': vector('float32', 1)}
        if self.size == 18:
            result['action.mobile_command_id'] = vector('int64', 1)
        for key in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index'):
            result[key] = vector('float32' if key == 'timestamp' else 'int64', 1)
        for name in CAMERAS:
            result['observation.images.' + name] = {
                'dtype': 'video', 'shape': [self.height, self.width, 3],
                'names': ['height', 'width', 'channels'],
                'info': {'video.fps': self.config['fps'], 'video.height': self.height,
                         'video.width': self.width, 'video.channels': 3,
                         'video.codec': 'h264', 'video.pix_fmt': 'yuv420p',
                         'video.is_depth_map': False, 'has_audio': False}}
        return result

    def begin(self):
        if self.stage is not None:
            raise ValueError('An episode is already open')
        self.stage = self.root / ('.pending-' + uuid4().hex)
        self.stage.mkdir()
        self.rows = []
        self.origin_ns = None
        self.identity = None
        self.hist = {name: np.zeros((3, 256), dtype=np.int64) for name in CAMERAS}
        self.capture_batches = 0
        self.capture_seconds = 0.
        self.raw_files = {}
        self.raw_paths = {}
        self.raw_transcoded = False

    def add(self, capture):
        if self.stage is None:
            raise ValueError('No open episode')
        if len(capture.state) != self.size or len(capture.action) != self.size:
            raise ValueError('Capture dimension differs from dataset scope')
        observed_until = max(*(image.stamp_ns for image in capture.images), *capture.state_stamps_ns)
        if (len(capture.action_stamps_ns) != self.size or len(capture.sent_times_ns) != self.size
                or len(capture.accepted_times_ns) != self.size or capture.arm_command_id <= 0
                or self.size == 18 and capture.mobile_command_id <= 0
                or any(not observed_until <= generated <= sent <= accepted
                       for generated, sent, accepted in zip(capture.action_stamps_ns,
                                                           capture.sent_times_ns, capture.accepted_times_ns))):
            raise ValueError('Action must be a new identified command generated after the observation')
        identity = (capture.calibration_ids, capture.sessions)
        if self.identity is not None and self.identity != identity:
            raise ValueError('Calibration or control session changed inside an episode')
        self.identity = identity
        if self.origin_ns is None:
            self.origin_ns = capture.stamp_ns
        index = len(self.rows)
        if index and capture.stamp_ns <= self.last_ns:
            raise ValueError('Source timestamp repeated or went backwards')
        self.last_ns = capture.stamp_ns
        encode_started = time.perf_counter()
        if len(capture.images) != len(CAMERAS):
            raise ValueError('Exactly four camera images are required')
        # The cameras already provide JPEG. Persist those bytes in order now;
        # decoding and H264 encoding run after the episode stops collecting.
        for name, image in zip(CAMERAS, capture.images):
            if name not in self.raw_files:
                path = self.stage / (name + '.mjpeg')
                self.raw_paths[name] = path
                self.raw_files[name] = path.open('wb', buffering=1024 * 1024)
            if self.raw_files[name].write(image.jpeg) != len(image.jpeg):
                raise OSError(f'{name}: incomplete JPEG staging write')
        self.capture_batches += 1
        self.capture_seconds += time.perf_counter() - encode_started
        relative = lambda t: (t - self.origin_ns) / 1e9
        action = list(capture.action)
        if self.size == 18:
            action[14:] = checked_ternary_mobile_action(action[14:])
        self.rows.append({'observation.state': list(capture.state), 'action': action,
                          'observation.source_time': relative(capture.stamp_ns),
                          'observation.camera_times': [relative(f.stamp_ns) for f in capture.images],
                          'observation.state_times': [relative(t) for t in capture.state_stamps_ns],
                          'action.sample_times': [relative(t) for t in capture.action_stamps_ns],
                          'action.sent_times': [relative(t) for t in capture.sent_times_ns],
                          'action.accepted_times': [relative(t) for t in capture.accepted_times_ns],
                          'action.arm_command_id': capture.arm_command_id,
                          'observation.camera_skew_s': capture.camera_skew_s,
                          'timestamp': index / self.config['fps'], 'frame_index': index,
                          'episode_index': self.info['total_episodes'],
                          'index': self.info['total_frames'] + index, 'task_index': self.task_index})
        if self.size == 18:
            self.rows[-1]['action.mobile_command_id'] = capture.mobile_command_id

    def encode_image(self, name, image, index):
        bgr = cv2.imdecode(np.frombuffer(image.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape != (self.height, self.width, 3):
            raise ValueError(f'{name}: JPEG dimensions differ from dataset')
        if name not in self.videos:
            container = av.open(str(self.stage / (name + '.mp4')), mode='w')
            stream = container.add_stream('libx264', rate=self.config['fps'])
            stream.width, stream.height, stream.pix_fmt = self.width, self.height, 'yuv420p'
            stream.codec_context.thread_count = 1
            stream.options = {'preset': 'ultrafast', 'crf': '23', 'tune': 'zerolatency'}
            self.videos[name] = (container, stream)
        container, stream = self.videos[name]
        # Convert once with OpenCV's ARM-optimized path. Passing BGR to x264
        # otherwise adds an implicit libswscale conversion on every camera.
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
        frame = av.VideoFrame.from_ndarray(yuv, format='yuv420p')
        frame.pts, frame.time_base = index, Fraction(1, self.config['fps'])
        for packet in stream.encode(frame):
            container.mux(packet)
        # RGB statistics over a deterministic spatial/temporal subsample.
        if index % 10 == 0:
            rgb = bgr[::max(1, self.height // 32), ::max(1, self.width // 32), ::-1]
            for channel in range(3):
                self.hist[name][channel] += np.bincount(rgb[:, :, channel].ravel(), minlength=256)

    def transcode_camera(self, name, path):
        count = 0
        with av.open(str(path), format='mjpeg') as source:
            for packet in source.demux(video=0):
                if packet.size:
                    self.encode_image(name, CameraFrame(1, bytes(packet)), count)
                    count += 1
        if count != len(self.rows):
            raise ValueError(f'{name}: staged JPEG count {count} differs from data rows {len(self.rows)}')

    def close_videos(self, *, encode=True):
        files, self.raw_files = self.raw_files, {}
        for output in files.values():
            output.close()
        if encode and self.raw_paths and not self.raw_transcoded:
            pending = [self.encode_workers.submit(self.transcode_camera, name, path)
                       for name, path in self.raw_paths.items()]
            errors = []
            for future in pending:
                try:
                    future.result()
                except Exception as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]  # All jobs finish before error cleanup can close videos.
            self.raw_transcoded = True
        videos, self.videos = self.videos, {}
        for container, stream in videos.values():
            try:
                for packet in stream.encode():
                    container.mux(packet)
            finally:
                container.close()

    def save(self, *, interruption=None, transfer_id=None, transfer_sha256=None):
        if self.committing:
            raise ValueError('Interrupted commit must be recovered before retrying')
        if self.stage is None or len(self.rows) < 2:
            raise ValueError('At least two complete samples are required')
        video_started = time.perf_counter()
        self.close_videos()
        video_encode_seconds = time.perf_counter() - video_started
        episode = self.info['total_episodes']
        chunk, file = divmod(episode, 1000)
        length, first = len(self.rows), self.info['total_frames']
        fields = []
        for key, feature in self.info['features'].items():
            if feature['dtype'] == 'video':
                continue
            dtype = pa.from_numpy_dtype(feature['dtype'])
            fields.append(pa.field(key, dtype if feature['shape'] == [1] else pa.list_(dtype, feature['shape'][0])))
        table = pa.Table.from_pylist(self.rows, schema=pa.schema(fields))
        data_path = self.root / self.info['data_path'].format(chunk_index=chunk, file_index=file)
        pq.write_table(table, self.stage / 'data.parquet', compression='snappy')
        numeric_stats = {key: scalar_stats([row[key] for row in self.rows])
                         for key, feature in self.info['features'].items() if feature['dtype'] != 'video'}
        episode_stats = {**numeric_stats, **{
            'observation.images.' + name: histogram_stats(self.hist[name], length) for name in CAMERAS}}
        metadata = {'episode_index': episode, 'tasks': [self.task], 'length': length,
                    'dataset_from_index': first, 'dataset_to_index': first + length,
                    'data/chunk_index': chunk, 'data/file_index': file,
                    'meta/episodes/chunk_index': chunk, 'meta/episodes/file_index': file}
        for name in CAMERAS:
            prefix = 'videos/observation.images.' + name
            metadata.update({prefix + '/chunk_index': chunk, prefix + '/file_index': file,
                             prefix + '/from_timestamp': 0., prefix + '/to_timestamp': length / self.config['fps']})
        for key, statistics in episode_stats.items():
            for stat, values in statistics.items():
                metadata[f'stats/{key}/{stat}'] = values
        pq.write_table(pa.Table.from_pylist([metadata]), self.stage / 'episode.parquet')
        pairs = [(self.stage / 'data.parquet', data_path),
                 (self.stage / 'episode.parquet', self.root / f'meta/episodes/chunk-{chunk:03d}/file-{file:03d}.parquet')]
        for name in CAMERAS:
            target = self.root / self.info['video_path'].format(
                video_key='observation.images.' + name, chunk_index=chunk, file_index=file)
            pairs.append((self.stage / (name + '.mp4'), target))
        if any(target.exists() for _, target in pairs):
            raise ValueError('Destination file already exists; refusing to overwrite')
        self.committing = True
        for source, target in pairs:
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
        if self.task not in self.tasks.index:
            self.tasks.loc[self.task] = self.task_index
        self.tasks['task_index'] = self.tasks['task_index'].astype('int64')
        self.tasks.to_parquet(self.stage / 'tasks.parquet')
        os.replace(self.stage / 'tasks.parquet', self.root / 'meta/tasks.parquet')
        # Numeric vectors are small: recompute exact dataset quantiles on commit.
        columns = list(numeric_stats)
        paths = sorted((self.root / 'data').glob('chunk-*/file-*.parquet'))
        stats = {}
        for key in columns:
            # Keep one numeric feature in memory instead of materializing the
            # entire historical dataset as Python dictionaries after each save.
            arrays = [np.asarray(pq.read_table(path, columns=[key])[key].to_pylist(), dtype=np.float64)
                      for path in paths]
            stats[key] = scalar_stats(np.concatenate(arrays, axis=0))
        histogram_path = self.root / 'meta/image_histograms.npz'
        if histogram_path.exists():
            with np.load(histogram_path) as previous:
                for name in CAMERAS:
                    self.hist[name] += previous[name]
        for name in CAMERAS:
            stats['observation.images.' + name] = histogram_stats(self.hist[name], first + length)
        np.savez_compressed(self.stage / 'histograms.npz', **self.hist)
        os.replace(self.stage / 'histograms.npz', histogram_path)
        json_write(self.root / 'meta/stats.json', stats)
        quality = {'interruption': interruption,
                   'video_encoding': 'JPEG staging during capture; H264 encoding on save',
                   'video_encode_seconds': video_encode_seconds,
                   'camera_time_semantics': 'host read completion, not exposure',
                   'action_semantics': ACTION_SEMANTICS,
                   'min_action_after_observation_s': min(
                       min(row['action.sample_times']) - max(*row['observation.camera_times'],
                                                              *row['observation.state_times']) for row in self.rows),
                   'max_action_after_observation_s': max(
                       max(row['action.sample_times']) - max(*row['observation.camera_times'],
                                                               *row['observation.state_times']) for row in self.rows),
                   'source_duration_s': self.rows[-1]['observation.source_time'],
                   'nominal_duration_s': length / self.config['fps'],
                   'fresh_fps': (length - 1) / max(1e-9, self.rows[-1]['observation.source_time']),
                   'max_camera_skew_s': max(r['observation.camera_skew_s'] for r in self.rows),
                   'max_sample_gap_s': max(np.diff([r['observation.source_time'] for r in self.rows])).item(),
                   'calibration_ids': list(self.identity[0]), 'control_sessions': list(self.identity[1]),
                   'image_stats': 'RGB 32x32 spatial samples every 10 frames'}
        if transfer_id is not None:
            quality.update(transfer_id=transfer_id, cache_sha256=transfer_sha256,
                           episode=episode, frames=length, path=str(self.root))
        json_write(self.root / f'meta/quality/episode-{episode:06d}.json', quality)
        self.info.update(total_episodes=episode + 1, total_frames=first + length,
                         total_tasks=len(self.tasks), splits={'train': f'0:{episode + 1}'})
        json_write(self.root / 'meta/info.json', self.info)  # final commit marker
        self.remove_stage()
        self.committing = False
        self.rows = []
        return {'episode': episode, 'frames': length, 'path': str(self.root), **quality}

    def remove_stage(self):
        if self.stage is not None:
            target = self.stage.resolve()
            if target.parent != self.root or not target.name.startswith('.pending-') or self.stage.is_symlink():
                raise ValueError('Unsafe staging path')
            shutil.rmtree(target)
            self.stage = None

    def discard(self):
        if self.committing:
            raise ValueError('部分提交发生错误，保留.pending和已落盘文件，禁止直接丢弃或覆盖')
        self.close_videos(encode=False)
        self.remove_stage()
        self.rows = []

    def close(self):
        # An interrupted episode is deliberately kept as .pending, never committed.
        try:
            self.close_videos(encode=False)
        finally:
            if self.encode_workers is not None:
                self.encode_workers.shutdown(wait=True)
                self.encode_workers = None
            if self.lock_fd is not None:
                os.close(self.lock_fd)
                self.lock_fd = None
