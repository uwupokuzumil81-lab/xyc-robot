"""Shared LeRobot v3 metadata and video readers; no PyTorch/ROS dependency."""

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from xyc.contracts import (
    ACTION, ACTION_NAMES, ACTION_UNITS, OBS_STATE, PolicySchema, checked_mobile_action,
)


@dataclass(frozen=True)
class Episode:
    index: int
    length: int
    metadata: dict


class DatasetMetadata:
    def __init__(self, root, cameras=None):
        self.root = Path(root).expanduser().resolve()
        self.info = json.loads((self.root / 'meta/info.json').read_text(encoding='utf-8'))
        if self.info.get('codebase_version') != 'v3.0':
            raise ValueError('This reader requires a committed LeRobot v3.0 dataset')
        self.features = self.info['features']
        available = [key.removeprefix('observation.images.') for key, spec in self.features.items()
                     if key.startswith('observation.images.') and spec['dtype'] == 'video']
        self.cameras = tuple(cameras or available)
        if not self.cameras or set(self.cameras) - set(available) or len(set(self.cameras)) != len(self.cameras):
            raise ValueError('Requested camera names do not match the dataset')
        assets_path = self.root / 'meta/robot.json'
        assets = json.loads(assets_path.read_text(encoding='utf-8')) if assets_path.exists() else {}
        state_dim, action_dim = (self.features[key]['shape'][0] for key in (OBS_STATE, ACTION))
        state_names = tuple(self.features[OBS_STATE].get('names') or [f'state_{i}' for i in range(state_dim)])
        action_names = tuple(self.features[ACTION].get('names') or [f'action_{i}' for i in range(action_dim)])
        state_units = tuple(assets.get('state_units', ['unknown'] * state_dim))
        action_units = tuple(assets.get('action_units', ['unknown'] * action_dim))
        if any(name.startswith(('lift.velocity_', 'base.vx_ratio', 'base.vy_ratio', 'base.wz_ratio'))
               for name in action_names) or (action_names == ACTION_NAMES and action_units != ACTION_UNITS):
            raise ValueError('旧动作格式需先离线迁移或重新采集；数据读取不会自动转换动作')
        self.assets = assets
        self.schema = PolicySchema(
            self.cameras,
            state_names, action_names, state_units, action_units, float(self.info['fps']))
        if len(self.schema.state_names) != state_dim or len(self.schema.action_names) != action_dim:
            raise ValueError('Dataset feature names and vector dimensions differ')
        episodes = []
        for path in sorted((self.root / 'meta/episodes').glob('chunk-*/file-*.parquet')):
            for row in pq.read_table(path).to_pylist():
                if row['episode_index'] < self.info['total_episodes']:
                    episodes.append(Episode(int(row['episode_index']), int(row['length']), row))
        self.episodes = {e.index: e for e in sorted(episodes, key=lambda e: e.index)}
        if (len(episodes) != len(self.episodes) or len(episodes) != self.info['total_episodes']
                or any(e.length < 1 for e in episodes)):
            raise ValueError('Missing, duplicate or empty committed episode metadata')
        self._numeric = OrderedDict()
        self._task_indices = OrderedDict()
        self._tasks = None

    def path(self, template, **values):
        path = (self.root / template.format(**values)).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError('Dataset file is missing or outside the dataset root: ' + str(path))
        return path

    def data_path(self, episode):
        row = self.episodes[episode].metadata
        return self.path(self.info['data_path'], chunk_index=row['data/chunk_index'], file_index=row['data/file_index'])

    def video(self, episode, camera):
        row = self.episodes[episode].metadata
        key = 'observation.images.' + camera
        prefix = 'videos/' + key
        return (self.path(self.info['video_path'], video_key=key,
                          chunk_index=row[prefix + '/chunk_index'], file_index=row[prefix + '/file_index']),
                float(row[prefix + '/from_timestamp']))

    def numeric(self, episode):
        if episode in self._numeric:
            self._numeric.move_to_end(episode)
            return self._numeric[episode]
        table = pq.read_table(self.data_path(episode), columns=[OBS_STATE, ACTION, 'episode_index', 'frame_index'])
        rows = [r for r in table.to_pylist() if r['episode_index'] == episode]
        rows.sort(key=lambda r: r['frame_index'])
        if len(rows) != self.episodes[episode].length or [r['frame_index'] for r in rows] != list(range(len(rows))):
            raise ValueError('Episode frame count/order differs from metadata')
        values = {key: np.asarray([r[key] for r in rows], dtype=np.float32) for key in (OBS_STATE, ACTION)}
        if (values[OBS_STATE].shape != (len(rows), len(self.schema.state_names))
                or values[ACTION].shape != (len(rows), len(self.schema.action_names))
                or not all(np.isfinite(v).all() for v in values.values())):
            raise ValueError('Invalid state/action dimensions or non-finite values')
        if self.schema.action_names == ACTION_NAMES:
            for row in values[ACTION]:
                checked_mobile_action(row[14:])
        self._numeric[episode] = values
        while len(self._numeric) > 4:
            self._numeric.popitem(last=False)
        return values

    def split(self, *, validation_fraction=.2, seed=42, train_episodes=None, validation_episodes=None):
        ids = sorted(self.episodes)
        if not ids:
            raise ValueError('No committed demonstrations found')
        if not 0 <= validation_fraction < 1:
            raise ValueError('validation_fraction must be in [0,1)')
        if validation_episodes is not None:
            val = list(validation_episodes)
        elif train_episodes is not None or len(ids) < 2 or validation_fraction == 0:
            val = []
        else:
            count = min(len(ids) - 1, max(1, round(len(ids) * validation_fraction)))
            val = sorted(np.random.default_rng(seed).permutation(ids)[:count].tolist())
        train = list(train_episodes) if train_episodes is not None else [i for i in ids if i not in val]
        if (not train or len(set(train)) != len(train) or len(set(val)) != len(val)
                or set(train) & set(val) or (set(train) | set(val)) - set(ids)):
            raise ValueError('Invalid, overlapping or empty episode split')
        return sorted(train), sorted(val)

    def task_text(self, episode, frame):
        """Resolve per-frame LeRobot task IDs, including pandas-index task tables."""
        if episode not in self._task_indices:
            path = self.data_path(episode)
            if 'task_index' in pq.read_schema(path).names:
                table = pq.read_table(path, columns=['episode_index', 'frame_index', 'task_index'])
                rows = sorted((r for r in table.to_pylist() if r['episode_index'] == episode),
                              key=lambda r: r['frame_index'])
                self._task_indices[episode] = [row['task_index'] for row in rows]
            else:
                self._task_indices[episode] = None
            while len(self._task_indices) > 4:
                self._task_indices.popitem(last=False)
        self._task_indices.move_to_end(episode)
        indices = self._task_indices[episode]
        if indices is None:
            tasks = self.episodes[episode].metadata.get('tasks', [])
            if len(tasks) != 1:
                raise ValueError('Adapter requests task text but episode has no unambiguous task')
            text = tasks[0]
        else:
            if self._tasks is None:
                tasks = pq.read_table(self.root / 'meta/tasks.parquet')
                if 'task' in tasks.column_names:
                    text_column = 'task'
                else:
                    pandas = json.loads((tasks.schema.metadata or {}).get(b'pandas', b'{}'))
                    indexes = [name for name in pandas.get('index_columns', []) if isinstance(name, str)]
                    if len(indexes) != 1:
                        raise ValueError('Task table must contain a task text column or a named string index')
                    text_column = indexes[0]
                self._tasks = dict(zip(tasks['task_index'].to_pylist(), tasks[text_column].to_pylist()))
            text = self._tasks[int(indices[frame])]
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Dataset task must be nonempty text')
        return text

    def statistics(self, episodes, spec):
        """Compute only adapter-requested statistics, using training episodes only."""
        from copy import deepcopy
        result = deepcopy(spec.constants)
        for key, methods in spec.features.items():
            if key not in (OBS_STATE, ACTION):
                raise ValueError('Unsupported numeric statistics feature: ' + key)
            values = np.concatenate([self.numeric(i)[key].astype(np.float64) for i in episodes])
            reducers = {'mean': lambda: values.mean(0), 'std': lambda: values.std(0),
                        'min': lambda: values.min(0), 'max': lambda: values.max(0),
                        'q01': lambda: np.quantile(values, .01, axis=0),
                        'q99': lambda: np.quantile(values, .99, axis=0)}
            result[key] = {name: reducers[name]().tolist() for name in methods}
            result[key]['count'] = len(values)
        return result

    def fingerprint(self, episodes):
        digest = hashlib.sha256()
        digest.update(json.dumps({'schema': self.schema.to_dict(), 'assets': self.assets}, sort_keys=True).encode())
        paths = set()
        if (self.root / 'meta/tasks.parquet').is_file():
            paths.add(self.root / 'meta/tasks.parquet')
        for index in sorted(episodes):
            e = self.episodes[index]
            digest.update(json.dumps({'index': e.index, 'length': e.length, 'metadata': e.metadata}, sort_keys=True).encode())
            paths.add(self.data_path(index))
            paths.update(self.video(index, camera)[0] for camera in self.cameras)
        for path in sorted(paths):
            digest.update(path.relative_to(self.root).as_posix().encode())
            with path.open('rb') as source:
                while data := source.read(1024 * 1024):
                    digest.update(data)
        return digest.hexdigest()


class RGBVideos:
    def __init__(self, max_frames=64):
        self.max_frames = max_frames
        self.containers, self.frames = OrderedDict(), OrderedDict()

    def read(self, path, seconds, fps):
        import av
        key = (str(path), round(seconds * fps))
        if key in self.frames:
            self.frames.move_to_end(key)
            return self.frames[key].copy()
        if str(path) not in self.containers:
            self.containers[str(path)] = av.open(str(path))
        self.containers.move_to_end(str(path))
        while len(self.containers) > 8:
            self.containers.popitem(last=False)[1].close()
        container = self.containers[str(path)]
        stream = container.streams.video[0]
        container.seek(max(0, int(seconds / stream.time_base)), stream=stream, backward=True, any_frame=False)
        image = None
        tolerance = .5 / fps + 1e-5
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            stamp = float(frame.pts * frame.time_base)
            if abs(stamp - seconds) <= tolerance:
                image = frame.to_ndarray(format='rgb24')
                break
            if stamp > seconds + tolerance:
                break
        if image is None:
            raise ValueError(f'No video frame near {seconds:.6f}s in {path}')
        self.frames[key] = image
        while len(self.frames) > self.max_frames:
            self.frames.popitem(last=False)
        return image.copy()

    def close(self):
        for container in self.containers.values():
            container.close()
        self.containers.clear()
        self.frames.clear()
