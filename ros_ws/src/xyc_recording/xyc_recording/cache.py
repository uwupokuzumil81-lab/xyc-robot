# SPDX-License-Identifier: Apache-2.0
"""Durable robot-side episode cache; transfer only after capture stops."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
from uuid import uuid4

from .stream import MAX_PACKET, pack_event, unpack_event


def framed_capture(index, capture):
    packet = pack_event(index, 'capture', capture=capture)
    return struct.pack('!I', len(packet)) + packet


def write_manifest(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as output:
        json.dump(value, output, ensure_ascii=False, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


class CachedDesktopWriter:
    """Recorder API: local sequential writes now, acknowledged desktop save later."""

    def __init__(self, relay, config, *, robot_assets, cache_dir):
        self.relay, self.config, self.assets = relay, config, robot_assets
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stage = self.output = None
        self.count = 0
        self.recovered = False
        self.digest = hashlib.sha256()
        self.manifest = None
        matches = []
        identity = lambda c: {k: v for k, v in c.items() if k not in ('resume', 'num_episodes')}
        for path in self.cache_dir.glob('episode-*/manifest.json'):
            data = json.loads(path.read_text(encoding='utf-8'))
            if identity(data['config']) == identity(config) and data['robot_assets'] == robot_assets:
                matches.append((path.parent, data))
        if len(matches) > 1:
            raise ValueError(f'有多份未传完缓存，请保留并检查：{self.cache_dir}')
        if matches:
            if not config['resume']:
                raise ValueError('发现未传完的回合缓存，请加 --resume 恢复传输；缓存不会覆盖')
            self.stage, self.manifest = matches[0]
            if self.stage.is_symlink() or self.stage.resolve().parent != self.cache_dir:
                raise ValueError('Unsafe episode cache recovery path')
            self.recovered = True
            self.seal()  # Count complete durable packets; discard only a torn final write.
        self.cache_id = self.manifest['cache_id'] if self.manifest else uuid4().hex
        result = relay.call('open', {'config': config, 'robot_assets': robot_assets,
                                     'recover_transfer_id': self.cache_id if self.recovered else None})
        self.info, self.root = result['info'], result['path']

    @property
    def rows(self):
        return range(self.count)

    def begin(self):
        if self.recovered:
            return
        if self.stage is not None:
            raise ValueError('A cached episode is already open')
        self.cache_id = uuid4().hex
        self.stage = self.cache_dir / ('episode-' + self.cache_id)
        self.stage.mkdir()
        self.manifest = {'version': 1, 'cache_id': self.cache_id, 'config': self.config,
                         'robot_assets': self.assets, 'frames': 0, 'status': 'recording'}
        write_manifest(self.stage / 'manifest.json', self.manifest)
        self.output = (self.stage / 'frames.bin').open('wb', buffering=1024 * 1024)
        self.count, self.digest = 0, hashlib.sha256()
        self.capture_batches, self.capture_seconds = 0, 0.

    def add(self, capture):
        started = time.perf_counter()
        if self.output is None:
            raise ValueError('No recording cache is open')
        if self.count % 100 == 0 and shutil.disk_usage(self.cache_dir).free < 128 * 1024 * 1024:
            raise OSError(f'树莓派缓存空间不足，现有缓存保留：{self.stage}')
        packet = framed_capture(self.count + 1, capture)
        if self.output.write(packet) != len(packet):
            raise OSError('Incomplete episode cache write')
        self.digest.update(packet)
        self.count += 1
        self.capture_batches += 1
        self.capture_seconds += time.perf_counter() - started

    def packets(self):
        with (self.stage / 'frames.bin').open('rb') as source:
            while True:
                header = source.read(4)
                if not header:
                    return
                if len(header) != 4:
                    return
                length = struct.unpack('!I', header)[0]
                if not 4 <= length <= MAX_PACKET:
                    raise ValueError(f'缓存包长度异常，请保留检查：{self.stage}')
                packet = source.read(length)
                if len(packet) != length:
                    return
                yield header + packet, unpack_event(packet)['capture']

    def seal(self):
        if self.output is not None:
            self.output.flush()
            os.fsync(self.output.fileno())
            self.output.close()
            self.output = None
        digest, count, size = hashlib.sha256(), 0, 0
        for packet, _capture in self.packets():
            digest.update(packet)
            count += 1
            size += len(packet)
        if self.manifest.get('sha256') and self.manifest['sha256'] != digest.hexdigest():
            raise ValueError(f'已封存缓存校验失败，请保留检查：{self.stage}')
        with (self.stage / 'frames.bin').open('r+b') as source:
            source.truncate(size)
        self.count, self.digest = count, digest
        self.manifest.update(frames=count, sha256=digest.hexdigest(), status='ready')
        write_manifest(self.stage / 'manifest.json', self.manifest)

    def remove_cache(self):
        target = self.stage.resolve()
        if target.parent != self.cache_dir or self.stage.is_symlink() or target.name != 'episode-' + self.cache_id:
            raise ValueError('Unsafe episode cache path')
        shutil.rmtree(target)
        self.stage, self.count = None, 0
        self.recovered = False

    def save(self, *, interruption=None):
        started = time.monotonic()
        self.seal()
        if interruption is not None:
            self.manifest['interruption'] = interruption
            write_manifest(self.stage / 'manifest.json', self.manifest)
        interruption = self.manifest.get('interruption')
        request = {'cache_id': self.cache_id, 'frames': self.count,
                   'sha256': self.digest.hexdigest(), 'retry': True}
        try:
            self.relay.call('open', {'config': self.config, 'robot_assets': self.assets,
                                    'recover_transfer_id': self.cache_id})
            result = self.relay.call('begin', request)
            if not result.get('committed'):
                for _packet, capture in self.packets():
                    self.relay.put('capture', capture=capture, wait_for_space=True)
                result = self.relay.call('save', {**request, 'interruption': interruption},
                                         timeout=max(180., self.count / self.config['fps'] * 4.))
            if (result.get('cache_id') != self.cache_id or result.get('cache_sha256') != request['sha256']
                    or result.get('frames') != self.count):
                raise ValueError('电脑保存回执与缓存不一致，保留树莓派缓存')
            self.info = result.pop('info')
            self.remove_cache()  # Only after desktop commit and end-to-end digest ACK.
            result['transfer_save_seconds'] = time.monotonic() - started
            return result
        except Exception as exc:
            raise OSError(f'传输/保存未完成，树莓派缓存保留，可B重试：{self.stage}；{exc}') from exc

    def discard(self):
        if self.output is not None:
            self.output.close()
            self.output = None
        if self.relay.alive():
            self.relay.call('discard', {'cache_id': self.cache_id})
        if self.stage is not None:
            self.remove_cache()

    def close(self):
        if self.output is not None:
            self.output.flush()
            self.output.close()
            self.output = None
        if self.relay.alive():
            try:
                self.relay.call('close')
            except (OSError, ValueError):
                pass
