"""Desktop dataset receiver, independent of ROS and robot control."""

from xyc.robot.profile import robot_profile
# SPDX-License-Identifier: Apache-2.0

import http.client
import hashlib
from collections import OrderedDict
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4

from .config import checked_config
from .stream import MAX_PACKET, unpack_event
from .cache import framed_capture
from xyc.contracts import ACTION_SEMANTICS, ACTION_UNITS, STATE_UNITS
from .alignment import RECORDING_CONTRACT


class LocalDatasetReceiver:
    def __init__(self, datasets_dir, config, *, url=None,
                 width=None, height=None):
        try:
            from .writer import LeRobotV3Writer
        except ModuleNotFoundError as exc:
            raise ValueError('电脑缺少采集依赖，请安装 requirements-recording.txt：' + str(exc)) from exc
        self.writer_factory = LeRobotV3Writer
        self.datasets_dir = Path(datasets_dir).expanduser().resolve()
        self.config = checked_config(config)
        default_height, default_width = robot_profile().image_size
        self.width = default_width if width is None else width
        self.height = default_height if height is None else height
        width, height = self.width, self.height
        address = urlsplit(url or f"http://localhost:{robot_profile().ports['data']}/api/recording")
        if address.scheme != 'http' or address.hostname not in ('localhost', '127.0.0.1', '::1'):
            raise ValueError('Recording must use a local SSH-forwarded HTTP endpoint')
        self.address = (address.hostname, address.port or 80)
        self.path = address.path.rstrip('/')
        self.client = uuid4().hex
        self.writer = None
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.closing = threading.Event()
        self.closing_at = None
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.error = ''
        self.message = '等待树莓派数据通道，数据将保存在本地电脑'
        self.frames = 0
        self.saved_episodes = 0
        self.last_sequence = 0
        self.last_ack = None
        self.recent_acks = OrderedDict()
        self.epoch = None
        self.cache_id = None
        self.capture_digest = hashlib.sha256()
        self.robot_assets = None
        self.last_activity = time.monotonic()
        self.worker = self.heartbeat_worker = None

    def start(self):
        self.heartbeat_worker = threading.Thread(target=self.heartbeat, name='recording-heartbeat', daemon=True)
        self.worker = threading.Thread(target=self.run, name='local-dataset-writer', daemon=True)
        self.heartbeat_worker.start()
        self.worker.start()
        return self

    def update(self, **values):
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)

    def snapshot(self):
        with self.lock:
            return {'error': self.error, 'message': self.message, 'frames': self.frames,
                    'saved_episodes': self.saved_episodes, 'path': str(self.datasets_dir / self.config['dataset'])}

    def connection(self):
        return http.client.HTTPConnection(*self.address, timeout=3.)

    def request(self, connection, operation, body):
        data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode()
        connection.request('POST', self.path + '/' + operation, body=data,
                           headers={'Content-Type': 'application/json', 'Content-Length': str(len(data))})
        response = connection.getresponse()
        length = int(response.getheader('Content-Length', '0'))
        if not 0 < length <= MAX_PACKET:
            raise ConnectionError('Invalid recording response length')
        packet = response.read(length)
        if response.status != 200 or len(packet) != length:
            raise ConnectionError(f'电脑采集通道未就绪（HTTP {response.status}），请检查后端版本及{self.address[1]}转发')
        return packet

    def heartbeat(self):
        connection = self.connection()
        try:
            while not self.stop.is_set():
                try:
                    result = json.loads(self.request(connection, 'heartbeat', {'client': self.client}))
                    if result != {'version': 1, 'storage': 'pc'}:
                        raise ValueError('Unsupported local recording backend')
                    self.ready.set()
                except (OSError, ValueError, http.client.HTTPException):
                    self.ready.clear()
                    connection.close()
                self.stop.wait(.25)
        finally:
            connection.close()

    def info(self):
        return {key: self.writer.info[key] for key in ('total_episodes', 'total_frames')}

    def handle(self, event):
        operation, payload = event['operation'], event['payload']
        if operation == 'open':
            if checked_config(payload['config']) != self.config:
                raise ValueError('树莓派请求的采集配置与本地命令不一致')
            assets = payload.get('robot_assets')
            size = 18 if self.config['scope'] == 'mobile' else 14
            if (not isinstance(assets, dict)
                    or assets.get('recording_contract') != RECORDING_CONTRACT
                    or assets.get('action_semantics') != ACTION_SEMANTICS
                    or tuple(assets.get('state_units', ())) != STATE_UNITS[:size]
                    or tuple(assets.get('action_units', ())) != ACTION_UNITS[:size]):
                raise ValueError(
                    '树莓派采集契约不是当前ternary版本；请同步并重新编译xyc_recording，'
                    '旧后端不会再被允许生成混合数据集')
            if self.writer is not None:
                if assets != self.robot_assets:
                    raise ValueError('Robot calibration assets changed')
                return {'info': self.info(), 'path': str(self.writer.root)}
            self.writer = self.writer_factory(self.datasets_dir, self.config,
                robot_assets=assets, width=self.width, height=self.height,
                recover_transfer_id=payload.get('recover_transfer_id'))
            self.robot_assets = assets
            self.update(saved_episodes=self.writer.info['total_episodes'], message='本地数据集已打开')
            return {'info': self.info(), 'path': str(self.writer.root)}
        if self.writer is None:
            raise ValueError('本地数据集尚未打开；请重新启动一次遥操客户端')
        if operation == 'begin':
            cache_id = payload.get('cache_id')
            if cache_id is not None:
                if not self.writer.is_transfer_id(cache_id):
                    raise ValueError('Invalid episode cache identity')
                if self.writer.committing:
                    self.writer.close()
                    self.writer = self.writer_factory(
                        self.datasets_dir, {**self.config, 'resume': True},
                        robot_assets=self.robot_assets, width=self.width, height=self.height,
                        recover_transfer_id=cache_id)
                result = self.receipt(cache_id)
                if result is not None:
                    self.update(saved_episodes=self.writer.info['total_episodes'], frames=result['frames'])
                    return {**result, 'committed': True, 'info': self.info()}
                if self.writer.stage is not None:
                    if self.cache_id != cache_id:
                        raise ValueError('Another episode is still pending; refusing to discard it')
                    self.writer.discard()
            self.writer.begin(cache_id=cache_id)
            self.cache_id = cache_id
            self.capture_digest = hashlib.sha256()
            self.update(frames=0, error='', message='等待完整新样本，保存位置：本地电脑')
        elif operation == 'capture':
            self.writer.add(event['capture'])
            self.capture_digest.update(framed_capture(len(self.writer.rows), event['capture']))
            self.update(frames=len(self.writer.rows), message='正在写入本地电脑')
        elif operation == 'save':
            if payload.get('cache_id') is not None:
                if (payload['cache_id'] != self.cache_id or payload.get('frames') != len(self.writer.rows)
                        or payload.get('sha256') != self.capture_digest.hexdigest()):
                    raise ValueError('缓存传输帧数或SHA256不一致，不提交回合')
            self.update(message='正在本地编码视频并保存回合，遥操继续')
            result = self.writer.save(interruption=payload.get('interruption'), transfer_id=self.cache_id,
                                      transfer_sha256=self.capture_digest.hexdigest() if self.cache_id else None)
            if self.cache_id:
                from xyc.io import write_json
                result.update(cache_id=self.cache_id, cache_sha256=self.capture_digest.hexdigest())
                write_json(self.writer.root / 'meta/transfers' / (self.cache_id + '.json'), result)
            self.update(saved_episodes=self.writer.info['total_episodes'], frames=result['frames'],
                        message=f'本地已保存 {result["frames"]}帧，实际{result["fresh_fps"]:.2f}FPS')
            return {**result, 'info': self.info()}
        elif operation == 'discard':
            if payload.get('cache_id') and self.receipt(payload['cache_id']) is not None:
                return {}  # Already saved files are never deleted by discarding a cache.
            self.writer.discard()
            self.update(frames=0, error='', message='本地未保存回合已丢弃')
        elif operation == 'close':
            self.writer.close()
            self.writer = None
        else:
            raise ValueError('Unsupported recording operation')
        return {}

    def receipt(self, cache_id):
        if not self.writer.is_transfer_id(cache_id):
            raise ValueError('Invalid episode cache identity')
        path = self.writer.root / 'meta/transfers' / (cache_id + '.json')
        if path.is_file():
            return json.loads(path.read_text(encoding='utf-8'))
        # A commit may have completed just before the receipt file was written.
        for path in (self.writer.root / 'meta/quality').glob('episode-*.json'):
            quality = json.loads(path.read_text(encoding='utf-8'))
            if (quality.get('transfer_id') == cache_id
                    and quality.get('episode', self.writer.info['total_episodes']) < self.writer.info['total_episodes']):
                return {**quality, 'cache_id': cache_id, 'cache_sha256': quality.get('cache_sha256')}
        return None

    def run(self):
        connection = self.connection()
        network_failure_at = None
        try:
            while not self.stop.is_set():
                try:
                    packet = self.request(connection, 'exchange', {'client': self.client, 'ack': self.last_ack})
                    event = unpack_event(packet)
                    network_failure_at = None
                except (OSError, ValueError, http.client.HTTPException) as exc:
                    self.ready.clear()
                    connection.close()
                    if network_failure_at is None:
                        network_failure_at = time.monotonic()
                    if self.writer is not None and self.writer.stage is not None:
                        if time.monotonic() - network_failure_at > 1.:
                            self.update(error=str(exc))
                            break
                    elif self.closing.is_set():
                        break
                    self.update(message=str(exc))
                    self.stop.wait(.1)
                    continue
                if event['operation'] == 'idle':
                    if self.closing.is_set():
                        idle_s = time.monotonic() - self.last_activity
                        if time.monotonic() - self.closing_at > 2. and idle_s > 1.:
                            break
                    continue
                events = event['events'] if event['operation'] == 'batch' else [event]
                acknowledgments = []
                for item in events:
                    epoch, sequence = item.get('epoch'), item['sequence']
                    if self.epoch is None:
                        self.epoch = epoch
                    if epoch != self.epoch:
                        raise ValueError('Recording source restarted; refusing an ambiguous replay')
                    if sequence in self.recent_acks:
                        acknowledgments.append(self.recent_acks[sequence])
                        continue
                    if sequence <= self.last_sequence:
                        raise ValueError('Recording source sequence moved backwards')
                    self.last_activity = time.monotonic()
                    try:
                        result = self.handle(item)
                        ack = {'epoch': epoch, 'sequence': sequence, 'accepted': True, 'result': result}
                    except Exception as exc:
                        self.update(error=str(exc), message='本地采集失败，已保存回合保留；遥操继续')
                        ack = {'epoch': epoch, 'sequence': sequence, 'accepted': False, 'message': str(exc)}
                    self.last_sequence = sequence
                    self.recent_acks[sequence] = ack
                    while len(self.recent_acks) > 128:
                        self.recent_acks.popitem(last=False)
                    acknowledgments.append(ack)
                    if not ack['accepted']:
                        break
                self.last_ack = acknowledgments
        except Exception as exc:
            self.update(error=str(exc))
        finally:
            connection.close()
            if self.writer is not None:
                try:
                    if self.writer.stage is not None and not self.error:
                        if self.cache_id is not None:
                            self.update(message='传输未完成，保留电脑临时文件及树莓派缓存，恢复后重传')
                        elif len(self.writer.rows) >= 2:
                            self.writer.save(interruption={'reason': '电脑采集客户端关闭', 'gap_s': 0.})
                        else:
                            self.writer.discard()
                    elif self.writer.stage is not None and self.error and not self.writer.committing:
                        # A broken transport can still leave a valid local prefix.
                        # Disk/encoding errors keep .pending for inspection instead.
                        if self.cache_id is None and network_failure_at is not None and len(self.writer.rows) >= 2:
                            result = self.writer.save(interruption={'reason': '采集网络中断', 'gap_s': 1.})
                            self.update(saved_episodes=self.writer.info['total_episodes'], frames=result['frames'],
                                        message='网络中断，电脑已保存收到的有效前段')
                except Exception as exc:
                    self.update(error=str(exc), message='本地保存失败，保留.pending文件供检查')
                finally:
                    try:
                        self.writer.close()
                    except Exception as exc:
                        self.update(error=str(exc), message='本地关闭失败，请保留数据目录检查')
            self.stop.set()
            self.done.set()

    def close(self):
        if self.worker is None:
            return
        self.closing_at = time.monotonic()
        self.closing.set()
        # Keep receiving queued samples and the robot's disconnect/save request.
        # Video encoding is deliberately outside the robot's control connection.
        while not self.done.wait(1.):
            pass
        self.worker.join()
        self.stop.set()
        self.heartbeat_worker.join()
