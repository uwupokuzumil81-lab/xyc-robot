"""Pair read-only ROS inputs and relay complete samples to the desktop writer."""

from xyc.robot.profile import robot_profile
from xyc.assets import follower_calibration_ids
# SPDX-License-Identifier: Apache-2.0

from collections import Counter, OrderedDict
import json
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from xyc_interfaces.srv import RecordingControl

from xyc.contracts import ACTION_SEMANTICS, ACTION_UNITS, CAMERAS, STATE_UNITS, mobile_action_metadata
from xyc.recording.alignment import RECORDING_CONTRACT, CaptureBuffer
from xyc.recording.config import checked_config
from xyc.recording.stream import CaptureRelay, RecordingBackpressure, RelayServer, STREAM_PORT
from xyc.recording.cache import CachedDesktopWriter


class RecorderNode(Node):
    def __init__(self, **kwargs):
        super().__init__('recorder', **kwargs)
        self.project = Path(self.declare_parameter(
            'project_dir', str(robot_profile().robot_root)).value).expanduser().resolve()
        self.cache_dir = Path(self.declare_parameter(
            'cache_dir', str(self.project / '.runtime/recording-cache')).value).expanduser().resolve()
        port = self.declare_parameter('stream_port', STREAM_PORT).value
        self.relay = CaptureRelay()
        self.stream_server = RelayServer(('127.0.0.1', port), self.relay)
        self.stream_thread = threading.Thread(target=self.stream_server.serve_forever,
                                              name='recording-data-source', daemon=True)
        self.stream_thread.start()
        self.buffer = CaptureBuffer()
        self.commands = Queue(maxsize=8)
        self.state_lock = threading.Lock()
        self.state = {'phase': 'idle', 'frames': 0, 'saved_episodes': 0, 'message': '采集已就绪，尚未录制'}
        self.request_results = OrderedDict()
        self.stop_event = threading.Event()
        self.writer = None
        self.config = None
        self.phase = 'idle'
        self.last_capture = None
        self.interruption = None
        self.operator_at = 0.
        self.input_counts = Counter()
        self.previous_counts = Counter()
        self.rates_at = time.monotonic()
        self.subscriptions_ = [self.create_subscription(
            CompressedImage, f'/camera/{name}/image_raw/compressed',
            lambda msg, name=name: self.on_camera(name, msg), qos_profile_sensor_data) for name in CAMERAS]
        self.subscriptions_ += [self.create_subscription(
            String, f'/motors/{side}/recording_sample',
            lambda msg, side=side: self.on_arm(side, msg), qos_profile_sensor_data) for side in ('left', 'right')]
        self.subscriptions_ += [self.create_subscription(
            String, topic, self.on_command, qos_profile_sensor_data) for topic in
            ('/robot/arms/recording_command', '/robot/chassis/recording_command')]
        self.service = self.create_service(RecordingControl, '/robot/recording/control', self.on_control)
        self.lease_subscription = self.create_subscription(
            String, '/robot/recording/operator_lease', self.on_lease, 1)
        self.publisher = self.create_publisher(String, '/robot/recording/status', 1)
        self.status_timer = self.create_timer(.2, self.publish_status)
        self.worker = threading.Thread(target=self.work, name='capture-relay', daemon=True)
        self.worker.start()
        self.get_logger().info('采集就绪：树莓派按回合临时缓存，录完传到电脑校验保存，确认后清理缓存')

    def on_camera(self, name, msg):
        try:
            if 'jpeg' not in msg.format.lower() and msg.format.lower() != 'jpg':
                return
            self.buffer.add_camera(name, msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec,
                                   bytes(msg.data))
            self.input_counts['camera/' + name] += 1
        except ValueError as exc:
            self.get_logger().warning(str(exc), throttle_duration_sec=2.)

    def on_arm(self, side, msg):
        try:
            self.buffer.add_arm(side, json.loads(msg.data))
            self.input_counts['state/' + side] += 1
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self.get_logger().warning(f'录制状态帧无效：{exc}', throttle_duration_sec=2.)

    def on_lease(self, msg):
        if msg.data:
            self.operator_at = time.monotonic()

    def on_command(self, msg):
        try:
            event = json.loads(msg.data)
            if event.get('accepted') is False:
                return
            self.buffer.add_command(event)
            self.input_counts['command/' + event['kind']] += 1
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            self.get_logger().warning(f'录制新动作事件无效：{exc}', throttle_duration_sec=2.)

    def on_control(self, request, response):
        if (not request.request_id or len(request.request_id) > 160
                or request.operation not in ('start', 'save', 'discard', 'disconnect')
                or len(request.config_json.encode()) > 4096):
            response.message = 'Invalid recording-only request'
            return response
        signature = (request.operation, request.config_json)
        previous = self.request_results.get(request.request_id)
        if previous is not None:
            response.accepted = previous == signature
            response.message = '已接收过相同请求' if response.accepted else '请求ID已用于不同操作'
            return response
        try:
            self.commands.put_nowait((request.request_id, request.operation, request.config_json))
        except Full:
            response.message = '采集操作队列繁忙，请稍后重试'
            return response
        self.request_results[request.request_id] = signature
        while len(self.request_results) > 64:
            self.request_results.popitem(last=False)
        response.accepted = True
        response.message = '采集操作已排队；保存完成以采集状态为准'
        return response

    def update(self, **values):
        with self.state_lock:
            self.state.update(values)
            self.state['phase'] = self.phase

    def publish_status(self):
        now = time.monotonic()
        if now - self.rates_at >= 1.:
            rates = {key: round((count - self.previous_counts[key]) / (now - self.rates_at), 1)
                     for key, count in self.input_counts.items()}
            self.previous_counts = self.input_counts.copy()
            self.rates_at = now
            self.update(input_hz=rates)
        with self.state_lock:
            snapshot = dict(self.state)
        snapshot['storage'] = 'pc'
        snapshot['buffering'] = 'robot_episode_cache'
        snapshot['data_channel_ready'] = self.relay.alive()
        writer = self.writer
        if writer is not None and getattr(writer, 'capture_batches', 0):
            snapshot['capture_write_ms'] = round(1000 * writer.capture_seconds / writer.capture_batches, 2)
        self.publisher.publish(String(data=json.dumps(snapshot, ensure_ascii=False, allow_nan=False)))

    def assets(self, scope='mobile'):
        files = {}
        for folder in ('calibration', 'leader_calibration'):
            for side in ('left', 'right'):
                path = self.project / 'config' / folder / (side + '.json')
                if not path.is_file():
                    raise ValueError(f'缺少标定资产：{path}')
                files[f'{folder}/{side}'] = json.loads(path.read_text(encoding='utf-8'))
        size = 14 if scope == 'arms' else 18
        return {'calibrations': files,
                'recording_contract': RECORDING_CONTRACT,
                'action_semantics': ACTION_SEMANTICS,
                'state_units': list(STATE_UNITS[:size]),
                'action_units': list(ACTION_UNITS[:size]),
                'mobile_action_scale': mobile_action_metadata(),
                'coordinate_frame': 'base_link',
                'time_semantics': 'camera read completion and servo receipt; no hardware exposure sync'}

    def command(self, request_id, operation, config_json):
        if operation == 'start':
            if self.phase == 'error' and isinstance(self.writer, CachedDesktopWriter) and self.writer.stage is not None:
                self.writer.close()
                self.writer, self.config = None, None
                self.phase = 'idle'  # An explicit --resume start can recover the durable cache.
            if self.phase not in ('idle',):
                raise ValueError('请先B保存或N丢弃当前回合，再R开始下一回合')
            config = checked_config(json.loads(config_json))
            owner = request_id.rsplit(':', 1)[0]
            if self.state.get('recording_session') != owner or config != self.config:
                self.last_capture, self.interruption = None, None
                self.update(recording_session=owner, frames=0, saved_episodes=0,
                            dataset=config['dataset'], target_frames=config['episode_frames'],
                            fps=config['fps'], missing_input='', actual_fps=None,
                            message='正在打开本地数据集；尚未录制')
            if self.writer is None or config != self.config:
                if self.writer is not None:
                    self.writer.close()
                    self.writer = None
                self.writer = CachedDesktopWriter(self.relay, config, robot_assets=self.assets(config['scope']),
                    cache_dir=self.cache_dir)
                self.config = config
                self.series_start = self.writer.info['total_episodes']
            if (config['num_episodes'] and self.writer.info['total_episodes'] - self.series_start
                    >= config['num_episodes']):
                raise ValueError('本次已达到指定回合数量；如需继续请调整 --episodes')
            self.writer.begin()
            if self.writer.recovered:
                self.phase = 'saving'
                self.update(frames=len(self.writer.rows), message='发现未传完缓存，正在恢复传输到电脑')
                self.command(request_id, 'save', '{}')
                return
            self.buffer.reset(self.get_clock().now().nanoseconds, config['fps'], config['scope'])
            self.phase = 'waiting'
            self.last_capture = None
            self.interruption = None
            self.update(frames=0, target_frames=config['episode_frames'], fps=config['fps'],
                        dataset=config['dataset'], path=str(self.writer.root),
                        saved_episodes=self.writer.info['total_episodes'],
                        message='等待四路新帧、双臂上力和升降回零；不会自动移动机器人')
        elif operation in ('save', 'disconnect'):
            if (operation == 'disconnect'
                    and request_id.rsplit(':', 1)[0] != self.state.get('recording_session')):
                return  # A late old-client disconnect cannot close a new client's dataset.
            if self.writer is None or self.writer.stage is None:
                if operation == 'disconnect' and self.writer is not None:
                    self.writer.close()
                    self.writer, self.config = None, None
                return
            if self.phase == 'error' and not isinstance(self.writer, CachedDesktopWriter):
                raise ValueError('写入错误后请保留现场数据，或N丢弃尚未提交的回合')
            if len(self.writer.rows) < 2:
                self.writer.discard()
                self.phase = 'idle'
                self.update(frames=0, message='没有足够有效样本，当前空回合已丢弃')
            else:
                self.phase = 'saving'
                self.update(message='正在将树莓派临时缓存传到电脑并封装视频；完成后才能开始下一回合，遥操继续')
                result = self.writer.save(interruption=self.interruption)
                self.phase = 'idle'
                self.update(frames=result['frames'], saved_episodes=self.writer.info['total_episodes'],
                            actual_fps=round(result['fresh_fps'], 2),
                            transfer_save_seconds=round(result.get('transfer_save_seconds', 0.), 2),
                            message=(f'已保存回合 {result["episode"]}，{result["frames"]}帧，'
                                     f'实际{result["fresh_fps"]:.1f}FPS（目标{self.config["fps"]}）；R开始下一回合'))
                self.get_logger().info(f'LeRobot v3 回合已保存到电脑：{result["path"]}；{result["frames"]}帧')
            if operation == 'disconnect':
                self.writer.close()
                self.writer, self.config = None, None
        elif operation == 'discard':
            if self.writer is not None:
                self.writer.discard()
            self.phase = 'idle'
            self.update(frames=0, message='当前未提交回合已丢弃，R重新开始；已保存回合不受影响')
        self.update(last_request_id=request_id, last_operation=operation, operation_ok=True)

    def finish_interrupted(self, reason, gap_s):
        """Commit the valid prefix instead of joining samples across a long gap."""
        self.interruption = {'reason': reason, 'gap_s': round(gap_s, 3)}
        self.get_logger().warning(f'录制数据中断 {gap_s:.2f}s：{reason}；正在保存有效前段')
        enough = self.writer is not None and len(self.writer.rows) >= 2
        self.command('automatic-gap-save', 'save', '{}')
        outcome = '有效前段已保存' if enough else '有效样本不足2帧，未提交回合'
        self.update(missing_input=reason,
                    message=f'{outcome}：{reason}；检查遥操状态，恢复后R开始新回合')

    def capture(self):
        now = time.monotonic()
        if now - self.operator_at > .5:
            if self.last_capture is not None:
                self.finish_interrupted('电脑遥操连接中断', now - self.operator_at)
            else:
                self.update(message='等待电脑遥操连接', missing_input='电脑遥操连接中断')
            return
        capture = self.buffer.pick(self.get_clock().now().nanoseconds)
        if capture is None:
            self.update(message=self.buffer.reason, missing_input=self.buffer.reason)
            if self.last_capture is not None and now - self.last_capture > .5:
                self.finish_interrupted(self.buffer.reason, now - self.last_capture)
            return
        expected = tuple(follower_calibration_ids(self.writer_assets))
        if capture.calibration_ids != expected:
            raise ValueError('从臂报告的标定版本与采集器保存的标定文件不一致')
        try:
            self.writer.add(capture)
        except RecordingBackpressure as exc:
            self.finish_interrupted(str(exc), .5)
            return
        self.last_capture = time.monotonic()
        self.phase = 'recording'
        count = len(self.writer.rows)
        self.update(frames=count, missing_input='', camera_skew_ms=round(capture.camera_skew_s * 1000, 2),
                    message='正在录制：四路新图像＋18维状态/动作' if self.config['scope'] == 'mobile'
                    else '正在录制：四路新图像＋14维状态/动作')
        if count >= self.config['episode_frames']:
            self.command('automatic-save', 'save', '{}')

    def work(self):
        self.writer_assets = None
        try:
            while not self.stop_event.is_set():
                try:
                    request_id, operation, config = self.commands.get_nowait()
                except Empty:
                    pass
                else:
                    try:
                        self.command(request_id, operation, config)
                        if operation == 'start':
                            self.writer_assets = self.assets(self.config['scope'])
                    except Exception as exc:
                        if self.phase == 'saving':
                            self.phase = 'error'
                        self.update(last_request_id=request_id, last_operation=operation,
                                    operation_ok=False, message=str(exc))
                        self.get_logger().error(f'采集操作失败，遥操不受影响：{exc}')
                if self.phase in ('waiting', 'recording'):
                    try:
                        self.capture()
                    except Exception as exc:
                        self.phase = 'error'
                        self.update(message=f'采集已停止，遥操继续：{exc}')
                        self.get_logger().error(f'采集写入失败，已保存回合保留：{exc}')
                self.stop_event.wait(.002)
        finally:
            if self.writer is not None:
                # Graceful ROS shutdown saves a valid partial episode. A process
                # kill leaves .pending files and never advertises a complete one.
                try:
                    if self.phase != 'error':
                        self.command('shutdown', 'save', '{}')
                except Exception as exc:
                    self.get_logger().error(f'关闭时未完成传输，树莓派缓存保留：{exc}')
                finally:
                    self.writer.close()

    def destroy_node(self):
        self.stop_event.set()
        self.worker.join()
        self.relay.close()
        self.stream_server.shutdown()
        self.stream_server.server_close()
        self.stream_thread.join()
        return super().destroy_node()


def main(args=None):
    # ROS receipt and four video workers share this interpreter. Returning
    # from native codecs must not wait a full default 5ms GIL slice per worker.
    sys.setswitchinterval(.001)
    rclpy.init(args=args)
    node = None
    try:
        node = RecorderNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
