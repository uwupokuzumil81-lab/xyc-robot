"""Shared image/state time pairing for recording and live model inference.

All input timestamps use the robot ROS clock; this aligns read completion,
not physical camera exposure. This module has no ROS, GPU or motor dependency.
"""

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass
import math
import threading

from .contracts import CAMERAS, ARM_NAMES

MAX_CAMERA_AGE_NS = 150_000_000
MAX_CAMERA_SKEW_NS = 33_333_334
MAX_STATE_SKEW_NS = 80_000_000
MAX_JOINT_SKEW_NS = 100_000_000


@dataclass(frozen=True)
class CameraFrame:
    stamp_ns: int
    jpeg: bytes


@dataclass(frozen=True)
class AlignedObservation:
    stamp_ns: int
    images: tuple
    state: tuple
    state_stamps_ns: tuple
    calibration_ids: tuple
    sessions: tuple
    camera_skew_s: float
    end_ns: int
    arm_batch_stamps_ns: tuple = ()
    mobile_batch_stamp_ns: int = 0


def finite_vector(values, size):
    return (isinstance(values, (tuple, list)) and len(values) == size
            and all(type(v) in (int, float) and math.isfinite(v) for v in values))


class ObservationBuffer:
    """Bounded nearest-time pairing, with explicit consumption of new frames."""

    def __init__(self):
        self.lock = threading.RLock()
        self.cameras = {name: deque(maxlen=8) for name in CAMERAS}
        self.arms = {side: deque(maxlen=128) for side in ('left', 'right')}
        self.reset(0, 30, 'mobile')

    def reset(self, started_ns, fps, scope):
        if type(fps) is not int or not 1 <= fps <= 30 or scope not in ('mobile', 'arms'):
            raise ValueError('fps must be 1..30; scope must be mobile or arms')
        with self.lock:
            self.cursors = dict.fromkeys(CAMERAS, started_ns)
            self.period_ns = round(1e9 / fps)
            self.fps, self.next_ns, self.scope = fps, started_ns, scope
            self.identity = None
            self.reason = '等待四路新图像和有效控制状态'

    def add_camera(self, name, stamp_ns, jpeg):
        if name not in CAMERAS or type(stamp_ns) is not int or stamp_ns <= 0:
            raise ValueError('Invalid camera name or timestamp')
        if not isinstance(jpeg, bytes) or not 4 <= len(jpeg) <= 2 * 1024 * 1024:
            raise ValueError('Invalid or oversized JPEG')
        with self.lock:
            queue = self.cameras[name]
            if not queue or stamp_ns > queue[-1].stamp_ns:
                queue.append(CameraFrame(stamp_ns, jpeg))

    def add_arm(self, side, sample):
        if side not in self.arms or type(sample.get('stamp_ns')) is not int:
            raise ValueError('Invalid arm sample')
        with self.lock:
            queue = self.arms[side]
            if not queue or sample['stamp_ns'] > queue[-1]['stamp_ns']:
                queue.append(sample)

    def pick(self, now_ns):
        with self.lock:
            observation = self.match(now_ns)
            if observation is not None:
                self.consume(observation)
            return observation

    def match(self, now_ns):
        """Find a complete observation; caller holds lock and decides when to consume."""
        queues = {name: [f for f in queue if
                        self.cursors[name] < f.stamp_ns <= now_ns
                        and now_ns - f.stamp_ns <= MAX_CAMERA_AGE_NS]
                  for name, queue in self.cameras.items()}
        if not all(queues.values()):
            self.reason = '等待四路新图像，旧图像不会重复写入'
            return None
        chosen = None
        for anchor in queues[CAMERAS[0]]:
            if self.fps < 30 and anchor.stamp_ns + 1_000_000 < self.next_ns:
                continue
            # A camera whose next frame has not arrived yet may be closer than
            # its cached older frame. Wait at most one source period rather
            # than greedily consuming mismatched camera generations.
            if any(queue[-1].stamp_ns < anchor.stamp_ns for queue in queues.values()):
                break
            frames = tuple(min(queues[name], key=lambda f: abs(f.stamp_ns - anchor.stamp_ns))
                           for name in CAMERAS)
            stamps = [f.stamp_ns for f in frames]
            if max(stamps) - min(stamps) <= MAX_CAMERA_SKEW_NS:
                chosen = frames
                break
        if chosen is None:
            self.reason = '等待采样时刻或四相机时间差小于33.3ms的图像组'
            return None
        stamps = sorted(f.stamp_ns for f in chosen)
        anchor_ns = (stamps[1] + stamps[2]) // 2
        states, state_times, ids, sessions = [], [], [], []
        batch_times = []
        observation_end = stamps[-1]
        for side in ('left', 'right'):
            history = self.arms[side]
            if not history:
                self.reason = f'等待{side}从臂状态'
                return None
            index = bisect_left(history, anchor_ns, key=lambda s: s['stamp_ns'])
            state = min((history[max(0, index - 1)], history[min(index, len(history) - 1)]),
                        key=lambda s: abs(s['stamp_ns'] - anchor_ns))
            if abs(state['stamp_ns'] - anchor_ns) > MAX_STATE_SKEW_NS:
                self.reason = f'{side}状态时间未对齐'
                return None
            if (not state.get('valid') or not finite_vector(state.get('state'), 7)
                    or state.get('names') != list(ARM_NAMES[:7] if side == 'left' else ARM_NAMES[7:])):
                self.reason = f'{side}尚未全部上力或反馈无效：{state.get("reason", "等待有效控制状态")}'
                return None
            sample_times = state.get('state_stamps_ns', [])
            if (len(sample_times) != 7 or any(type(t) is not int or t <= 0
                    or abs(t - anchor_ns) > MAX_JOINT_SKEW_NS for t in sample_times)):
                self.reason = f'{side}关节反馈时间不匹配'
                return None
            states.extend(state['state'])
            state_times.extend(sample_times)
            observation_end = max(observation_end, state['stamp_ns'], *sample_times)
            ids.append(state.get('calibration_id', ''))
            sessions.append(state.get('session', ''))
            batch_times.append(state['stamp_ns'])
            if side == 'left':
                mobile_state = state.get('mobile', {})
                mobile_state_time = state['stamp_ns']
        if any(not isinstance(i, str) or len(i) != 64 for i in ids) or not all(sessions):
            raise ValueError('缺少标定版本或控制会话')
        if self.scope == 'mobile':
            if not mobile_state.get('valid'):
                self.reason = '整机录制等待底盘使能、升降回零和4轴有效反馈'
                if mobile_state.get('reason'):
                    self.reason += '：' + str(mobile_state['reason'])
                return None
            if not finite_vector(mobile_state.get('state'), 4):
                self.reason = '底盘或升降数据不完整'
                return None
            states.extend(mobile_state['state'])
            state_times.extend([mobile_state_time] * 4)
        return AlignedObservation(anchor_ns, chosen, tuple(states), tuple(state_times),
                                  tuple(ids), tuple(sessions), (stamps[-1] - stamps[0]) / 1e9,
                                  observation_end, tuple(batch_times),
                                  mobile_state_time if self.scope == 'mobile' else 0)

    def consume(self, observation):
        identity = (observation.calibration_ids, observation.sessions)
        if self.identity is not None and identity != self.identity:
            raise ValueError('观测控制会话/标定改变，请结束当前运行')
        self.identity = identity
        self.cursors.update({name: frame.stamp_ns for name, frame in zip(CAMERAS, observation.images)})
        self.next_ns = max(self.next_ns + self.period_ns, observation.images[0].stamp_ns)
        self.reason = ''
