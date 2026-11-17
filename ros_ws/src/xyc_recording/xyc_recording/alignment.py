# SPDX-License-Identifier: Apache-2.0
"""Bounded read-completion-time pairing; this does not synchronize exposure."""

from collections import deque
from bisect import bisect_left
from dataclasses import dataclass
import math
import threading


CAMERAS = ('forward', 'chest', 'wrist_left', 'wrist_right')
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex',
          'wrist_yaw', 'wrist_roll', 'gripper')
ARM_NAMES = tuple(f'arm_{side}_{joint}' for side in ('left', 'right') for joint in JOINTS)
STATE_NAMES = ARM_NAMES + ('lift.height_m', 'base.vx_m_s', 'base.vy_m_s', 'base.wz_rad_s')
ACTION_NAMES = ARM_NAMES + ('lift.command', 'base.vx_command', 'base.vy_command', 'base.wz_command')
ACTION_SEMANTICS = 'observation_then_new_ternary_command_v4'
MOBILE_ACTION_SCALE = (0.015, 0.15, 0.15, 0.60)
STATE_UNITS = ('normalized',) * 14 + ('m', 'm/s', 'm/s', 'rad/s')
ACTION_UNITS = ('normalized',) * 14 + ('ternary',) * 4
RECORDING_CONTRACT = 'xyc.recording.v5.ternary'


def normalize_mobile_action(values):
    """Convert accepted fixed physical velocities to -1/0/+1 labels."""
    if not finite_vector(values, 4):
        raise ValueError('Mobile command needs four physical velocities')
    commands = []
    for value, scale in zip(values, MOBILE_ACTION_SCALE):
        candidates = (-scale, 0., scale)
        selected = min(candidates, key=lambda item: abs(item - value))
        if abs(value - selected) > max(1e-6, scale * 1e-4):
            raise ValueError('Accepted mobile command is not a fixed -1/0/+1 speed')
        commands.append(0. if selected == 0 else 1. if selected > 0 else -1.)
    return commands


def checked_ternary_mobile_action(values):
    """Accept only the canonical labels persisted in a new dataset."""
    if not finite_vector(values, 4):
        raise ValueError('Mobile action needs four finite commands')
    commands = []
    for value in values:
        selected = min((-1., 0., 1.), key=lambda item: abs(item - value))
        if abs(value - selected) > 1e-6:
            raise ValueError('Dataset mobile actions must be exactly -1, 0 or +1')
        commands.append(selected)
    return commands


@dataclass(frozen=True)
class CameraFrame:
    stamp_ns: int
    jpeg: bytes


@dataclass(frozen=True)
class Capture:
    stamp_ns: int
    images: tuple
    state: tuple
    action: tuple
    state_stamps_ns: tuple
    action_stamps_ns: tuple
    calibration_ids: tuple
    sessions: tuple
    camera_skew_s: float
    sent_times_ns: tuple = ()
    accepted_times_ns: tuple = ()
    arm_command_id: int = 0
    mobile_command_id: int = 0


def finite_vector(values, size):
    return (isinstance(values, (tuple, list)) and len(values) == size
            and all(type(v) in (int, float) and math.isfinite(v) for v in values))


class CaptureBuffer:
    def __init__(self):
        self.lock = threading.RLock()
        self.cameras = {name: deque(maxlen=8) for name in CAMERAS}
        self.arms = {side: deque(maxlen=128) for side in ('left', 'right')}
        self.commands = {kind: deque(maxlen=256) for kind in ('arms', 'mobile')}
        self.reset(0, 30, 'mobile')

    def reset(self, started_ns, fps, scope):
        if type(fps) is not int or not 1 <= fps <= 30 or scope not in ('mobile', 'arms'):
            raise ValueError('fps must be 1..30; scope must be mobile or arms')
        with self.lock:
            self.cursors = dict.fromkeys(CAMERAS, started_ns)
            self.period_ns = round(1e9 / fps)
            self.fps = fps
            self.next_ns = started_ns
            self.scope = scope
            self.identity = None
            self.command_cursors = dict.fromkeys(self.commands, 0)
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
            return self._pick(now_ns)

    def add_command(self, event):
        kind = event.get('kind')
        if kind not in self.commands or not finite_vector(event.get('targets'), 14 if kind == 'arms' else 4):
            raise ValueError('Invalid new command event')
        if type(event.get('sequence')) is not int or not 0 < event['sequence'] < 2**63:
            raise ValueError('Command ID must be a positive int64')
        generated, sent, accepted = (event.get(key) for key in ('generated_ns', 'sent_ns', 'accepted_ns'))
        if (any(type(t) is not int for t in (generated, sent, accepted))
                or not 0 < generated <= sent <= accepted):
            raise ValueError('Command must have ordered generation/send/acceptance times')
        with self.lock:
            queue = self.commands[kind]
            if not queue or event['sequence'] > queue[-1]['sequence']:
                queue.append(event)

    def next_command(self, kind, observation_end, now_ns):
        return next((event for event in self.commands[kind]
                     if event['sequence'] > self.command_cursors[kind]
                     and observation_end <= event['generated_ns'] <= observation_end + 100_000_000
                     and event['accepted_ns'] <= now_ns), None)

    def _pick(self, now_ns):
        queues = {name: [f for f in queue if
                        self.cursors[name] < f.stamp_ns <= now_ns
                        and now_ns - f.stamp_ns <= 150_000_000]
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
            if max(stamps) - min(stamps) <= 33_333_334:
                chosen = frames
                break
        if chosen is None:
            self.reason = '等待采样时刻或四相机时间差小于33.3ms的图像组'
            return None
        stamps = sorted(f.stamp_ns for f in chosen)
        anchor_ns = (stamps[1] + stamps[2]) // 2
        states, state_times, ids, sessions = [], [], [], []
        observation_end = stamps[-1]
        for side in ('left', 'right'):
            history = self.arms[side]
            if not history:
                self.reason = f'等待{side}从臂状态'
                return None
            index = bisect_left(history, anchor_ns, key=lambda s: s['stamp_ns'])
            state = min((history[max(0, index - 1)], history[min(index, len(history) - 1)]),
                        key=lambda s: abs(s['stamp_ns'] - anchor_ns))
            if abs(state['stamp_ns'] - anchor_ns) > 80_000_000:
                self.reason = f'{side}状态时间未对齐'
                return None
            if (not state.get('valid') or not finite_vector(state.get('state'), 7)
                    or state.get('names') != list(ARM_NAMES[:7] if side == 'left' else ARM_NAMES[7:])):
                self.reason = f'{side}尚未全部上力或反馈无效：{state.get("reason", "等待有效控制状态")}'
                return None
            sample_times = state.get('state_stamps_ns', [])
            if (len(sample_times) != 7 or any(type(t) is not int or t <= 0
                    or abs(t - anchor_ns) > 100_000_000 for t in sample_times)):
                self.reason = f'{side}关节反馈时间不匹配'
                return None
            states.extend(state['state'])
            state_times.extend(sample_times)
            observation_end = max(observation_end, state['stamp_ns'], *sample_times)
            ids.append(state.get('calibration_id', ''))
            sessions.append(state.get('session', ''))
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
        command = self.next_command('arms', observation_end, now_ns)
        if command is None:
            self.reason = '等待观测之后新生成、已发送且被双臂接受的动作'
            return None
        if command.get('calibration_ids') != ids or command.get('sessions') != sessions:
            raise ValueError('新动作的控制会话/标定与观测不一致')
        actions = list(command['targets'])
        action_times = [command['generated_ns']] * 14
        sent_times = [command['sent_ns']] * 14
        accepted_times = [command['accepted_ns']] * 14
        mobile_command = None
        if self.scope == 'mobile':
            mobile_command = self.next_command('mobile', observation_end, now_ns)
            if mobile_command is None:
                self.reason = '等待观测之后新发送、已被驱动接受的底盘/升降动作'
                return None
            if mobile_command.get('session') != sessions[0]:
                raise ValueError('底盘/升降新动作的控制会话与观测不一致')
            actions.extend(normalize_mobile_action(mobile_command['targets']))
            action_times.extend([mobile_command['generated_ns']] * 4)
            sent_times.extend([mobile_command['sent_ns']] * 4)
            accepted_times.extend([mobile_command['accepted_ns']] * 4)
        identity = (tuple(ids), tuple(sessions))
        if self.identity is not None and identity != self.identity:
            raise ValueError('录制期间从臂会话/标定改变，请丢弃或结束当前回合')
        self.identity = identity
        self.command_cursors['arms'] = command['sequence']
        if mobile_command is not None:
            self.command_cursors['mobile'] = mobile_command['sequence']
        for kind, cursor in self.command_cursors.items():
            queue = self.commands[kind]
            while queue and queue[0]['sequence'] <= cursor:
                queue.popleft()
        self.cursors.update({name: frame.stamp_ns for name, frame in zip(CAMERAS, chosen)})
        # Preserve phase across small 29.5-Hz jitter; do not accidentally halve FPS.
        self.next_ns = max(self.next_ns + self.period_ns, chosen[0].stamp_ns)
        self.reason = ''
        return Capture(anchor_ns, chosen, tuple(states), tuple(actions), tuple(state_times),
                       tuple(action_times), tuple(ids), tuple(sessions), (stamps[-1] - stamps[0]) / 1e9,
                       tuple(sent_times), tuple(accepted_times), command['sequence'],
                       mobile_command['sequence'] if mobile_command is not None else 0)
