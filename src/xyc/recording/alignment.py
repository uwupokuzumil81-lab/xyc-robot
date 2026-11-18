# SPDX-License-Identifier: Apache-2.0
"""Bounded read-completion-time pairing; this does not synchronize exposure."""

from collections import deque
from dataclasses import dataclass

from xyc.observation import ObservationBuffer, finite_vector
from xyc.contracts import checked_mobile_action


RECORDING_CONTRACT = 'xyc.recording.v5.ternary'


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


class CaptureBuffer(ObservationBuffer):
    def __init__(self):
        self.commands = {kind: deque(maxlen=256) for kind in ('arms', 'mobile')}
        super().__init__()

    def reset(self, started_ns, fps, scope):
        super().reset(started_ns, fps, scope)
        self.command_cursors = dict.fromkeys(self.commands, 0)

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
        if kind == 'mobile':
            event = {**event, 'targets': checked_mobile_action(event['targets'])}
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
        observation = self.match(now_ns)
        if observation is None:
            return None
        chosen, anchor_ns = observation.images, observation.stamp_ns
        states, state_times = observation.state, observation.state_stamps_ns
        ids, sessions = list(observation.calibration_ids), list(observation.sessions)
        observation_end = observation.end_ns
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
            actions.extend(mobile_command['targets'])
            action_times.extend([mobile_command['generated_ns']] * 4)
            sent_times.extend([mobile_command['sent_ns']] * 4)
            accepted_times.extend([mobile_command['accepted_ns']] * 4)
        self.consume(observation)
        self.command_cursors['arms'] = command['sequence']
        if mobile_command is not None:
            self.command_cursors['mobile'] = mobile_command['sequence']
        for kind, cursor in self.command_cursors.items():
            queue = self.commands[kind]
            while queue and queue[0]['sequence'] <= cursor:
                queue.popleft()
        return Capture(anchor_ns, chosen, tuple(states), tuple(actions), tuple(state_times),
                       tuple(action_times), tuple(ids), tuple(sessions), observation.camera_skew_s,
                       tuple(sent_times), tuple(accepted_times), command['sequence'],
                       mobile_command['sequence'] if mobile_command is not None else 0)
