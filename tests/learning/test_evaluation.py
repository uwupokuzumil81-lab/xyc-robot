"""Manual preparation, pause/resume and common execution use fake hardware."""

import json
import threading

import pytest
import torch

from test_platform_interfaces import (
    FakeCameras, FakeSink, PortableAdapter, live_fixture, schema,
)
from xyc.learning import deploy as deployment
from xyc.contracts import OBS_STATE, robot_action
from xyc.runtime.evaluation import EvaluationConsole, TerminalKeys
from xyc.runtime.policy_executor import PolicyExecutor
from xyc.runtime.policy_runner import PolicyRunner
from xyc.runtime.strategies.base import ChunkExecutionSpec, RTCChunk
from xyc.teleop.input import LeaderFrame
from xyc.teleop.mapping import AlignedFollow


class Keys:
    def __init__(self, rows):
        self.rows = list(rows)
        self.opened = self.closed = False

    def open(self): self.opened = True
    def close(self): self.closed = True
    def read(self): return self.rows.pop(0) if self.rows else ['q']
    def is_held(self, key): return None


class ManualSink(FakeSink):
    def __init__(self):
        super().__init__(FakeCameras(), ['a' * 64, 'b' * 64])
        self.arms, self.height = [0.] * 14, .5
        self.homing = ''
        self.starts = self.homes = 0

    def status(self):
        result = super().status()
        result['state'] = [*self.arms, self.height, 0., 0., 0.]
        result['chassis']['homing'] = self.homing
        return result

    def start(self):
        self.starts += 1
        return super().start()

    def home_lift(self):
        self.homes += 1
        self.homing = 'seeking'
        return self.status()

    def heartbeat(self):
        if self.homing:
            self.homing, self.height = '', 0.
        return self.status()

    def dispatch(self, command, **kwargs):
        self.arms = list(command['arm_position_normalized'])
        self.height += command['mobile_command']['lift'] * .015 * .05
        return super().dispatch(command, **kwargs)


def console(rows):
    sink = ManualSink()
    adapter = PortableAdapter(schema())
    executor = PolicyExecutor(PolicyRunner(adapter), sink)
    keys = Keys(rows)
    ui = EvaluationConsole(schema(), sink.ids, keys=keys, sleep=lambda _: None)
    ui.show = lambda *args, **kwargs: None
    return ui, executor, sink, adapter


def test_manual_setup_does_not_predict_and_preserves_all_manual_dimensions():
    ui, executor, sink, adapter = console([['i', 'w'], ['p']])
    sink.start()
    assert ui.prepare(executor)
    assert adapter.calls == 0
    assert sink.starts == 1 and sink.homes == 0
    assert sink.sent[0]['arm_position_normalized'] == [0.] * 14
    assert sink.sent[0]['mobile_command'] == {'lift': 1., 'x': 1., 'y': 0., 'yaw': 0.}
    assert sink.sent[-1]['mobile_command'] == {'lift': 0., 'x': 0., 'y': 0., 'yaw': 0.}
    assert sink.height > .5 and sink.arms == [0.] * 14
    assert ui.rounds == 1


def test_home_once_then_raise_and_start_without_reinitializing_reference():
    ui, executor, sink, _ = console([['p'], ['h'], ['i'], ['p']])
    sink.height = None
    sink.start()
    assert ui.prepare(executor)
    assert sink.homes == 1 and sink.starts == 1 and sink.height > 0.


def test_pause_freezes_arms_and_base_and_next_start_discards_old_chunk():
    ui, executor, sink, adapter = console([['p'], ['y'], ['p']])
    assert ui.prepare(executor)
    executor.step({OBS_STATE: torch.zeros(1, 18)})
    assert ui.model_keys(executor, 1) == 'prepare'
    assert ui.successes == 1
    height = sink.height
    assert ui.prepare(executor)
    executor.runner.reset()
    executor.step({OBS_STATE: torch.zeros(1, 18)})
    assert adapter.calls == 2 and sink.height == height
    assert not sink.stopped and sink.homes == 0 and ui.rounds == 2


@pytest.mark.parametrize('start_key', ['p', 'r'])
def test_quit_takes_priority_over_a_same_batch_start_or_resume(start_key):
    ui, executor, sink, adapter = console([[start_key, 'q']])
    ui.model_start_key = start_key
    sink.start()
    assert ui.prepare(executor) is False
    assert adapter.calls == 0 and ui.rounds == 0 and not sink.sent


def test_quit_takes_priority_over_a_same_batch_pause():
    ui, executor, sink, _ = console([['p', 'q']])
    assert ui.model_keys(executor, 1, display=False) == 'quit'
    assert not sink.sent
    executor.close()
    assert sink.stopped


def test_manual_quit_never_runs_a_model():
    ui, executor, sink, adapter = console([['q']])
    assert ui.prepare(executor) is False
    assert adapter.calls == 0 and sink.sent == []


class Arms:
    notice = '双主臂反馈正常'
    active = False
    calls = 0

    def __init__(self, frames):
        self.frames = iter(frames)

    def resume(self): self.active = True
    def pause(self): self.active = False

    def fresh_snapshot(self):
        assert self.active
        self.calls += 1
        return next(self.frames)


def leader_frame(sample, *, shoulder=0., gripper=0., wrist_roll=0., reused=False):
    left, right = [0.] * 7, [0.] * 7
    left[0], left[6], right[5] = shoulder, gripper, wrist_roll
    return LeaderFrame(sample, 0, {'left': left, 'right': right}, {}, reused_positions=reused)


def test_teleop_preparation_tracks_aligned_leaders_then_pauses_them_for_model_control():
    ui, executor, sink, adapter = console([['i', 'w'], [], [], ['p']])
    source = Arms([leader_frame(i, wrist_roll=2.) for i in range(1, 4)])
    ui.arm_input = source
    sink.start()
    assert ui.prepare(executor)
    assert adapter.calls == 0 and sink.arms[12] == 2.
    assert sink.sent[0]['mobile_command']['lift'] == 1.
    assert source.active is False
    sampled = source.calls
    executor.step({OBS_STATE: torch.zeros(1, 18)})
    assert source.calls == sampled and sink.starts == 1


def test_takeover_holds_after_pause_then_tracks_and_hands_control_back():
    ui, executor, sink, adapter = console([['p'], ['i', 'w', 'h'], [], [], [], [], [], ['r']])
    sink.arms[0], sink.arms[6] = 60., 40.
    ui.arm_input = Arms([
        leader_frame(1, shoulder=10., gripper=40.),
        leader_frame(2, shoulder=61., gripper=40.),
        leader_frame(2, shoulder=61., gripper=40.),
        leader_frame(3, shoulder=61., gripper=40.),
        leader_frame(4, shoulder=61., gripper=40.),
        leader_frame(5, shoulder=65., gripper=42.),
    ])
    assert ui.model_keys(executor, 1) == 'prepare'
    assert ui.prepare(executor)
    assert [command['arm_position_normalized'][0] for command in sink.sent] == [60.] * 5 + [61., 65., 65.]
    assert all(all(value == 0. for value in command['mobile_command'].values()) for command in sink.sent)
    assert sink.arms[6] == 42. and sink.homes == 0 and sink.height == .5
    assert adapter.calls == 0 and not ui.arm_input.active


def test_takeover_can_resume_model_without_matching_leaders():
    ui, executor, sink, adapter = console([['p'], ['p'], ['p'], ['r']])
    sink.arms[0] = 60.
    ui.arm_input = Arms([leader_frame(1, shoulder=10.), leader_frame(2, shoulder=10.)])
    assert ui.model_keys(executor, 1) == 'prepare'
    assert ui.prepare(executor)
    assert all(command['arm_position_normalized'][0] == 60. for command in sink.sent)
    assert not ui.follow.engaged and not ui.arm_input.active
    assert ui.arm_input.calls == 2 and not ui.keys.rows


def test_alignment_restarts_on_bad_sample_and_checks_gripper_separately():
    follow = AlignedFollow()
    follower = [0.] * 14
    assert follow.update(leader_frame(1), follower) is None
    assert follow.update(leader_frame(2), follower) is None
    assert follow.update(leader_frame(3, reused=True), follower) is None and follow.matches == 0
    assert follow.update(leader_frame(4, gripper=1.5), follower) is None and follow.matches == 0
    for sample in (5, 6):
        assert follow.update(leader_frame(sample), follower) is None
    assert follow.update(leader_frame(7), follower) == follower and follow.engaged
    assert follow.update(None, follower) is None and not follow.engaged
    assert follow.update(leader_frame(8, shoulder=4.), follower) is None and follow.matches == 0


def test_terminal_arrows_cannot_become_base_commands_across_reads():
    keys = TerminalKeys()
    assert keys.decode('W\x1b[A\x1b[D') == ['w']
    assert keys.decode('\x1b[') == []
    assert keys.decode('1;5AP') == ['p']
    assert keys.decode('\x1bw') == []


def test_warmup_does_not_send_commands_or_reuse_its_prediction():
    _, executor, sink, adapter = console([])
    batch = {OBS_STATE: torch.zeros(1, 18)}
    executor.runner.warmup(batch)
    assert adapter.calls == 1 and sink.sent == []
    executor.step(batch)
    assert adapter.calls == 2 and len(sink.sent) == 1


def test_shared_execution_bounds_full_calibrated_domain_without_step_shortening():
    ui, executor, sink, adapter = console([])
    action = [0.] * 18
    action[0], action[2], action[6] = -100.2, 100.39, 101.
    action[7] = 90.
    executor.execute_action(action, record=False)
    assert sink.sent[-1]['arm_position_normalized'][:7] == [-100., 0., 100., 0., 0., 0., 100.]
    assert sink.sent[-1]['arm_position_normalized'][7] == 90.
    with pytest.raises(ValueError, match='finite|contract'):
        executor.execute_action([float('nan')] * 18, record=False)
    with pytest.raises(ValueError, match='outside'):
        robot_action(action, schema())  # Strict recorded-data validation remains strict.


@pytest.mark.parametrize('strategy', ('sync_chunk', 'rtc'))
def test_deploy_waits_for_operator_before_predicting_and_keeps_same_sink(monkeypatch, tmp_path, strategy):
    adapter, _, _, sink = live_fixture(monkeypatch)
    transitions = []
    predictions_per_round = 2 if strategy == 'sync_chunk' else 1
    if strategy == 'rtc':
        adapter.chunk_execution_spec = lambda: ChunkExecutionSpec(4, 12)

        def predict_rtc(batch, *, previous, **kwargs):
            assert previous is None
            adapter.calls += 1
            actions = torch.zeros(1, 12, 18)
            return RTCChunk(actions, actions)

        adapter.predict_rtc_chunk = predict_rtc

    class UI:
        rounds = 0
        successes = failures = 0
        def __init__(self, *args, **kwargs): pass
        def open(self): transitions.append('open')
        def close(self): transitions.append('close')
        def prepare(self, executor):
            assert executor.command_sink is sink
            assert adapter.calls == self.rounds * predictions_per_round
            assert executor.runner.last_prediction is None
            self.rounds += 1
            transitions.append('prepare')
            return True
        def model_keys(self, executor, steps, *, display=True):
            return 'prepare' if steps == 1 and self.rounds == 1 else 'model'

        def read_model_key(self):
            return None
    monkeypatch.setattr(deployment, 'EvaluationConsole', UI)
    result = deployment.deploy('test', device='cpu', steps=2, strategy=strategy, record_dir=tmp_path / 'trace')
    assert transitions == ['open', 'prepare', 'prepare', 'close']
    assert sink.started and sink.stopped and adapter.calls == 2 * predictions_per_round
    assert result['evaluations']['started'] == 2
    assert json.loads((tmp_path / 'trace/meta.json').read_text())['status'] == 'completed'


@pytest.mark.parametrize('strategy', ('sync_chunk', 'rtc'))
@pytest.mark.parametrize('key', ('p', 'q', ' '))
def test_key_buffered_after_tick_poll_blocks_model_dispatch(monkeypatch, key, strategy):
    adapter, _, _, sink = live_fixture(monkeypatch)
    pending = ['p']
    keys = Keys([])
    keys.read = lambda: [pending.pop(0)] if pending else []
    predict = adapter.predict_chunk

    def predict_with_key(batch):
        chunk = predict(batch)
        if adapter.calls == 2:  # The first call is warmup.
            pending.append(key)
            chunk[:, :, 0] = 25.
            chunk[:, :, 15] = 1.
        return chunk

    def predict_rtc(batch, **kwargs):
        chunk = predict(batch)
        chunk[:, :, 0] = 25.
        chunk[:, :, 15] = 1.
        return RTCChunk(chunk, chunk)

    class UI(EvaluationConsole):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, keys=keys, **kwargs)

        def prepare(self, executor):
            return not self.rounds and super().prepare(executor)

        def show(self, *args, **kwargs):
            pass

        def model_keys(self, executor, steps, *, display=True):
            choice = super().model_keys(executor, steps, display=display)
            if display and strategy == 'rtc' and executor.runner.strategy.pending is not None:
                executor.runner.strategy.pending.future.result(timeout=3)
                pending.append(key)
            return choice

    if strategy == 'rtc':
        adapter.predict_rtc_chunk = predict_rtc
    else:
        monkeypatch.setattr(adapter, 'predict_chunk', predict_with_key)
    monkeypatch.setattr(deployment, 'EvaluationConsole', UI)
    result = deployment.deploy('test', device='cpu', record=False, steps=2, strategy=strategy)
    assert result['frames'] == 0 and adapter.calls == (1 if strategy == 'rtc' else 2)
    assert sink.stopped and not sink.faulted and len(sink.sent) == (1 if key == 'q' else 2)
    assert all(not any(action['arm_position_normalized']) and not any(action['mobile_command'].values())
               for action in sink.sent)


@pytest.mark.parametrize(('key', 'transition', 'operations'), [
    ('p', 'prepare', ['prepare', 'start', 'hold', 'stop']),
    ('q', 'quit', ['prepare', 'start', 'stop']),
])
def test_model_key_while_waiting_for_tcp_lock_drops_the_target(monkeypatch, key, transition, operations):
    from test_motion_replay import FakeRobot, deadline_server
    from xyc.runtime import live_robot
    from xyc.transport.motion import MotionClient

    ids = ['a' * 64, 'b' * 64]
    robot = FakeRobot(ids)
    adapter = PortableAdapter(schema())
    chunk = torch.zeros(1, 4, 18)
    chunk[:, :, :14] = 25.
    chunk[:, :, 15] = 1.
    adapter.predict_chunk = lambda _: chunk
    keys = Keys([[]])
    ui = EvaluationConsole(schema(), ids, keys=keys)
    result, pending_key = [], []
    entering_tcp = threading.Event()

    def before_send():
        pending_key.append(ui.read_model_key())
        return pending_key[-1] is None

    with deadline_server(robot) as (port, messages, errors, _):
        monkeypatch.setattr(live_robot, 'MotionClient',
                            lambda host, port: MotionClient(host, port, keepalive_period=None))
        sink = live_robot.RobotControlSink(port=port)
        policy = PolicyExecutor(PolicyRunner(adapter), sink)
        try:
            sink.prepare(schema(), ids)
            sink.start()
            assert ui.read_model_key() is None
            call = sink.client.call

            def track_call(operation, **kwargs):
                if operation == 'target':
                    entering_tcp.set()
                return call(operation, **kwargs)

            monkeypatch.setattr(sink.client, 'call', track_call)
            worker = threading.Thread(target=lambda: result.append(policy.step(
                {OBS_STATE: torch.zeros(1, 18)}, allow_dispatch=before_send)), daemon=True)
            with sink.client._io_lock:
                worker.start()
                assert entering_tcp.wait(2.)
                keys.rows.append([key])
            worker.join(2.)
            assert result == [None] and pending_key == [key]
            assert ui.apply_model_key(policy, 0, pending_key[0]) == transition
        finally:
            policy.close()
    assert not errors and not robot.commands
    assert [m['operation'] for m in messages] == operations
