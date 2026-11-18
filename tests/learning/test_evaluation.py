"""Manual preparation, pause/resume and common execution use fake hardware."""

import json

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
    assert ui.model_keys(executor, sink.status(), 1) == 'prepare'
    assert ui.successes == 1
    height = sink.height
    assert ui.prepare(executor)
    executor.runner.reset()
    executor.step({OBS_STATE: torch.zeros(1, 18)})
    assert adapter.calls == 2 and sink.height == height
    assert not sink.stopped and sink.homes == 0 and ui.rounds == 2


def test_manual_quit_never_runs_a_model():
    ui, executor, sink, adapter = console([['q']])
    assert ui.prepare(executor) is False
    assert adapter.calls == 0 and sink.sent == []


def test_teleop_preparation_tracks_leaders_then_pauses_them_for_model_control():
    ui, executor, sink, adapter = console([['i', 'w'], ['p']])
    class Arms:
        notice = '双主臂反馈正常'
        active = False
        calls = 0
        def resume(self): self.active = True
        def pause(self): self.active = False
        def targets(self, follower):
            assert self.active
            self.calls += 1
            values = [0.] * 14
            values[12] = 2.
            return values
    source = Arms()
    ui.arm_input = source
    sink.start()
    assert ui.prepare(executor)
    assert adapter.calls == 0 and sink.arms[12] == 2.
    assert source.active is False
    sampled = source.calls
    executor.step({OBS_STATE: torch.zeros(1, 18)})
    assert source.calls == sampled and sink.starts == 1


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


def test_deploy_waits_for_operator_before_predicting_and_keeps_same_sink(monkeypatch, tmp_path):
    adapter, _, _, sink = live_fixture(monkeypatch)
    transitions = []
    class UI:
        rounds = 0
        successes = failures = 0
        def __init__(self, *args, **kwargs): pass
        def open(self): transitions.append('open')
        def close(self): transitions.append('close')
        def prepare(self, executor):
            assert executor.command_sink is sink
            assert adapter.calls == self.rounds * 2
            self.rounds += 1
            transitions.append('prepare')
            return True
        def model_keys(self, executor, state, steps):
            return 'prepare' if steps == 1 and self.rounds == 1 else 'model'
    monkeypatch.setattr(deployment, 'EvaluationConsole', UI)
    result = deployment.deploy('test', device='cpu', steps=2, record_dir=tmp_path / 'trace')
    assert transitions == ['open', 'prepare', 'prepare', 'close']
    assert sink.started and sink.stopped and adapter.calls == 4
    assert result['evaluations']['started'] == 2
    assert json.loads((tmp_path / 'trace/meta.json').read_text())['status'] == 'completed'
