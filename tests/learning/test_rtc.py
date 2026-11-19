"""Regression tests for overlap, lifecycle and bounded RTC actions."""

from threading import Event, get_ident

import pytest
import torch

from test_platform_interfaces import PortableAdapter, schema
from xyc.contracts import OBS_STATE
from xyc.runtime.policy_runner import PolicyRunner
from xyc.runtime.strategies.base import RTCChunk


def batch(value=0.):
    return {OBS_STATE: torch.full((1, 18), value)}


def chunk(start):
    actions = torch.arange(start, start + 4).reshape(1, 4, 1).expand(1, 4, 18).float()
    return RTCChunk(actions, actions + 100.)


def test_rtc_consumes_old_actions_while_predicting_and_skips_actual_overlap():
    adapter = PortableAdapter(schema())
    started, finish = Event(), Event()
    calls = []
    caller = get_ident()

    def predict(sample, *, previous, inference_delay, execution_horizon):
        assert get_ident() != caller and not torch.is_grad_enabled()
        calls.append((float(sample[OBS_STATE][0, 0]), previous, inference_delay, execution_horizon))
        if previous is None:
            return chunk(0)
        started.set()
        assert finish.wait(3)
        return chunk(10)

    adapter.predict_rtc_chunk = predict
    runner = PolicyRunner(adapter, strategy='rtc')

    def select(index):
        return runner.strategy.select(batch(float(index)), observation={'frame': index}, predict=None,
                                      now_ns=index * 10_000_000)

    try:
        assert select(0) is None
        runner.strategy.pending.future.result(timeout=3)
        assert [select(i).action[0, 0].item() for i in (1, 2)] == [0, 1]
        assert select(3).action[0, 0] == 2
        assert started.wait(3) and not runner.strategy.pending.future.done()
        finish.set()
        runner.strategy.pending.future.result(timeout=3)
        selected = select(4)
        assert selected.action[0, 0] == 11 and selected.chunk_index == 1
        assert selected.valid_until_ns == 30_000_000 + runner.strategy.window_ns
        _, prefix, delay, horizon = calls[1]
        assert prefix[0, :, 0].tolist() == [102., 103.]
        assert 1 <= delay <= horizon == 2
    finally:
        finish.set()
        runner.close()


def test_rtc_reset_discards_inflight_result_and_serializes_model_access():
    adapter = PortableAdapter(schema())
    started, finish = Event(), Event()
    resets = []
    adapter.reset = lambda: resets.append(get_ident())

    def predict(sample, **kwargs):
        value = int(sample[OBS_STATE][0, 0])
        if value == 0:
            started.set()
            assert finish.wait(3)
        return chunk(value)

    adapter.predict_rtc_chunk = predict
    runner = PolicyRunner(adapter, strategy='rtc')
    select = lambda value: runner.strategy.select(batch(float(value)), observation=None, predict=None, now_ns=0)
    try:
        assert select(0) is None and started.wait(3)
        old = runner.strategy.pending.future
        runner.reset()
        assert select(20) is None and len(resets) == 1
        finish.set()
        old.result(timeout=3)
        assert select(20) is None
        runner.strategy.pending.future.result(timeout=3)
        assert select(20).action[0, 0] == 20
        assert len(resets) == 2 and all(t != get_ident() for t in resets)
    finally:
        finish.set()
        runner.close()


def test_rtc_refuses_unguided_successor_at_the_end_of_old_chunk():
    adapter = PortableAdapter(schema())
    finish = Event()

    def predict(_, *, previous, **kwargs):
        if previous is None:
            return chunk(0)
        assert finish.wait(3)
        return chunk(50)

    adapter.predict_rtc_chunk = predict
    runner = PolicyRunner(adapter, strategy='rtc')
    strategy = runner.strategy

    def select(tick):
        return strategy.select(batch(), observation=None, predict=None, now_ns=tick * 10_000_000)

    try:
        assert select(0) is None
        strategy.pending.future.result(timeout=3)
        assert [select(i).action[0, 0].item() for i in range(1, 5)] == [0, 1, 2, 3]
        finish.set()
        strategy.pending.future.result(timeout=3)
        with pytest.raises(TimeoutError, match='ran out of valid actions'):
            select(5)
        assert strategy.failed
    finally:
        finish.set()
        runner.close()


@pytest.mark.parametrize('mode', ('empty', 'expired'))
def test_rtc_fault_latches_if_old_actions_expire_or_run_out(mode):
    adapter = PortableAdapter(schema())
    finish = Event()

    def predict(_, *, previous, **kwargs):
        if previous is not None:
            assert finish.wait(3)
        return chunk(0)

    adapter.predict_rtc_chunk = predict
    runner = PolicyRunner(adapter, strategy='rtc')
    strategy = runner.strategy

    def select(ns):
        return strategy.select(batch(), observation=None, predict=None, now_ns=ns)

    try:
        assert select(0) is None
        strategy.pending.future.result(timeout=3)
        for index in range(1, 5 if mode == 'empty' else 4):
            assert select(index).action[0, 0] == index - 1
        with pytest.raises(TimeoutError, match='ran out of valid actions'):
            select(5 if mode == 'empty' else strategy.window_ns)
        finish.set()
        strategy.pending.future.result(timeout=3)
        with pytest.raises(RuntimeError, match='faulted'):
            select(6)
    finally:
        finish.set()
        runner.close()
