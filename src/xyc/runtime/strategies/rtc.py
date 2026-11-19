"""Background action chunk scheduling, independent of model and robot I/O."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math

from .base import ActionSelection, InferenceStrategy, Prediction


@dataclass(frozen=True)
class PendingChunk:
    future: Future
    batch: dict
    observation: dict | None
    requested_tick: int
    valid_until_ns: int
    guided_steps: int | None


class RTCStrategy(InferenceStrategy):
    """Keep executing a checked chunk while a single worker predicts its successor."""

    name = 'rtc'
    asynchronous = True

    def __init__(self, adapter, *, compute, record):
        if not callable(getattr(adapter, 'predict_rtc_chunk', None)):
            raise ValueError(f'{adapter.name} does not support RTC; use sync_chunk')
        spec = adapter.chunk_execution_spec()
        if spec.ensemble or spec.horizon < 2:
            raise ValueError('RTC needs a non-ensembled prediction horizon of at least two actions')
        self.horizon = spec.horizon
        self.stride = min(spec.action_steps, self.horizon - 1)
        self.fps = adapter.schema.fps
        self.period_ns = round(1e9 / self.fps)
        self.window_ns = round(self.horizon * 1e9 / self.fps)
        self.compute, self.record = compute, record
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='xyc-rtc')
        self.pending = self.discarded = None
        self.reset()

    def reset(self):
        if self.pending is not None:
            self.discarded = self.pending.future
            self.discarded.cancel()
        self.pending = None
        self.active: Prediction | None = None
        self.cursor = self.tick = self.adopted_tick = 0
        self.active_until_ns = self.startup_until_ns = None
        self.delay_steps = 1
        self.failed = False

    def _submit(self, batch, observation, now_ns):
        initial = self.active is None
        tail = None if initial else self.active.conditioning[:, self.cursor:]
        overlap = 0 if initial else min(tail.shape[1], self.delay_steps + self.stride)
        future = self.worker.submit(
            self.compute, batch, method='predict_rtc_chunk', reset_model=initial,
            previous=tail, inference_delay=min(self.delay_steps, overlap), execution_horizon=overlap,
        )
        self.pending = PendingChunk(future, batch, observation, self.tick, now_ns + self.window_ns,
                                    None if initial else overlap)

    def _adopt(self, now_ns):
        pending = self.pending
        if pending is None:
            return
        if now_ns >= pending.valid_until_ns:
            raise TimeoutError('RTC prediction exceeded its action horizon')
        if not pending.future.done():
            return
        result = pending.future.result()
        consumed = self.tick - pending.requested_tick
        if result.chunk.shape[1] != self.horizon or consumed >= self.horizon:
            raise ValueError('RTC prediction has no usable steps at the current action index')
        self.delay_steps = max(1, consumed, math.ceil(result.latency_s * self.fps))
        self.pending = None
        if pending.guided_steps is not None and consumed >= pending.guided_steps:
            # Keep any valid remaining actions and request a better-aligned successor.
            return
        self.active = self.record(result, batch=pending.batch, observation=pending.observation)
        self.cursor, self.adopted_tick = consumed, self.tick
        self.active_until_ns = pending.valid_until_ns

    def select(self, batch, *, observation, predict, now_ns):
        if self.failed:
            raise RuntimeError('RTC evaluation is faulted; reset before starting another evaluation')
        try:
            return self._select(batch, observation, now_ns)
        except Exception:
            self.failed = True
            raise

    def _select(self, batch, observation, now_ns):
        if self.active is None and self.startup_until_ns is not None and now_ns >= self.startup_until_ns:
            raise TimeoutError('RTC initial chunk did not arrive within one action horizon')
        self._adopt(now_ns)
        if self.active is None:
            if self.startup_until_ns is None:
                self.startup_until_ns = now_ns + self.window_ns
            if self.discarded is not None:
                if not self.discarded.done():
                    return None
                self.discarded = None
            if self.pending is None:
                self._submit(batch, observation, now_ns)
            return None

        if self.cursor >= self.horizon or now_ns >= self.active_until_ns:
            raise TimeoutError('RTC ran out of valid actions before the next chunk was ready')
        if self.pending is None and (
            self.tick - self.adopted_tick >= self.stride
            or self.horizon - self.cursor <= self.delay_steps + 1
            or self.active_until_ns - now_ns <= (self.delay_steps + 1) * self.period_ns
        ):
            self._submit(batch, observation, now_ns)

        index = self.cursor
        self.cursor += 1
        self.tick += 1
        return ActionSelection(self.active.chunk[:, index], self.active.prediction_id, index,
                               'rtc', valid_until_ns=self.active_until_ns)

    def close(self):
        self.reset()
        self.worker.shutdown(wait=False, cancel_futures=True)
