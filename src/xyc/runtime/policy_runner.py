"""Model-independent prediction recording and per-tick action selection."""

from dataclasses import replace
import time

import torch

from xyc.contracts import OBS_STATE
from .strategies import Prediction, RTCStrategy, SyncChunkStrategy
from .strategies.base import RTCChunk


def create_strategy(adapter, strategy=None, options=None, *, compute=None, record=None):
    """Build an explicit runtime strategy without guessing model capability."""
    if strategy is None or strategy == 'sync_chunk':
        return SyncChunkStrategy(adapter, **(options or {}))
    if strategy == 'rtc':
        return RTCStrategy(adapter, compute=compute, record=record, **(options or {}))
    if hasattr(strategy, 'select') and hasattr(strategy, 'reset'):
        if options:
            raise ValueError('Strategy options apply only to a named strategy')
        return strategy
    raise ValueError(f'Unknown inference strategy: {strategy!r}')


class PolicyRunner:
    """Return one canonical 18-D action per tick using a selected strategy."""

    def __init__(self, adapter, *, recorder=None, strategy=None, strategy_options=None):
        self.adapter = adapter
        self.recorder = recorder
        if recorder is not None and recorder.schema != adapter.schema:
            raise ValueError('Recorder and policy schemas differ')
        self.strategy = create_strategy(adapter, strategy, strategy_options,
                                        compute=self._compute, record=self._record_prediction)
        self.asynchronous = getattr(self.strategy, 'asynchronous', False)
        self.reset()

    def reset(self):
        self.last_step_id = self.last_prediction_id = None
        self.last_prediction = None
        self.action_deadline = None
        self.strategy.reset()
        if not self.asynchronous:
            self.adapter.reset()
        if self.recorder is not None:
            self.recorder.event('runner_reset', strategy=self.strategy.name)

    @staticmethod
    def _state(batch):
        # A model may request state history; traces retain the current measured state.
        return batch.get('observation.current_state', batch[OBS_STATE])[0]

    @torch.no_grad()
    def _compute(self, batch, *, method='predict_chunk', reset_model=False, **kwargs):
        if reset_model:
            self.adapter.reset()
        self.adapter.eval()
        predictor = getattr(self.adapter, method, None)
        if predictor is None:
            raise ValueError(
                f'{getattr(self.adapter, "name", type(self.adapter).__name__)} '
                f'does not implement {method}()')
        started = time.perf_counter()
        result = predictor(batch, **kwargs)
        if method == 'predict_rtc_chunk' and not isinstance(result, RTCChunk):
            raise ValueError('RTC adapter must return actions and a matching model-space chunk')
        chunk = result.actions if isinstance(result, RTCChunk) else result
        conditioning = result.conditioning if isinstance(result, RTCChunk) else None
        if not isinstance(chunk, torch.Tensor) or chunk.ndim != 3 or chunk.shape[0] != len(batch[OBS_STATE]):
            raise ValueError('Model adapter must return an action chunk shaped [batch,time,action_dim]')
        if chunk.shape[2] != len(self.adapter.schema.action_names) or not torch.isfinite(chunk).all():
            raise ValueError('Model adapter returned an invalid canonical action chunk')
        if isinstance(result, RTCChunk) and (
            not isinstance(conditioning, torch.Tensor)
            or conditioning.ndim != 3 or conditioning.shape[:2] != chunk.shape[:2]
            or not torch.isfinite(conditioning).all()
        ):
            raise ValueError('RTC adapter returned an invalid model-space chunk')
        if chunk.is_cuda:
            torch.cuda.synchronize(chunk.device)
        return Prediction(chunk, None, time.perf_counter() - started, conditioning)

    def _record_prediction(self, prediction, *, batch, observation):
        self.last_prediction = prediction.chunk
        self.last_prediction_id = None
        if self.recorder is not None:
            self.last_prediction_id = self.recorder.prediction(
                prediction.chunk[0], state=self._state(batch), observation=observation,
                latency_s=prediction.latency_s)
        return replace(prediction, prediction_id=self.last_prediction_id)

    def _predict(self, batch, *, observation=None):
        return self._record_prediction(self._compute(batch), batch=batch, observation=observation)

    @torch.no_grad()
    def warmup(self, batch, *, observation=None):
        """Initialize model kernels while the robot holds still; discard the chunk."""
        if self.asynchronous:
            raise ValueError('RTC initializes its first chunk in the background')
        reference = {**(observation or {}), 'phase': 'warmup_no_execution'}
        prediction = self._predict(batch, observation=reference)
        self.reset()
        return prediction.latency_s

    @torch.no_grad()
    def step(self, batch, *, observation=None, video_frames=None):
        if self.recorder is not None and len(batch[OBS_STATE]) != 1:
            raise ValueError('An inference recording represents one robot; expected batch size 1')
        self.last_step_id = self.action_deadline = None
        selected = self.strategy.select(
            batch, observation=observation, predict=self._predict, now_ns=time.monotonic_ns())
        if selected is None:
            return None  # Waiting for the first background chunk; keep the existing hold.
        if selected.valid_until_ns is not None:
            self.action_deadline = selected.valid_until_ns / 1e9
        action = selected.action
        if (not isinstance(action, torch.Tensor) or action.ndim != 2 or action.shape[0] != 1
                or action.shape[1] != len(self.adapter.schema.action_names)
                or not torch.isfinite(action).all()):
            raise ValueError('Inference strategy must select one finite action shaped [1,action_dim]')
        if self.recorder is not None:
            images = video_frames
            if self.recorder.record_video and images is None:
                images = {}
                for name in self.adapter.schema.cameras:
                    key = 'observation.current_images.' + name
                    frame = batch[key][0] if key in batch else batch['observation.images.' + name][0]
                    if frame.ndim != 3:
                        raise ValueError('Historical image batches must include observation.current_images')
                    images[name] = frame.detach().cpu().permute(1, 2, 0).numpy()
            self.last_step_id = self.recorder.selection(
                action[0], state=self._state(batch), prediction_id=selected.prediction_id,
                chunk_index=selected.chunk_index, observation=observation,
                selection=selected.selection, contributors=list(selected.contributors), images=images)
        return action

    def record_execution(self, step_id=None, **details):
        """Record a command sink result; the runner never infers acceptance."""
        if self.recorder is not None:
            self.recorder.execution(self.last_step_id if step_id is None else step_id, **details)

    def record_feedback(self, step_id=None, **details):
        if self.recorder is not None:
            self.recorder.feedback(self.last_step_id if step_id is None else step_id, **details)

    def close(self):
        self.strategy.close()
