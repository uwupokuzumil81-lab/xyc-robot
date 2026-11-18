"""Model-independent prediction recording and per-tick action selection."""

import time

import torch

from xyc.contracts import OBS_STATE
from .strategies import Prediction, RTCStrategy, SyncChunkStrategy


def create_strategy(adapter, strategy=None, options=None):
    """Build an explicit runtime strategy without guessing model capability."""
    if strategy is None or strategy == 'sync_chunk':
        return SyncChunkStrategy(adapter)
    if strategy == 'rtc':
        return RTCStrategy(adapter, **(options or {}))
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
        self.strategy = create_strategy(adapter, strategy, strategy_options)
        self.reset()

    def reset(self):
        self.last_step_id = self.last_prediction_id = None
        self.last_prediction = None
        self.adapter.reset()
        self.strategy.reset()
        if self.recorder is not None:
            self.recorder.event('runner_reset', strategy=self.strategy.name)

    @staticmethod
    def _state(batch):
        # A model may request state history; traces retain the current measured state.
        return batch.get('observation.current_state', batch[OBS_STATE])[0]

    def _predict(self, batch, *, observation=None, method='predict_chunk', **kwargs):
        predictor = getattr(self.adapter, method, None)
        if predictor is None:
            raise ValueError(
                f'{getattr(self.adapter, "name", type(self.adapter).__name__)} '
                f'does not implement {method}()')
        started = time.perf_counter()
        chunk = predictor(batch, **kwargs)
        if not isinstance(chunk, torch.Tensor) or chunk.ndim != 3 or chunk.shape[0] != len(batch[OBS_STATE]):
            raise ValueError('Model adapter must return an action chunk shaped [batch,time,action_dim]')
        if chunk.shape[2] != len(self.adapter.schema.action_names) or not torch.isfinite(chunk).all():
            raise ValueError('Model adapter returned an invalid canonical action chunk')
        if chunk.is_cuda:
            torch.cuda.synchronize(chunk.device)
        latency = time.perf_counter() - started
        self.last_prediction = chunk
        self.last_prediction_id = None
        if self.recorder is not None:
            self.last_prediction_id = self.recorder.prediction(
                chunk[0], state=self._state(batch), observation=observation, latency_s=latency)
        return Prediction(chunk, self.last_prediction_id, latency)

    @torch.no_grad()
    def warmup(self, batch, *, observation=None):
        """Initialize model kernels while the robot holds still; discard the chunk."""
        self.adapter.eval()
        reference = {**(observation or {}), 'phase': 'warmup_no_execution'}
        prediction = self._predict(batch, observation=reference)
        self.reset()
        return prediction.latency_s

    @torch.no_grad()
    def step(self, batch, *, observation=None, video_frames=None):
        if self.recorder is not None and len(batch[OBS_STATE]) != 1:
            raise ValueError('An inference recording represents one robot; expected batch size 1')
        self.adapter.eval()
        selected = self.strategy.select(
            batch, observation=observation, predict=self._predict, now_ns=time.perf_counter_ns())
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
