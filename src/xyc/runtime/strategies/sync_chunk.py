"""Synchronous chunk queue and adapter-provided ensemble selection."""

from collections import deque
from dataclasses import replace

from .base import ActionSelection, InferenceStrategy


class SyncChunkStrategy(InferenceStrategy):
    """Select actions using an adapter's declared chunk execution capability."""

    name = 'sync_chunk'

    def __init__(self, adapter, *, action_steps=None):
        self.adapter = adapter
        spec = adapter.chunk_execution_spec()
        if action_steps is not None:
            if type(action_steps) is not int or not 1 <= action_steps <= spec.horizon:
                raise ValueError(f'--action-steps must be an integer from 1 to {spec.horizon}')
            if spec.ensemble and action_steps != 1:
                raise ValueError('Temporal ensembling requires --action-steps 1')
            spec = replace(spec, action_steps=action_steps)
        self.spec = spec
        if spec.ensemble and not callable(getattr(adapter, 'ensemble_action', None)):
            raise ValueError('This adapter does not provide ensemble_action()')
        self.reset()

    def reset(self):
        self.actions = deque()
        self.ensemble_history = deque()
        self.tick = 0

    def select(self, batch, *, observation, predict, now_ns):
        contributors = ()
        if self.spec.ensemble:
            prediction = predict(batch, observation=observation)
            action = self.adapter.ensemble_action(prediction.chunk)
            identity, index, selection = prediction.prediction_id, 0, 'temporal_ensemble'
            self.ensemble_history.append((identity, self.tick))
            while (self.ensemble_history
                   and self.tick - self.ensemble_history[0][1] >= self.spec.horizon):
                self.ensemble_history.popleft()
            contributors = tuple(
                {'prediction_id': prediction_id, 'chunk_index': self.tick - prediction_tick}
                for prediction_id, prediction_tick in self.ensemble_history)
        else:
            if not self.actions:
                prediction = predict(batch, observation=observation)
                chunk = prediction.chunk[:, :self.spec.action_steps]
                self.actions.extend(
                    (item, prediction.prediction_id, index)
                    for index, item in enumerate(chunk.transpose(0, 1)))
            action, identity, index = self.actions.popleft()
            selection = 'queue'
        self.tick += 1
        return ActionSelection(action, identity, index, selection, contributors)
