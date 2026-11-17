"""Synchronous chunk queue and ACT temporal-ensemble selection."""

from collections import deque

from .base import ActionSelection, InferenceStrategy


class SyncChunkStrategy(InferenceStrategy):
    """Use a fixed prefix of every prediction, or ACT temporal ensembling.

    The scheduler is shared runtime code.  The temporal ensembler itself is a
    model capability supplied by ACT; other adapters are not assumed to have
    it.
    """

    name = 'sync_chunk'

    def __init__(self, adapter):
        self.adapter = adapter
        self.config = adapter.config
        if not hasattr(self.config, 'n_action_steps') or not hasattr(self.config, 'chunk_size'):
            raise ValueError('Synchronous chunk inference requires n_action_steps and chunk_size')
        coefficient = getattr(self.config, 'temporal_ensemble_coeff', None)
        if coefficient is not None and not hasattr(adapter.policy, 'temporal_ensembler'):
            raise ValueError('This adapter does not provide temporal ensembling')
        self.reset()

    def reset(self):
        self.actions = deque()
        self.ensemble_history = deque()
        self.tick = 0

    def select(self, batch, *, observation, predict, now_ns):
        contributors = ()
        coefficient = getattr(self.config, 'temporal_ensemble_coeff', None)
        if coefficient is not None:
            prediction = predict(batch, observation=observation)
            action = self.adapter.policy.temporal_ensembler.update(prediction.chunk)
            identity, index, selection = prediction.prediction_id, 0, 'temporal_ensemble'
            self.ensemble_history.append((identity, self.tick))
            while (self.ensemble_history
                   and self.tick - self.ensemble_history[0][1] >= self.config.chunk_size):
                self.ensemble_history.popleft()
            contributors = tuple(
                {'prediction_id': prediction_id, 'chunk_index': self.tick - prediction_tick}
                for prediction_id, prediction_tick in self.ensemble_history)
        else:
            if not self.actions:
                prediction = predict(batch, observation=observation)
                chunk = prediction.chunk[:, :self.config.n_action_steps]
                self.actions.extend(
                    (item, prediction.prediction_id, index)
                    for index, item in enumerate(chunk.transpose(0, 1)))
            action, identity, index = self.actions.popleft()
            selection = 'queue'
        self.tick += 1
        return ActionSelection(action, identity, index, selection, contributors)
