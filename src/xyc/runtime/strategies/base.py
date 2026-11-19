"""Small model-independent contracts used by inference strategies."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class RTCChunk:
    """Robot-space actions paired with the same steps in the model's space."""

    actions: Any
    conditioning: Any


@dataclass(frozen=True)
class Prediction:
    """One canonical action chunk produced by a model adapter."""

    chunk: Any
    prediction_id: int | None
    latency_s: float
    conditioning: Any = None


@dataclass(frozen=True)
class ActionSelection:
    """One action chosen from one or more model predictions."""

    action: Any
    prediction_id: int | None
    chunk_index: int
    selection: str
    contributors: tuple[dict, ...] = field(default_factory=tuple)
    valid_until_ns: int | None = None


Predict = Callable[..., Prediction]


@dataclass(frozen=True)
class ChunkExecutionSpec:
    action_steps: int
    horizon: int
    ensemble: bool = False

    def __post_init__(self):
        if not 1 <= self.action_steps <= self.horizon:
            raise ValueError('Chunk execution needs 1 <= action_steps <= horizon')


class InferenceStrategy:
    """Interface shared by synchronous chunks, ensembling and RTC."""

    name = 'base'
    asynchronous = False

    def reset(self):
        raise NotImplementedError

    def select(self, batch, *, observation, predict: Predict, now_ns: int) -> ActionSelection:
        raise NotImplementedError

    def close(self):
        """Release optional background inference resources."""
