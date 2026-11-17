"""Small model-independent contracts used by inference strategies."""

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class Prediction:
    """One canonical action chunk produced by a model adapter."""

    chunk: Any
    prediction_id: int | None
    latency_s: float


@dataclass(frozen=True)
class ActionSelection:
    """One action chosen from one or more model predictions."""

    action: Any
    prediction_id: int | None
    chunk_index: int
    selection: str
    contributors: tuple[dict, ...] = field(default_factory=tuple)


Predict = Callable[..., Prediction]


class InferenceStrategy:
    """Interface shared by synchronous chunks, ensembling and RTC."""

    name = 'base'

    def reset(self):
        raise NotImplementedError

    def select(self, batch, *, observation, predict: Predict, now_ns: int) -> ActionSelection:
        raise NotImplementedError

    def close(self):
        """Release optional background inference resources."""
