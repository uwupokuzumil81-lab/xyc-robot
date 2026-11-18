"""Adapter-declared sampling and statistics, independent of any model family."""

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class SampleSpec:
    action_offsets: tuple[int, ...]
    state_offsets: tuple[int, ...] | None = None
    image_offsets: tuple[int, ...] | None = None
    include_task: bool = False

    def __post_init__(self):
        for name in ('action_offsets', 'state_offsets', 'image_offsets'):
            values = getattr(self, name)
            if values is None and name != 'action_offsets':
                continue
            values = tuple(values)
            if not values or any(type(v) is not int for v in values) or len(set(values)) != len(values):
                raise ValueError('Sample offsets must be nonempty unique integer frame offsets')
            object.__setattr__(self, name, values)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class StatisticsSpec:
    features: dict[str, tuple[str, ...]]
    constants: dict = field(default_factory=dict)

    def __post_init__(self):
        supported = {'mean', 'std', 'min', 'max', 'q01', 'q99'}
        if not self.features or set(self.features) & set(self.constants):
            raise ValueError('Statistics need feature requests separate from constants')
        if any(not methods or set(methods) - supported for methods in self.features.values()):
            raise ValueError('Unknown or empty statistics request')

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ValidationSpec:
    primary: str
    direction: str = 'min'

    def __post_init__(self):
        if not self.primary or self.direction not in ('min', 'max'):
            raise ValueError('Validation must declare a primary metric and min/max direction')

    def initial_best(self):
        return float('inf') if self.direction == 'min' else -float('inf')

    def improved(self, value, previous):
        return value < previous if self.direction == 'min' else value > previous

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class MetricValue:
    value: float
    weight: float

    def __post_init__(self):
        import math
        if not math.isfinite(self.value) or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError('Validation metrics need a finite value and positive weight')
