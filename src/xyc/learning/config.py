"""Strict, model-independent training settings and project-relative paths."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import os
import yaml


@dataclass
class TrainConfig:
    dataset: str
    output_dir: str
    policy: str = 'act'
    model: dict = field(default_factory=dict)
    cameras: list[str] | None = None
    steps: int = 10000
    batch_size: int = 8
    learning_rate: float = 1e-5
    backbone_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip_norm: float = 10.
    scheduler: str = 'constant'
    warmup_steps: int = 0
    save_every: int = 1000
    eval_every: int = 1000
    eval_batches: int = 4
    log_every: int = 10
    num_workers: int = 0
    seed: int = 42
    validation_fraction: float = .2
    train_episodes: list[int] | None = None
    validation_episodes: list[int] | None = None
    device: str = 'cuda'
    precision: str = 'float32'
    cpu_threads: int = 4

    def validate(self):
        for name in ('steps', 'batch_size', 'save_every', 'eval_every', 'eval_batches', 'log_every', 'cpu_threads'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if (self.num_workers < 0 or self.warmup_steps < 0 or self.warmup_steps > self.steps
                or self.learning_rate <= 0 or self.backbone_learning_rate <= 0
                or self.weight_decay < 0 or self.grad_clip_norm <= 0
                or self.scheduler not in ('constant', 'cosine') or self.precision not in ('float32', 'bfloat16')
                or not 0 <= self.validation_fraction < 1):
            raise ValueError('Invalid training, optimizer or precision setting')
        return self

    def to_dict(self):
        return asdict(self)


def project_root():
    if value := os.environ.get('XYC_PROJECT_ROOT'):
        return Path(value).expanduser().resolve()
    candidates = [Path(__file__).resolve().parents[3], Path.cwd(), *Path.cwd().parents]
    return next((p for p in candidates if (p / 'pyproject.toml').is_file()
                 and (p / 'config/train/act.yaml').is_file()), Path.cwd())


def load_config(path):
    path = Path(path).expanduser().resolve()
    value = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or set(value) - {f.name for f in fields(TrainConfig)}:
        raise ValueError('Unknown or malformed training settings')
    base = next((p for p in path.parents if (p / 'pyproject.toml').is_file()), path.parent)
    for key in ('dataset', 'output_dir'):
        target = Path(value[key]).expanduser()
        value[key] = str((base / target).resolve()) if not target.is_absolute() else str(target.resolve())
    return TrainConfig(**value).validate()
