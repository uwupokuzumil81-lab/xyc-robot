"""Local π0.5 checkpoint options and architecture settings."""

from dataclasses import dataclass
from typing import ClassVar

from xyc.contracts import OBS_IMAGES


DEFAULT_IMAGE_SIZE = 224
SMOKE_VOCAB_SIZE = 258  # All byte-level symbols, plus padding and unknown tokens.


@dataclass(frozen=True)
class Pi05Options:
    pretrained: str | None = 'lerobot/pi05_base'
    pretrained_revision: str | None = None
    dtype: str = 'bfloat16'
    chunk_size: int = 50
    n_action_steps: int = 10
    num_inference_steps: int = 10
    gradient_checkpointing: bool = True
    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    smoke: bool = False

    def __post_init__(self):
        if (any(type(n) is not int or n < 1 for n in
                (self.chunk_size, self.n_action_steps, self.num_inference_steps))
                or self.n_action_steps > self.chunk_size):
            raise ValueError('Expected positive π0.5 steps and n_action_steps <= chunk_size')
        if self.dtype not in ('float32', 'bfloat16'):
            raise ValueError('π0.5 dtype must be float32 or bfloat16')
        if self.pretrained is not None and (not isinstance(self.pretrained, str) or not self.pretrained.strip()):
            raise ValueError('π0.5 pretrained must be a Hub repository, local directory or null')
        if type(self.smoke) is not bool or (self.smoke and self.pretrained is not None):
            raise ValueError('π0.5 smoke architecture requires randomly initialized weights')


@dataclass(frozen=True)
class PI05Config(Pi05Options):
    cameras: tuple[str, ...] = ()
    action_dim: int = 18

    paligemma_variant: ClassVar[str] = 'gemma_2b'
    action_expert_variant: ClassVar[str] = 'gemma_300m'
    max_state_dim: ClassVar[int] = 32
    max_action_dim: ClassVar[int] = 32
    time_sampling_beta_alpha: ClassVar[float] = 1.5
    time_sampling_beta_beta: ClassVar[float] = 1.0
    time_sampling_scale: ClassVar[float] = .999
    time_sampling_offset: ClassVar[float] = .001
    min_period: ClassVar[float] = .004
    max_period: ClassVar[float] = 4.

    @property
    def image_resolution(self):
        size = 28 if self.smoke else DEFAULT_IMAGE_SIZE
        return size, size

    @property
    def image_features(self):
        return tuple(f'{OBS_IMAGES}.{camera}' for camera in self.cameras)
