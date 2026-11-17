"""Platform-owned ACT configuration; no LeRobot config or feature dependency."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Feature:
    shape: tuple[int, ...]


@dataclass
class ACTConfig:
    cameras: tuple[str, ...]
    state_dim: int
    action_dim: int
    image_size: tuple[int, int] = (240, 320)
    chunk_size: int = 50
    n_action_steps: int = 10
    n_obs_steps: int = 1
    vision_backbone: str = 'resnet18'
    pretrained_backbone_weights: str | None = 'ResNet18_Weights.IMAGENET1K_V1'
    replace_final_stride_with_dilation: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    pre_norm: bool = False
    dropout: float = .1
    feedforward_activation: str = 'relu'
    kl_weight: float = 10.
    temporal_ensemble_coeff: float | None = None
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        self.cameras, self.image_size = tuple(self.cameras), tuple(self.image_size)
        self.validate_features()

    def validate_features(self):
        if (not self.cameras or len(set(self.cameras)) != len(self.cameras)
                or self.state_dim <= 0 or self.action_dim <= 0 or self.n_obs_steps != 1
                or not 1 <= self.n_action_steps <= self.chunk_size or self.chunk_size < 1
                or len(self.image_size) != 2 or min(self.image_size) < 32
                or self.dim_model % self.n_heads or self.dim_model % 4
                or min(self.n_encoder_layers, self.n_decoder_layers, self.n_vae_encoder_layers) < 1
                or not 0 <= self.dropout < 1 or self.kl_weight < 0
                or self.vision_backbone not in ('resnet18', 'resnet34', 'resnet50')):
            raise ValueError('Invalid ACT dimensions, layers or input/output configuration')
        if self.temporal_ensemble_coeff is not None and self.n_action_steps != 1:
            raise ValueError('Temporal ensembling requires n_action_steps=1')

    @property
    def image_features(self):
        return {'observation.images.' + name: Feature((3, *self.image_size)) for name in self.cameras}

    @property
    def robot_state_feature(self):
        return Feature((self.state_dim,))

    @property
    def action_feature(self):
        return Feature((self.action_dim,))

    @property
    def env_state_feature(self):
        return None

    def to_dict(self):
        return asdict(self)
