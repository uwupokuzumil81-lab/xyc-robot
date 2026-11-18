"""Translate the platform's RGB/state/action batches into local ACT tensors."""

import torch
from torch import nn
import numpy as np

from xyc.contracts import ACTION, OBS_STATE
from xyc.learning.data_spec import SampleSpec, StatisticsSpec, ValidationSpec, MetricValue
from xyc.images import decode_jpeg, resize_rgb
from xyc.runtime.strategies.base import ChunkExecutionSpec
from .configuration_act import ACTConfig
from .modeling_act import ACTPolicy


class ACTAdapter(nn.Module):
    name = 'act'

    def model_config(self):
        """Export ACT constructor options; dimensions come from the shared schema."""
        return {key: value for key, value in self.config.to_dict().items()
                if key not in ('cameras', 'state_dim', 'action_dim')}

    @staticmethod
    def training_options(options, *, smoke=False, no_pretrained=False):
        options = dict(options)
        if smoke:
            options.update(image_size=[64, 64], chunk_size=4, n_action_steps=2,
                           dim_model=64, n_heads=4, dim_feedforward=256,
                           n_encoder_layers=1, n_decoder_layers=1,
                           n_vae_encoder_layers=1, pretrained_backbone_weights=None)
        if no_pretrained:
            options['pretrained_backbone_weights'] = None
        return options

    @staticmethod
    def sample_spec(model_config, schema):
        config = ACTConfig(cameras=schema.cameras, state_dim=len(schema.state_names),
                           action_dim=len(schema.action_names), **model_config)
        return SampleSpec(action_offsets=tuple(range(config.chunk_size)))

    @staticmethod
    def statistics_spec(model_config, schema):
        return StatisticsSpec(
            features={OBS_STATE: ('mean', 'std'), ACTION: ('mean', 'std')},
            constants={'images': {'mode': 'imagenet', 'mean': [.485, .456, .406],
                                  'std': [.229, .224, .225]}, 'epsilon': 1e-6})

    @staticmethod
    def validation_spec():
        return ValidationSpec('action_l1_normalized', 'min')

    def chunk_execution_spec(self):
        return ChunkExecutionSpec(self.config.n_action_steps, self.config.chunk_size,
                                  ensemble=self.config.temporal_ensemble_coeff is not None)

    def ensemble_action(self, chunk):
        return self.policy.temporal_ensembler.update(chunk)

    def __init__(self, model_config, schema, statistics, *, initialize_pretrained=True):
        super().__init__()
        self.schema, self.statistics = schema, statistics
        options = dict(model_config)
        for key in ('cameras', 'state_dim', 'action_dim'):
            if key in options:
                raise ValueError(f'{key} comes from the dataset schema, not model overrides')
        self.config = ACTConfig(cameras=schema.cameras, state_dim=len(schema.state_names),
                                action_dim=len(schema.action_names), **options)
        if not initialize_pretrained:
            # Restoration loads all weights from the checkpoint, without downloads.
            from dataclasses import replace
            build_config = replace(self.config, pretrained_backbone_weights=None)
        else:
            build_config = self.config
        self.policy = ACTPolicy(build_config)
        for name, key in (('state', OBS_STATE), ('action', ACTION)):
            for stat in ('mean', 'std'):
                value = torch.tensor(statistics[key][stat], dtype=torch.float32)
                size = len(schema.state_names if name == 'state' else schema.action_names)
                if value.shape != (size,) or not torch.isfinite(value).all() or stat == 'std' and (value < 0).any():
                    raise ValueError('Invalid normalization statistics')
                self.register_buffer(name + '_' + stat, value)
        self.register_buffer('image_mean', torch.tensor(statistics['images']['mean']).reshape(1, 3, 1, 1))
        self.register_buffer('image_std', torch.tensor(statistics['images']['std']).reshape(1, 3, 1, 1))
        self.epsilon = float(statistics['epsilon'])
        import math
        if (not math.isfinite(self.epsilon) or self.epsilon <= 0
                or not torch.isfinite(self.image_mean).all() or not torch.isfinite(self.image_std).all()
                or (self.image_std <= 0).any()):
            raise ValueError('Invalid image normalization or epsilon')

    @property
    def device(self):
        return self.state_mean.device

    def prepare(self, batch, *, training=False):
        state = batch[OBS_STATE].to(self.device, dtype=torch.float32)
        if state.ndim != 2 or state.shape[1] != len(self.schema.state_names) or not torch.isfinite(state).all():
            raise ValueError('Expected finite batched robot state')
        result = {OBS_STATE: (state - self.state_mean) / self.state_std.clamp_min(self.epsilon)}
        for key in self.config.image_features:
            image = batch[key]
            if image.dtype != torch.uint8 or image.ndim != 4 or tuple(image.shape[:2]) != (len(state), 3):
                raise ValueError(f'{key} must be raw RGB uint8 [batch,3,height,width]')
            if tuple(image.shape[-2:]) != tuple(self.config.image_size):
                images = image.detach().cpu().permute(0, 2, 3, 1).numpy()
                image = torch.stack([torch.from_numpy(resize_rgb(rgb, self.config.image_size)).permute(2, 0, 1)
                                     for rgb in images])
            image = image.to(self.device, dtype=torch.float32) / 255.
            result[key] = (image - self.image_mean) / self.image_std
        if training:
            action = batch[ACTION].to(self.device, dtype=torch.float32)
            mask = batch['action_is_pad'].to(self.device)
            if (tuple(action.shape) != (len(state), self.config.chunk_size, len(self.schema.action_names))
                    or tuple(mask.shape) != action.shape[:2] or mask.dtype != torch.bool
                    or mask.all(1).any() or not torch.isfinite(action).all()):
                raise ValueError('Invalid action chunk or episode padding mask')
            result[ACTION] = (action - self.action_mean) / self.action_std.clamp_min(self.epsilon)
            result['action_is_pad'] = mask
        return result

    def live_batch(self, observation):
        """ACT-owned preprocessing for one canonical live robot observation."""
        state = observation.get('state')
        images = observation.get('images')
        if (not isinstance(state, (tuple, list)) or len(state) != len(self.schema.state_names)
                or not isinstance(images, dict) or set(images) != set(self.schema.cameras)):
            raise ValueError('Live ACT observation does not match the checkpoint schema')
        batch = {OBS_STATE: torch.tensor([state], dtype=torch.float32)}
        for name in self.schema.cameras:
            rgb = decode_jpeg(images[name])
            batch['observation.images.' + name] = torch.from_numpy(
                np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
        return batch

    def compute_loss(self, batch):
        self.train()
        return self.policy(self.prepare(batch, training=True))

    @torch.no_grad()
    def predict_chunk(self, batch):
        self.eval()
        normalized = self.policy.predict_action_chunk(self.prepare(batch))
        result = normalized.float() * self.action_std.clamp_min(self.epsilon) + self.action_mean
        if not torch.isfinite(result).all():
            raise ValueError('Model produced non-finite actions')
        return result

    @torch.no_grad()
    def validation_metrics(self, batch):
        # Deployed inference uses z=0; report its masked action error, not a
        # training-mode stochastic VAE ELBO labelled as deployment performance.
        predicted = self.predict_chunk(batch)
        prepared = self.prepare(batch, training=True)
        normalized = (predicted - self.action_mean) / self.action_std.clamp_min(self.epsilon)
        valid = (~prepared['action_is_pad']).unsqueeze(-1)
        error = ((normalized - prepared[ACTION]).abs() * valid).sum()
        denom = valid.sum() * len(self.schema.action_names)
        return {'action_l1_normalized': MetricValue(float(error / denom.clamp_min(1)), float(denom))}

    def optimizer_groups(self, *, learning_rate, backbone_learning_rate):
        groups = self.policy.get_optim_params()
        groups[0]['lr'], groups[1]['lr'] = learning_rate, backbone_learning_rate
        return groups

    def reset(self):
        self.policy.reset()
