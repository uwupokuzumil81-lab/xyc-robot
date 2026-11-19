"""Local π0.5 on the shared XYC data, checkpoint and action contracts."""

from dataclasses import asdict, replace

import torch
from torch import nn

from xyc.contracts import ACTION, OBS_STATE
from xyc.learning.data_spec import MetricValue, SampleSpec, StatisticsSpec, ValidationSpec
from xyc.learning.dataset import live_observation_batch
from xyc.runtime.strategies.base import ChunkExecutionSpec, RTCChunk
from . import backend
from .configuration_pi05 import PI05Config, Pi05Options
from .processor_pi05 import PI05Processor


class PI05Adapter(nn.Module):
    name = 'pi05'

    @staticmethod
    def training_options(options, *, smoke=False, no_pretrained=False):
        options = Pi05Options(**options)
        if smoke:
            options = replace(options, pretrained=None, dtype='float32', chunk_size=4,
                              n_action_steps=2, num_inference_steps=2, smoke=True,
                              freeze_vision_encoder=False, train_expert_only=False)
        elif no_pretrained:
            options = replace(options, pretrained=None)
        return asdict(options)

    @staticmethod
    def sample_spec(model_config, schema):
        options = Pi05Options(**model_config)
        return SampleSpec(action_offsets=tuple(range(options.chunk_size)), include_task=True)

    @staticmethod
    def statistics_spec(model_config, schema):
        return StatisticsSpec(features={key: ('min', 'max', 'q01', 'q99') for key in (OBS_STATE, ACTION)})

    @staticmethod
    def validation_spec():
        return ValidationSpec('action_l1_normalized', 'min')

    def __init__(self, model_config, schema, statistics, *, initialize_pretrained=True):
        super().__init__()

        self.options = Pi05Options(**model_config)
        self.schema, self.statistics = schema, statistics
        config = PI05Config(cameras=schema.cameras, action_dim=len(schema.action_names),
                            **asdict(self.options))
        if len(schema.state_names) > config.max_state_dim or config.action_dim > config.max_action_dim:
            raise ValueError('XYC schema exceeds the π0.5 state/action capacity')
        self.processor = PI05Processor(config, schema, statistics)
        self.policy = backend.build_policy(config, initialize_pretrained=initialize_pretrained)

    def model_config(self):
        return asdict(self.options)

    @property
    def device(self):
        return next(self.policy.parameters()).device

    def chunk_execution_spec(self):
        return ChunkExecutionSpec(self.options.n_action_steps, self.options.chunk_size)

    def live_batch(self, observation):
        return live_observation_batch(observation, self.schema)

    def prepare(self, batch, *, training=False):
        return self.processor.prepare(batch, device=self.device, training=training)

    def compute_loss(self, batch):
        self.train()
        prepared = self.prepare(batch, training=True)
        losses = self.policy(prepared)
        valid = ~prepared['action_is_pad']
        loss = losses.masked_fill(~valid.unsqueeze(-1), 0.).sum() / (valid.sum() * losses.shape[-1])
        return loss, {'flow_loss': float(loss.detach())}

    @torch.no_grad()
    def predict_chunk(self, batch):
        self.eval()
        return self._robot_actions(self.policy.predict_action_chunk(self.prepare(batch)))

    def _robot_actions(self, normalized):
        actions = self.processor.denormalize_actions(normalized)
        if not torch.isfinite(actions).all():
            raise ValueError('π0.5 produced non-finite actions')
        return actions

    @torch.no_grad()
    def predict_rtc_chunk(self, batch, *, previous, inference_delay, execution_horizon):
        normalized = self.policy.predict_action_chunk(
            self.prepare(batch),
            prev_chunk_left_over=None if previous is None else previous.to(self.device),
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
        )
        # RTC conditions on normalized actions. Keep them alongside the
        # physical chunk with identical time indices through every handoff.
        return RTCChunk(self._robot_actions(normalized), normalized.detach().cpu())

    @torch.no_grad()
    def validation_metrics(self, batch):
        self.eval()
        prepared = self.prepare(batch, training=True)
        # Evaluate checkpoints with identical noise, isolated from the training RNG.
        noise = torch.randn((len(prepared[OBS_STATE]), self.options.chunk_size, self.policy.config.max_action_dim),
                            dtype=torch.float32, generator=torch.Generator(device='cpu').manual_seed(0))
        prediction = self.policy.predict_action_chunk(prepared, noise=noise.to(self.device))
        valid = ~prepared['action_is_pad']
        errors = (prediction - prepared[ACTION]).abs().masked_fill(~valid.unsqueeze(-1), 0.)
        weight = valid.sum() * prediction.shape[-1]
        return {'action_l1_normalized': MetricValue(float(errors.sum() / weight), float(weight))}

    def optimizer_groups(self, *, learning_rate, backbone_learning_rate):
        backbone, expert = [], []
        for name, parameter in self.policy.named_parameters():
            if parameter.requires_grad:
                group = backbone if name.startswith('model.paligemma_with_expert.paligemma.') else expert
                group.append(parameter)
        groups = [{'params': expert, 'lr': learning_rate}]
        if backbone:
            groups.append({'params': backbone, 'lr': backbone_learning_rate})
        return groups

    def reset(self):
        """Action queues live in the shared runtime; each model call owns its KV cache."""
