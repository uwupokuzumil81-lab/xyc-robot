# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team.
# SPDX-License-Identifier: Apache-2.0
"""Local π0.5 preprocessing adapted from LeRobot v0.5.1. See PROVENANCE.json."""

import numpy as np
import torch
import torch.nn.functional as F

from xyc.contracts import ACTION, OBS_STATE


LANGUAGE_TOKENS = 'observation.language.tokens'
LANGUAGE_MASK = 'observation.language.attention_mask'
TOKENIZER_SOURCE = 'google/paligemma-3b-pt-224'
TOKENIZER_MAX_LENGTH = 200
STATE_BINS = np.linspace(-1, 1, 257)[:-1]


def _smoke_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {'[PAD]': 0, '[UNK]': 1}
    vocab.update({char: i for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()), 2)})
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', pad_token='[PAD]')


def _quantile_ranges(statistics, schema):
    """Select training-set bounds and preserve the known ternary command range."""
    result = {}
    for key, units in ((OBS_STATE, schema.state_units), (ACTION, schema.action_units)):
        bounds = {name: torch.as_tensor(statistics[key][name], dtype=torch.float32)
                  for name in ('min', 'max', 'q01', 'q99')}
        if any(value.shape != (len(units),) or not torch.isfinite(value).all() for value in bounds.values()):
            raise ValueError(f'Invalid π0.5 statistics for {key}')
        lo, hi = bounds['q01'], bounds['q99']
        minimum, maximum = bounds['min'], bounds['max']
        if ((minimum > lo) | (lo > hi) | (hi > maximum)).any():
            raise ValueError(f'Invalid π0.5 quantile ordering for {key}')
        collapsed = lo == hi
        lo, hi = torch.where(collapsed, minimum, lo), torch.where(collapsed, maximum, hi)
        if key == ACTION:
            ternary = torch.tensor([unit == 'ternary' for unit in units])
            lo, hi = torch.where(ternary, -1., lo), torch.where(ternary, 1., hi)
        span = hi - lo
        result[key] = lo, torch.where(span == 0, 1e-8, span)
    return result


def state_prompts(state, tasks):
    """Quantize the normalized robot state and append it to the task prompt."""
    bins = np.digitize(state.cpu().numpy(), STATE_BINS) - 1
    prompts = []
    for task, values in zip(tasks, bins, strict=True):
        text = task.strip().replace('_', ' ').replace('\n', ' ')
        numbers = ' '.join(map(str, values))
        prompts.append(f'Task: {text}, State: {numbers};\nAction: ')
    return prompts


def _siglip_image(image, resolution, device):
    image = image.to(device, dtype=torch.float32).div_(255.)
    height, width = resolution
    current_height, current_width = image.shape[-2:]
    if (current_height, current_width) != resolution:
        ratio = max(current_width / width, current_height / height)
        resized_height, resized_width = int(current_height / ratio), int(current_width / ratio)
        image = F.interpolate(image, (resized_height, resized_width),
                              mode='bilinear', align_corners=False).clamp_(0., 1.)
        top, extra_h = divmod(height - resized_height, 2)
        left, extra_w = divmod(width - resized_width, 2)
        image = F.pad(image, (left, left + extra_w, top, top + extra_h))
    return image * 2.0 - 1.0


class PI05Processor:
    """XYC batches in, PaliGemma inputs and denormalized robot actions out."""

    def __init__(self, config, schema, statistics):
        try:
            from transformers import AutoTokenizer
        except ModuleNotFoundError as exc:
            raise ValueError('Install π0.5 dependencies with pip install -e ".[pi05]"') from exc

        self.config, self.schema = config, schema
        self.ranges = _quantile_ranges(statistics, schema)
        try:
            self.tokenizer = (_smoke_tokenizer() if config.smoke else
                              AutoTokenizer.from_pretrained(TOKENIZER_SOURCE))
        except OSError as exc:
            raise ValueError('Unable to load the PaliGemma tokenizer. Request access to '
                             f'{TOKENIZER_SOURCE} and run hf auth login, or use its local Hub cache.') from exc

    def _normalize(self, key, value):
        lower, span = self.ranges[key]
        return 2.0 * (value - lower) / span - 1.0

    def prepare(self, batch, *, device, training=False):
        state, tasks = batch[OBS_STATE], batch.get('task')
        if state.ndim != 2 or state.shape[1] != len(self.schema.state_names) or not torch.isfinite(state).all():
            raise ValueError('π0.5 expects finite batched robot state')
        if (not isinstance(tasks, (tuple, list)) or len(tasks) != len(state)
                or any(not isinstance(task, str) or not task.strip() for task in tasks)):
            raise ValueError('π0.5 requires one nonempty task instruction per observation')
        normalized_state = self._normalize(OBS_STATE, state.to(device='cpu', dtype=torch.float32))
        prompts = state_prompts(normalized_state, tasks)
        tokens = self.tokenizer(prompts, max_length=TOKENIZER_MAX_LENGTH, truncation=True,
                                padding='max_length', padding_side='right', return_tensors='pt')
        prepared = {OBS_STATE: normalized_state.to(device), 'task': prompts,
                    LANGUAGE_TOKENS: tokens['input_ids'].to(device),
                    LANGUAGE_MASK: tokens['attention_mask'].to(device, dtype=torch.bool)}
        for key in self.config.image_features:
            image = batch[key]
            if image.dtype != torch.uint8 or image.ndim != 4 or image.shape[:2] != (len(state), 3):
                raise ValueError(f'{key} must be raw RGB uint8 [batch,3,height,width]')
            prepared[key] = _siglip_image(image, self.config.image_resolution, device)
        if training:
            actions, padding = batch[ACTION], batch['action_is_pad']
            if (actions.shape != (len(state), self.config.chunk_size, self.config.action_dim)
                    or padding.shape != actions.shape[:2] or padding.dtype != torch.bool
                    or padding.all(1).any() or not torch.isfinite(actions).all()):
                raise ValueError('Invalid π0.5 action chunk or episode padding mask')
            actions = actions.to(device='cpu', dtype=torch.float32)
            prepared[ACTION] = self._normalize(ACTION, actions).to(device)
            prepared['action_is_pad'] = padding.to(device)
        return prepared

    def denormalize_actions(self, actions):
        lower, span = self.ranges[ACTION]
        actions = actions.to(dtype=torch.float32)
        return ((actions + 1.0) * span.to(actions.device) / 2.0 + lower.to(actions.device)).cpu()
