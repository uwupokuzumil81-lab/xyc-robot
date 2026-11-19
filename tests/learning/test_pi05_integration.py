"""Exercise vendored π0.5 through its offline smoke architecture."""

import json

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import default_collate

from test_act_integration import dataset_root
from xyc.contracts import ACTION, CAMERAS, OBS_IMAGES, OBS_STATE
from xyc.learning.checkpoint import load_checkpoint
from xyc.learning.config import TrainConfig
from xyc.learning.dataset import EpisodeDataset
from xyc.learning.dataset_metadata import DatasetMetadata
from xyc.learning.inference import predict_dataset
from xyc.learning.registry import make_policy, policy_adapter
from xyc.learning.trainer import train


@pytest.fixture
def tiny_pi05():
    pytest.importorskip('transformers')
    return policy_adapter('pi05').training_options({}, smoke=True)


@pytest.mark.parametrize('dtype', ('float32', 'bfloat16'))
def test_pi05_train_restore_and_predict(dataset_root, tmp_path, tiny_pi05, dtype):
    config = TrainConfig(str(dataset_root), str(tmp_path / 'run'), policy='pi05',
                         model={**tiny_pi05, 'dtype': dtype},
                         device='cpu', cpu_threads=2, steps=2, batch_size=1, eval_batches=1,
                         train_episodes=[0], validation_episodes=[1])
    first = train(config, progress=lambda _: None)
    config.steps = 3
    result = train(config, resume=first['checkpoint'], progress=lambda _: None)
    adapter, manifest, _ = load_checkpoint(result['checkpoint'])
    assert manifest['policy'] == 'pi05' and manifest['sample_spec']['include_task']
    assert manifest['statistics'][OBS_STATE]['q99'][0] < 6  # Train episode only.
    assert manifest['validation_action_l1_normalized'] >= 0
    assert tuple(adapter.policy.config.image_features) == tuple(f'{OBS_IMAGES}.{name}' for name in CAMERAS)
    assert adapter.policy.model.action_out_proj.out_features == 32
    paligemma = adapter.policy.model.paligemma_with_expert.paligemma
    assert paligemma.lm_head.weight is paligemma.model.language_model.embed_tokens.weight
    prediction = predict_dataset(result['checkpoint'], dataset_root, frames=2, record=False)
    assert prediction['shape'] == [4, 18]
    assert prediction['replayed_frames'] == 2


def test_pi05_masked_flow_and_live_preprocessing(dataset_root, tiny_pi05, monkeypatch):
    from xyc.models.pi05.processor_pi05 import LANGUAGE_MASK, LANGUAGE_TOKENS

    metadata = DatasetMetadata(dataset_root)
    factory = policy_adapter('pi05')
    stats = metadata.statistics([0], factory.statistics_spec(tiny_pi05, metadata.schema))
    for name in ('min', 'max', 'q01', 'q99'):
        stats[ACTION][name][-1] = 0.  # A constant direction label still uses the canonical range.
    adapter = make_policy('pi05', model_config=tiny_pi05, schema=metadata.schema, statistics=stats)
    data = EpisodeDataset(metadata, [0], spec=factory.sample_spec(tiny_pi05, metadata.schema))
    try:
        batch = default_collate([data[5]])
    finally:
        data.close()
    assert batch['action_is_pad'].tolist() == [[False, True, True, True]]
    loss_grid = torch.full((1, 4, 32), 10_000.)
    loss_grid[0, 0, :18] = 2.

    prepared = adapter.prepare(batch, training=True)
    assert prepared['task'][0].startswith('Task: Pick up the block., State: ')
    assert prepared['task'][0].endswith(';\nAction: ')
    assert prepared[LANGUAGE_TOKENS].shape == (1, 200)
    assert prepared[LANGUAGE_MASK].dtype == torch.bool
    assert prepared[ACTION][..., -1].equal(batch[ACTION][..., -1])

    def inspect_flow(images, masks, tokens, token_mask, actions):
        assert all(image.shape == (1, 3, 28, 28) for image in images)
        assert all(mask.all() for mask in masks)
        assert actions.shape == (1, 4, 32) and torch.count_nonzero(actions[..., 18:]) == 0
        return loss_grid

    monkeypatch.setattr(adapter.policy.model, 'forward', inspect_flow)
    assert adapter.compute_loss(batch)[0] == 2.

    jpeg = cv2.imencode('.jpg', np.full((48, 64, 3), 128, dtype=np.uint8))[1].tobytes()
    sample = adapter.live_batch({'state': batch[OBS_STATE][0].tolist(),
                                 'images': {name: jpeg for name in CAMERAS}, 'task': 'Place the block.'})
    prepared = adapter.prepare(sample)
    assert sample['task'] == ['Place the block.']
    assert all(prepared[f'{OBS_IMAGES}.{name}'].dtype == torch.float32 for name in CAMERAS)
    pixels = prepared[f'{OBS_IMAGES}.forward']
    assert pixels.shape == (1, 3, 28, 28) and (pixels[..., 0, :] == -1.).all()
    torch.testing.assert_close(pixels[..., 14, 14], torch.full((1, 3), 2 * 128. / 255 - 1))
    assert torch.isfinite(adapter.predict_chunk(sample)).all()
    sample.pop('task')
    with pytest.raises(ValueError, match='task instruction'):
        adapter.predict_chunk(sample)


def test_pi05_pretrained_weights_must_load_strictly(dataset_root, tiny_pi05, tmp_path):
    from safetensors.torch import save_file
    from xyc.models.pi05.backend import _load_base, _pretrained_weights

    metadata = DatasetMetadata(dataset_root)
    factory = policy_adapter('pi05')
    stats = metadata.statistics([0], factory.statistics_spec(tiny_pi05, metadata.schema))
    with pytest.raises(ValueError, match='weights not found'):
        _pretrained_weights(str(tmp_path), None)
    save_file({'wrong.weight': torch.zeros(1)}, str(tmp_path / 'model.safetensors'))
    weights = _pretrained_weights(str(tmp_path), None)
    restored = make_policy('pi05', model_config=tiny_pi05, schema=metadata.schema, statistics=stats)
    with pytest.raises(RuntimeError, match='Missing key'):
        _load_base(restored.policy, weights)
    reference = make_policy('pi05', model_config=tiny_pi05, schema=metadata.schema, statistics=stats)
    state = {name.removeprefix('model.'): value for name, value in reference.policy.state_dict().items()}
    embedding = 'paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight'
    state.pop(embedding)
    save_file(state, str(tmp_path / 'model.safetensors'),
              metadata={embedding: 'paligemma_with_expert.paligemma.lm_head.weight'})
    _load_base(restored.policy, weights)
    for name, parameter in restored.policy.state_dict().items():
        torch.testing.assert_close(parameter, reference.policy.state_dict()[name])


def test_pi05_validation_is_repeatable_and_does_not_advance_training_rng(dataset_root, tiny_pi05):
    metadata = DatasetMetadata(dataset_root)
    factory = policy_adapter('pi05')
    stats = metadata.statistics([0], factory.statistics_spec(tiny_pi05, metadata.schema))
    adapter = make_policy('pi05', model_config=tiny_pi05, schema=metadata.schema, statistics=stats)
    data = EpisodeDataset(metadata, [0], spec=factory.sample_spec(tiny_pi05, metadata.schema))
    try:
        batch = default_collate([data[0], data[1]])
    finally:
        data.close()
    first_rng = torch.get_rng_state()
    first = adapter.validation_metrics(batch)
    assert torch.equal(first_rng, torch.get_rng_state())
    torch.rand(3)
    next_rng = torch.get_rng_state()
    assert adapter.validation_metrics(batch) == first
    assert torch.equal(next_rng, torch.get_rng_state())


def test_pi05_smoke_cli_trains_reloads_and_predicts_offline(dataset_root, tmp_path, tiny_pi05, monkeypatch):
    from transformers import AutoTokenizer
    from xyc.learning.cli import main
    from xyc.paths import project_root

    def forbid_download(*args, **kwargs):
        pytest.fail('π0.5 smoke must use its local tokenizer')

    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', forbid_download)
    output = tmp_path / 'smoke'
    assert main(['smoke', '--config', str(project_root() / 'config/train/pi05.yaml'),
                 '--dataset', str(dataset_root), '--output', str(output)]) == 0
    adapter, manifest, _ = load_checkpoint(output)
    assert manifest['step'] == 2 and manifest['train_config']['warmup_steps'] == 0
    assert manifest['model_config']['smoke'] is True
    assert sum(p.numel() for p in adapter.parameters()) < 1_000_000
    assert json.loads((output / 'prediction.json').read_text())['shape'] == [4, 18]


def test_pi05_preserves_state_bins_and_linear_rtc_prefix():
    from xyc.models.pi05.processor_pi05 import state_prompts
    from xyc.models.pi05.rtc_guidance import guided_denoise_step

    state = torch.tensor([[-1.1, -1., 0., 1., 1.1]])
    assert state_prompts(state, [' Pick_up\nblock ']) == ['Task: Pick up block, State: -1 0 128 255 255;\nAction: ']
    latent, previous = torch.zeros(1, 4, 32), torch.ones(1, 3, 18)
    with torch.no_grad():
        guided = guided_denoise_step(latent, previous, 1, 3, 1., torch.zeros_like)
    torch.testing.assert_close(guided[0, :, 0], torch.tensor([-10., -20. / 3, -10. / 3, 0.]))
    assert guided[..., 18:].count_nonzero() == 0


@pytest.mark.parametrize('dtype', ('float32', 'bfloat16'))
def test_pi05_rtc_runs_local_guided_denoising(dataset_root, tiny_pi05, dtype):
    from xyc.runtime.policy_runner import PolicyRunner

    metadata = DatasetMetadata(dataset_root)
    factory = policy_adapter('pi05')
    stats = metadata.statistics([0], factory.statistics_spec(tiny_pi05, metadata.schema))
    adapter = make_policy('pi05', model_config={**tiny_pi05, 'dtype': dtype},
                          schema=metadata.schema, statistics=stats)
    data = EpisodeDataset(metadata, [0], spec=factory.sample_spec(tiny_pi05, metadata.schema))
    try:
        batch = default_collate([data[0]])
    finally:
        data.close()
    runner = PolicyRunner(adapter, strategy='rtc')
    try:
        assert runner.step(batch) is None
        first = runner.strategy.pending.future.result(timeout=5)
        assert first.chunk.shape == (1, 4, 18)
        torch.testing.assert_close(first.chunk, adapter.processor.denormalize_actions(first.conditioning))
        prefix = torch.ones(1, 3, 18)
        torch.manual_seed(123)
        plain = runner._compute(batch, method='predict_rtc_chunk', previous=None,
                                inference_delay=0, execution_horizon=0)
        torch.manual_seed(123)
        guided = runner._compute(batch, method='predict_rtc_chunk', previous=prefix,
                                 inference_delay=1, execution_horizon=3)
        assert not torch.allclose(plain.conditioning[:, :3], guided.conditioning[:, :3])
    finally:
        runner.close()
