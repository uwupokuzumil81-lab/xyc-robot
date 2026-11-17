"""Exercise real ACT computation, v3 videos, normalization and resumable training."""

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from xyc.learning.checkpoint import load_checkpoint
from xyc.learning.config import TrainConfig, load_config
from xyc.learning.contracts import (
    ACTION, ACTION_NAMES, ACTION_UNITS, CAMERAS, OBS_STATE, STATE_NAMES, STATE_UNITS, robot_action,
)
from xyc.learning.dataset import DatasetMetadata, EpisodeDataset, RGBVideos
from xyc.learning.inference import PolicyRunner, predict_dataset, validate_inputs
from xyc.learning.registry import make_policy
from xyc.learning.trainer import StepBatches, train, validate_resume

TINY = {'image_size': [64, 64], 'chunk_size': 4, 'n_action_steps': 2, 'dim_model': 64, 'n_heads': 4,
        'dim_feedforward': 256, 'n_encoder_layers': 1, 'n_decoder_layers': 1, 'n_vae_encoder_layers': 1,
        'pretrained_backbone_weights': None}


@pytest.fixture(scope='module')
def dataset_root(tmp_path_factory):
    root = tmp_path_factory.mktemp('v3')
    (root / 'meta/episodes/chunk-000').mkdir(parents=True)
    (root / 'data/chunk-000').mkdir(parents=True)
    features = {OBS_STATE: {'dtype': 'float32', 'shape': [18], 'names': list(STATE_NAMES)},
                ACTION: {'dtype': 'float32', 'shape': [18], 'names': list(ACTION_NAMES)}}
    for camera in CAMERAS:
        features['observation.images.' + camera] = {'dtype': 'video', 'shape': [64, 64, 3]}
    info = {'codebase_version': 'v3.0', 'fps': 30, 'features': features, 'total_episodes': 2,
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
    (root / 'meta/info.json').write_text(json.dumps(info))
    assets = {'action_units': ACTION_UNITS, 'state_units': STATE_UNITS,
              'calibrations': {'test_identity': 'same'},
              'action_semantics': 'observation_then_new_ternary_command_v4'}
    (root / 'meta/robot.json').write_text(json.dumps(assets))
    for episode in range(2):
        rows = [{OBS_STATE: [float(episode * 100 + frame + d) for d in range(18)],
                 ACTION: [float(frame + d + .5) for d in range(14)]
                         + [float((frame + d) % 3 - 1) for d in range(4)],
                 'episode_index': episode, 'frame_index': frame} for frame in range(6)]
        pq.write_table(pa.Table.from_pylist(rows), root / f'data/chunk-000/file-{episode:03d}.parquet')
        meta = {'episode_index': episode, 'length': 6, 'data/chunk_index': 0, 'data/file_index': episode}
        for i, camera in enumerate(CAMERAS):
            key = 'observation.images.' + camera
            path = root / f'videos/{key}/chunk-000/file-{episode:03d}.mp4'
            path.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(path), 'w') as container:
                stream = container.add_stream('libx264', rate=30)
                stream.width = stream.height = 64
                stream.pix_fmt = 'yuv420p'
                for frame in range(6):
                    rgb = np.full((64, 64, 3), [200 - i * 30, 15 + frame, 30], dtype=np.uint8)
                    for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            prefix = 'videos/' + key
            meta.update({prefix + '/chunk_index': 0, prefix + '/file_index': episode,
                         prefix + '/from_timestamp': 0.})
        pq.write_table(pa.Table.from_pylist([meta]), root / f'meta/episodes/chunk-000/file-{episode:03d}.parquet')
    return root


def config_for(root, output, steps=2):
    return TrainConfig(str(root), str(output), model=TINY.copy(), steps=steps, batch_size=1,
                       save_every=2, eval_every=2, eval_batches=1, log_every=1,
                       device='cpu', train_episodes=[0], validation_episodes=[1], cpu_threads=2)


def batch_for(root):
    metadata = DatasetMetadata(root)
    dataset = EpisodeDataset(metadata, [0], chunk_size=4, image_size=[64, 64])
    try:
        sample = dataset[1]
        return metadata, {key: item.unsqueeze(0) for key, item in sample.items()}
    finally:
        dataset.close()


def test_episode_boundary_and_train_only_stats(dataset_root):
    meta = DatasetMetadata(dataset_root)
    dataset = EpisodeDataset(meta, [0, 1], chunk_size=4, image_size=[64, 64])
    try:
        last = dataset[5]
        assert last['action_is_pad'].tolist() == [False, True, True, True]
        assert torch.equal(last[ACTION][0], last[ACTION][3])
        assert dataset[6][OBS_STATE][0] == 100
        assert meta.statistics([0])[OBS_STATE]['mean'][0] == 2.5
        assert meta.statistics([1])[OBS_STATE]['mean'][0] == 102.5
        assert not set(meta.split()[0]) & set(meta.split()[1])
        assert meta.fingerprint([0]) != meta.fingerprint([1])
        with pytest.raises(ValueError, match='overlapping'):
            meta.split(train_episodes=[0], validation_episodes=[0])
    finally:
        dataset.close()


def test_rgb_camera_order_and_normalization(dataset_root):
    meta, batch = batch_for(dataset_root)
    adapter = make_policy('act', model_config=TINY, schema=meta.schema, statistics=meta.statistics([0]))
    assert tuple(adapter.config.image_features) == tuple('observation.images.' + c for c in CAMERAS)
    rgb = batch['observation.images.forward'][0, :, 0, 0]
    assert rgb[0] > 180 and rgb[2] < 40  # RGB, not OpenCV BGR.
    prepared = adapter.prepare(batch, training=True)
    recovered = prepared[ACTION] * adapter.action_std.clamp_min(adapter.epsilon) + adapter.action_mean
    torch.testing.assert_close(recovered, batch[ACTION])
    broken = dict(batch)
    broken['observation.images.forward'] = broken['observation.images.forward'].float()
    with pytest.raises(ValueError, match='uint8'):
        adapter.prepare(broken)
    values = list(range(14)) + [.6, -.6, .49, -.5]
    named = robot_action(values, meta.schema)
    assert named['mobile_command'] == {'lift': 1., 'x': -1., 'y': 0., 'yaw': -1.}


def test_real_loss_updates_backbone_and_head(dataset_root):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    meta, batch = batch_for(dataset_root)
    adapter = make_policy('act', model_config=TINY, schema=meta.schema, statistics=meta.statistics([0]))
    named = dict(adapter.named_parameters())
    names = ['policy.model.backbone.conv1.weight', 'policy.model.action_head.weight']
    before = {name: named[name].detach().clone() for name in names}
    optimizer = torch.optim.AdamW(adapter.optimizer_groups(learning_rate=1e-4, backbone_learning_rate=1e-4))
    loss, terms = adapter.compute_loss(batch)
    assert torch.isfinite(loss) and set(terms) == {'l1_loss', 'kld_loss'}
    loss.backward()
    assert all(named[name].grad is not None and named[name].grad.abs().sum() > 0 for name in names)
    optimizer.step()
    assert all(not torch.equal(before[name], named[name]) for name in names)
    prediction = adapter.predict_chunk(batch)
    assert prediction.shape == (1, 4, 18) and torch.isfinite(prediction).all()
    assert np.isfinite(adapter.validation_metrics(batch)['action_l1_normalized'])


def test_resume_matches_uninterrupted_and_offline_inference(dataset_root, tmp_path):
    uninterrupted = train(config_for(dataset_root, tmp_path / 'full', 4), progress=lambda _: None)
    first = train(config_for(dataset_root, tmp_path / 'resume', 2), progress=lambda _: None)
    resumed = train(config_for(dataset_root, tmp_path / 'resume', 4), resume=first['checkpoint'], progress=lambda _: None)
    # Reloading must work without ImageNet downloads and without external LeRobot imports.
    with patch('torch.hub.download_url_to_file', side_effect=AssertionError('unexpected download')):
        whole, manifest, _ = load_checkpoint(uninterrupted['checkpoint'])
        restored, _, _ = load_checkpoint(resumed['checkpoint'])
    for name, tensor in whole.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], atol=0, rtol=0)
    meta, batch = batch_for(dataset_root)
    torch.testing.assert_close(whole.predict_chunk(batch), restored.predict_chunk(batch), atol=0, rtol=0)
    output = tmp_path / 'prediction.json'
    result = predict_dataset(tmp_path / 'resume', dataset_root, frame=1, frames=3,
                             output=output, record_dir=tmp_path / 'trace')
    assert result['shape'] == [4, 18] and output.is_file()
    assert result['execution'] == 'offline_prediction_only'
    assert 'first_robot_action' in result
    trace = json.loads((tmp_path / 'trace/meta.json').read_text(encoding='utf-8'))
    assert trace['written']['prediction'] == 2 and trace['written']['selection'] == 3
    assert trace['context']['source']['dataset'] == str(dataset_root)
    assert trace['context']['source']['fingerprint'] == meta.fingerprint([0])
    assert result['record_status'] == 'completed'
    assert predict_dataset(tmp_path / 'resume', dataset_root, record=False)['record_dir'] is None
    with pytest.raises(FileExistsError):
        train(config_for(dataset_root, tmp_path / 'resume'), progress=lambda _: None)
    with pytest.raises(ValueError, match='schema|Camera'):
        validate_inputs(whole, manifest, DatasetMetadata(dataset_root, list(reversed(CAMERAS))))
    meta.assets = {'different': True}
    with pytest.raises(ValueError, match='assets'):
        validate_inputs(whole, manifest, meta)


@pytest.mark.parametrize('coefficient', [None, 0.])
def test_action_queue_and_reset(dataset_root, coefficient):
    meta, batch = batch_for(dataset_root)
    options = {**TINY, 'temporal_ensemble_coeff': coefficient, 'n_action_steps': 2 if coefficient is None else 1}
    adapter = make_policy('act', model_config=options, schema=meta.schema, statistics=meta.statistics([0]))
    runner = PolicyRunner(adapter)
    with patch.object(adapter, 'predict_chunk', wraps=adapter.predict_chunk) as call:
        first = runner.step(batch)
        second = runner.step(batch)
        assert call.call_count == (1 if coefficient is None else 2)
        runner.reset()
        torch.testing.assert_close(first, runner.step(batch))
        assert second.shape == (1, 18)


def test_prefetch_resume_cursor_and_config_guards(dataset_root, tmp_path):
    all_batches = list(StepBatches(6, 3, 0, 10, 42))
    assert all_batches[7:] == list(StepBatches(6, 3, 7, 10, 42))
    config = config_for(dataset_root, tmp_path)
    with pytest.raises(ValueError, match='batch_size'):
        replace(config, batch_size=0).validate()
    with pytest.raises(ValueError, match='learning_rate'):
        validate_resume(replace(config, learning_rate=.01), config.to_dict())


def test_configuration_and_imports_are_local():
    import sys
    from xyc.models.act.modeling_act import ACTPolicy
    assert ACTPolicy.__mro__[1] is torch.nn.Module
    assert not any(key == 'lerobot' or key.startswith('lerobot.') for key in sys.modules)
    config = load_config(Path(__file__).parents[2] / 'config/train/act.yaml')
    assert Path(config.dataset).is_absolute() and config.model['chunk_size'] == 50
