"""Regression checks for deadlines, crash recovery and portable entry points."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from test_act_integration import config_for, dataset_root, TINY
from test_platform_interfaces import live_fixture
from xyc.learning import deploy as deployment
from xyc.learning import trainer
from xyc.learning.config import TrainConfig
from xyc.locking import acquire_file_lock


def test_slow_prediction_is_never_dispatched(monkeypatch, tmp_path):
    adapter, _, _, sink = live_fixture(monkeypatch)
    clock = [10.]
    predict = adapter.predict_chunk
    def slow(batch):
        if adapter.calls:
            clock[0] += .4  # Warmup is discarded; delay the first actual control prediction.
        return predict(batch)
    monkeypatch.setattr(adapter, 'predict_chunk', slow)
    monkeypatch.setattr(deployment.time, 'monotonic', lambda: clock[0])
    with pytest.raises(TimeoutError, match='before dispatch'):
        deployment.deploy('test', interactive=False, device='cpu', steps=1,
                          record_dir=tmp_path / 'trace')
    assert sink.sent == [] and sink.stopped


def test_inspect_does_not_import_torch_or_ros(dataset_root):
    code = ('import sys; from xyc.learning.cli import main; '
            'main(["inspect", "--dataset", sys.argv[1]]); '
            'assert "torch" not in sys.modules and "rclpy" not in sys.modules')
    result = subprocess.run([sys.executable, '-c', code, str(dataset_root)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_training_and_dataset_locks_survive_a_crashed_owner(tmp_path):
    path = tmp_path / '.training.lock'
    code = 'import os,sys; from xyc.locking import acquire_file_lock; acquire_file_lock(sys.argv[1]); os._exit(0)'
    result = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert path.exists()
    descriptor = acquire_file_lock(path)
    try:
        denied = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True)
        assert denied.returncode != 0
    finally:
        os.close(descriptor)
    os.close(acquire_file_lock(path))


def test_common_trainer_uses_adapter_export_without_act_config(monkeypatch, dataset_root, tmp_path):
    from xyc.models.act.adapter import ACTAdapter
    original = trainer.make_policy
    class PortablePolicy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.statistics = inner.statistics
        def model_config(self): return {'portable_option': 17}
        def optimizer_groups(self, **options): return self.inner.optimizer_groups(**options)
        def sample_spec(self, *args): return self.inner.sample_spec(TINY, self.inner.schema)
        def statistics_spec(self, *args): return ACTAdapter.statistics_spec(TINY, self.inner.schema)
        def validation_spec(self): return ACTAdapter.validation_spec()
        def compute_loss(self, batch): return self.inner.compute_loss(batch)
    monkeypatch.setattr(trainer, 'make_policy', lambda *args, **kwargs: PortablePolicy(original(*args, **kwargs)))
    config = TrainConfig(dataset=str(dataset_root), output_dir=str(tmp_path / 'run'), model=TINY,
                         steps=1, batch_size=1, device='cpu', validation_fraction=0.)
    result = trainer.train(config, progress=lambda _: None)
    import json
    manifest = json.loads((Path(result['checkpoint']) / 'manifest.json').read_text())
    assert manifest['model_config'] == {'portable_option': 17}


def test_target_expires_before_send_if_another_rpc_blocks_the_control_link(monkeypatch):
    import threading
    from unittest.mock import Mock
    from xyc.transport import motion
    clock = [0.]
    class DelayedLock:
        def __enter__(self): clock[0] += .4
        def __exit__(self, *args): pass
    client = object.__new__(motion.MotionClient)
    client._io_lock, client._closed, client._transport_error = DelayedLock(), False, None
    client._rejection = None
    client._stop_keepalive, client.sequence = threading.Event(), 0
    send = Mock()
    monkeypatch.setattr(motion, 'send', send)
    monkeypatch.setattr(motion.time, 'monotonic', lambda: clock[0])
    with pytest.raises(TimeoutError, match='no target was sent'):
        client.call('target', action=[0.] * 18, valid_for_s=.25)
    send.assert_not_called()
    assert client.sequence == 0


def test_live_control_client_does_not_load_offline_replay_or_dataset_dependencies():
    code = """
import sys
from unittest.mock import MagicMock, patch
from xyc.transport.motion import MotionClient
assert not any(name in sys.modules for name in ('numpy', 'pyarrow', 'av', 'torch'))

from xyc.runtime.live_robot import RobotControlSink

with patch('xyc.transport.motion.socket.create_connection', return_value=MagicMock()):
    sink = RobotControlSink()
    try:
        assert type(sink.client) is MotionClient
        assert 'xyc.learning.replay' not in sys.modules
        assert 'xyc.learning.dataset_metadata' not in sys.modules
        assert not any(name in sys.modules for name in ('pyarrow', 'av'))
    finally:
        sink.stop()
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('index_name', ['last_checkpoint.json', 'best_checkpoint.json'])
def test_resume_uses_committed_checkpoint_after_index_write_failure(monkeypatch, tmp_path, index_name):
    import json
    from xyc.learning import checkpoint
    root = tmp_path / 'run'
    root.mkdir()
    model = torch.nn.Linear(2, 2)
    def manifest(step):
        return {'step': step, 'validation_spec': {'primary': 'score', 'direction': 'max'},
                'validation_score': float(step)}
    first = checkpoint.save_checkpoint(root, model, manifest(1), {})
    original = checkpoint.write_json
    def fail_index(path, value):
        if Path(path).name == index_name:
            raise OSError('injected checkpoint index failure')
        return original(path, value)
    monkeypatch.setattr(checkpoint, 'write_json', fail_index)
    with pytest.raises(OSError, match='injected'):
        checkpoint.save_checkpoint(root, model, manifest(2), {})
    assert checkpoint.resolve_checkpoint(root).name == 'checkpoint-00000002'
    assert checkpoint.resolve_checkpoint(first) == first
    # Stale, missing and damaged indexes all leave the committed directory authoritative.
    (root / 'last_checkpoint.json').write_text(json.dumps({'path': first.name}))
    assert checkpoint.resolve_checkpoint(root).name == 'checkpoint-00000002'
    (root / 'last_checkpoint.json').unlink()
    assert checkpoint.resolve_checkpoint(root).name == 'checkpoint-00000002'
    (root / 'last_checkpoint.json').write_text('{')
    assert checkpoint.resolve_checkpoint(root).name == 'checkpoint-00000002'


def test_best_uses_act_validation_definition_for_existing_manifests(tmp_path):
    from xyc.learning import checkpoint
    model = torch.nn.Linear(2, 2)
    for step, score in enumerate((.3, .1, .2), 1):
        checkpoint.save_checkpoint(tmp_path, model, {
            'step': step, 'policy': 'act', 'validation_action_l1_normalized': score}, {})
    (tmp_path / 'best_checkpoint.json').unlink()
    assert checkpoint.resolve_checkpoint(tmp_path, selection='best').name == 'checkpoint-00000002'
    assert checkpoint.resolve_checkpoint(tmp_path).name == 'checkpoint-00000003'


def test_checkpoint_discovery_ignores_staging_but_never_hides_a_damaged_commit(tmp_path):
    from xyc.learning import checkpoint
    model = torch.nn.Linear(2, 2)
    first = checkpoint.save_checkpoint(tmp_path, model, {'step': 1}, {})
    stage = tmp_path / '.checkpoint-interrupted'
    stage.mkdir()
    (stage / 'manifest.json').write_text('{')
    assert checkpoint.resolve_checkpoint(tmp_path) == first
    second = checkpoint.save_checkpoint(tmp_path, model, {'step': 2}, {})
    with pytest.raises(FileExistsError, match='overwrite'):
        checkpoint.save_checkpoint(tmp_path, model, {'step': 2}, {})
    (second / 'training.pt').unlink()
    with pytest.raises(ValueError, match='Incomplete'):
        checkpoint.resolve_checkpoint(tmp_path)


def test_training_resumes_after_committed_checkpoint_save_is_interrupted(monkeypatch, dataset_root, tmp_path):
    from xyc.learning import checkpoint
    expected = trainer.train(config_for(dataset_root, tmp_path / 'whole', 4), progress=lambda _: None)
    run = tmp_path / 'interrupted'
    original = checkpoint.write_json
    def fail_index(path, value):
        if Path(path) == run / 'last_checkpoint.json':
            raise OSError('injected interruption after directory commit')
        return original(path, value)
    with monkeypatch.context() as change:
        change.setattr(checkpoint, 'write_json', fail_index)
        with pytest.raises(OSError, match='injected interruption'):
            trainer.train(config_for(dataset_root, run, 2), progress=lambda _: None)
    resumed = trainer.train(config_for(dataset_root, run, 4), resume=run, progress=lambda _: None)
    expected_model, _, _ = checkpoint.load_checkpoint(expected['checkpoint'])
    resumed_model, _, _ = checkpoint.load_checkpoint(resumed['checkpoint'])
    for name, value in expected_model.state_dict().items():
        torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize('direction,scores', [('min', [2., 1., 2., 3.]), ('max', [1., 2., 1., 0.])])
def test_resume_recovers_the_best_index_even_if_no_later_score_improves(
        monkeypatch, dataset_root, tmp_path, direction, scores):
    import json
    from dataclasses import replace
    from xyc.learning import checkpoint
    from xyc.learning.data_spec import MetricValue, ValidationSpec
    from xyc.models.act.adapter import ACTAdapter
    spec = ValidationSpec('score', direction)
    values = iter(scores)
    monkeypatch.setattr(ACTAdapter, 'validation_spec', staticmethod(lambda: spec))
    monkeypatch.setattr(ACTAdapter, 'validation_metrics', lambda self, batch: {'score': MetricValue(next(values), 1.)})
    run = tmp_path / 'run'
    config = replace(config_for(dataset_root, run, 2), save_every=1, eval_every=1)
    original = checkpoint.write_json
    def fail_best(path, value):
        if Path(path).name == 'best_checkpoint.json' and value['path'] == 'checkpoint-00000002':
            raise OSError('injected best index interruption')
        return original(path, value)
    with monkeypatch.context() as change:
        change.setattr(checkpoint, 'write_json', fail_best)
        with pytest.raises(OSError, match='injected best'):
            trainer.train(config, progress=lambda _: None)
    assert checkpoint.resolve_checkpoint(run, selection='best').name == 'checkpoint-00000002'
    from xyc.learning.inference import predict_dataset
    assert predict_dataset(run, dataset_root, checkpoint_selection='best', record=False)['training_step'] == 2
    assert json.loads((run / 'best_checkpoint.json').read_text())['path'] == 'checkpoint-00000001'
    trainer.train(replace(config, steps=4), resume=run, progress=lambda _: None)
    assert json.loads((run / 'best_checkpoint.json').read_text())['path'] == 'checkpoint-00000002'
    # Index recovery does not execute a training step or depend on another improvement.
    (run / 'best_checkpoint.json').write_text('{')
    checkpoint.refresh_checkpoint_indexes(run, spec)
    assert json.loads((run / 'best_checkpoint.json').read_text())['path'] == 'checkpoint-00000002'
    assert checkpoint.resolve_checkpoint(run, selection='best').name == 'checkpoint-00000002'
