"""Recording must preserve provenance, queue/ensemble lineage and real I/O status."""

from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import av
import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from xyc.contracts import ACTION_NAMES, CAMERAS, OBS_STATE, STATE_NAMES, PolicySchema
from xyc.learning.inference import PolicyRunner, predict_dataset
from xyc.learning.inference_recording import InferenceRecorder
from xyc.models.act.modeling_act import ACTTemporalEnsembler
from xyc.runtime.strategies.base import ChunkExecutionSpec


def schema():
    return PolicySchema(CAMERAS, STATE_NAMES, ACTION_NAMES,
                        ('normalized',) * 14 + ('m', 'm/s', 'm/s', 'rad/s'),
                        ('normalized',) * 14 + ('m/s', 'm/s', 'm/s', 'rad/s'), 30.)


class FakePolicy:
    """Deterministic chunks make stale queue and ensemble lineage bugs visible."""
    def __init__(self, coefficient=None):
        self.schema = schema()
        self.config = SimpleNamespace(chunk_size=4, n_action_steps=2 if coefficient is None else 1,
                                      temporal_ensemble_coeff=coefficient)
        self.calls = 0
        if coefficient is not None:
            self.policy = SimpleNamespace(temporal_ensembler=ACTTemporalEnsembler(coefficient, 4))

    def chunk_execution_spec(self):
        return ChunkExecutionSpec(self.config.n_action_steps, self.config.chunk_size,
                                  self.config.temporal_ensemble_coeff is not None)

    def ensemble_action(self, chunk):
        return self.policy.temporal_ensembler.update(chunk)

    def reset(self):
        if self.config.temporal_ensemble_coeff is not None:
            self.policy.temporal_ensembler.reset()

    def eval(self):
        return self

    def predict_chunk(self, batch):
        self.calls += 1
        return torch.arange(4 * 18).reshape(1, 4, 18).float() + self.calls * 100


def batch():
    value = {OBS_STATE: torch.arange(18).float().reshape(1, 18)}
    for index, camera in enumerate(CAMERAS):
        rgb = torch.tensor([200 - index * 20, 10, 30], dtype=torch.uint8)
        value['observation.images.' + camera] = rgb.reshape(1, 3, 1, 1).expand(1, 3, 64, 64)
    return value


def test_queue_lineage_execution_and_feedback_are_distinct(tmp_path):
    policy = FakePolicy()
    with InferenceRecorder(tmp_path / 'run', schema=schema(), metadata={'checkpoint': 'test'}) as recorder:
        runner = PolicyRunner(policy, recorder=recorder)
        for index in range(3):
            runner.step(batch(), observation={'episode': 2, 'frame': index})
        sent = [4.] * 18  # Intentionally different from the model's selected action.
        runner.record_execution(2, sent_action=sent, accepted=False, command_id='reject-2',
                                source_time_ns=123, source_clock='pi_ros_time', message='target rejected')
        runner.record_feedback(2, state=[3.] * 17 + [float('nan')],
                               source_time_ns=124, source_clock='pi_ros_time')
    predictions = pq.read_table(recorder.path / 'predictions.parquet').to_pylist()
    rows = pq.read_table(recorder.path / 'steps.parquet').to_pylist()
    selected = [r for r in rows if r['record_kind'] == 'selection']
    assert [(r['prediction_id'], r['chunk_index']) for r in selected] == [(1, 0), (1, 1), (2, 0)]
    assert predictions[0]['actions'][1] == selected[1]['selected_action']
    assert all(r['sent_action'] is None and r['driver_accepted'] is None for r in selected)
    execution = next(r for r in rows if r['record_kind'] == 'execution')
    assert execution['step_id'] == 2 and execution['sent_action'] == sent and execution['driver_accepted'] is False
    assert execution['sent_action'] != selected[1]['selected_action']
    feedback = next(r for r in rows if r['record_kind'] == 'feedback')
    assert feedback['feedback_state'][-1] is None
    assert recorder.summary['status'] == 'completed' and recorder.summary['missing'] == {}
    assert recorder.close() == recorder.summary
    with pytest.raises(FileExistsError):
        InferenceRecorder(recorder.path, schema=schema())


def test_temporal_ensemble_lineage_and_reset(tmp_path):
    policy = FakePolicy(0.)
    with InferenceRecorder(tmp_path / 'ensemble', schema=schema()) as recorder:
        runner = PolicyRunner(policy, recorder=recorder)
        for _ in range(5):
            runner.step(batch())
        runner.reset()
        runner.step(batch())
    rows = pq.read_table(recorder.path / 'steps.parquet').to_pylist()
    # At tick 1: prediction 1, index 1 plus prediction 2, index 0.
    np.testing.assert_allclose(rows[1]['selected_action'], (np.arange(18) + 118 + np.arange(18) + 200) / 2)
    assert json.loads(rows[1]['contributors_json']) == [{'prediction_id': 1, 'chunk_index': 1},
                                                        {'prediction_id': 2, 'chunk_index': 0}]
    assert len(json.loads(rows[4]['contributors_json'])) == 4
    assert json.loads(rows[5]['contributors_json']) == [{'prediction_id': 6, 'chunk_index': 0}]
    assert [r['step_id'] for r in rows] == list(range(1, 7))


def test_four_rgb_videos_match_step_indices(tmp_path):
    with InferenceRecorder(tmp_path / 'video', schema=schema(), record_video=True) as recorder:
        runner = PolicyRunner(FakePolicy(), recorder=recorder)
        for index in range(3):
            runner.step(batch(), observation={'frame': index})
    rows = pq.read_table(recorder.path / 'steps.parquet').to_pylist()
    assert [r['video_frame_index'] for r in rows] == [0, 1, 2]
    for camera in CAMERAS:
        with av.open(str(recorder.path / 'videos' / (camera + '.mp4'))) as container:
            images = [f.to_ndarray(format='rgb24') for f in container.decode(video=0)]
            assert len(images) == 3 and images[0][0, 0, 0] > images[0][0, 0, 2] + 60
    assert recorder.summary['video_frames_per_camera'] == 3
    assert not list(recorder.path.rglob('*.partial.*'))


def test_backpressure_is_bounded_and_explicit(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = InferenceRecorder._write_images

    def delayed(self, *args):
        entered.set()
        assert release.wait(5)
        return original(self, *args)

    monkeypatch.setattr(InferenceRecorder, '_write_images', delayed)
    recorder = InferenceRecorder(tmp_path / 'slow', schema=schema(), record_video=True, max_queue=8)
    runner = PolicyRunner(FakePolicy(), recorder=recorder)
    try:
        runner.step(batch())
        assert entered.wait(5)
        for _ in range(24):
            runner.step(batch())
        assert recorder._queue.qsize() <= 8
    finally:
        release.set()
        recorder.close()
    assert recorder.summary['status'] == 'degraded'
    assert sum(recorder.summary['missing'].values()) > 0


def test_writer_error_and_failed_model_keep_honest_status(tmp_path, monkeypatch):
    def disk_failed(*_):
        raise OSError('disk full for test')

    monkeypatch.setattr(InferenceRecorder, '_flush', disk_failed)
    recorder = InferenceRecorder(tmp_path / 'disk', schema=schema(), flush_rows=1)
    runner = PolicyRunner(FakePolicy(), recorder=recorder)
    assert runner.step(batch()).shape == (1, 18)
    recorder.close()
    assert recorder.summary['status'] == 'logging_failed'
    assert 'disk full' in recorder.summary['writer_error']
    assert (recorder.path / 'meta.json').is_file()
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match='test model failure'):
        with InferenceRecorder(tmp_path / 'failed', schema=schema()) as failed:
            PolicyRunner(FakePolicy(), recorder=failed).step(batch())
            raise RuntimeError('test model failure')
    assert json.loads((failed.path / 'meta.json').read_text())['status'] == 'failed'
    assert len(pq.read_table(failed.path / 'predictions.parquet')) == 1


def test_invalid_timestamps_and_path_and_batch_do_not_lie(tmp_path):
    with InferenceRecorder(tmp_path / 'guards', schema=schema()) as recorder:
        runner = PolicyRunner(FakePolicy(), recorder=recorder)
        runner.step(batch())
        with pytest.raises(ValueError, match='trace asset'):
            recorder.check_export_path(recorder.path / 'meta.json')
        with pytest.raises(ValueError, match='clock'):
            runner.record_execution(sent_action=[0.] * 18, source_time_ns=123)
        with pytest.raises(ValueError, match='existing'):
            runner.record_feedback(99, state=[0.] * 18)
        with pytest.raises(ValueError, match='batch size 1'):
            runner.step({OBS_STATE: torch.zeros(2, 18)})
    with pytest.raises(ValueError, match='file names'):
        InferenceRecorder(tmp_path / 'bad', schema=replace(schema(), cameras=('../escape',)), record_video=True)


def test_record_disabled_rejects_video_request():
    with pytest.raises(ValueError, match='record=False'):
        predict_dataset('unused', 'unused', record=False, record_video=True)
