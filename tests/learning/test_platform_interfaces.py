from xyc.assets import follower_calibration_ids
"""Cross-model data requirements and live execution without physical drivers."""

import json
import threading

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from test_act_integration import dataset_root  # Shared real v3 video fixture.
from xyc.contracts import (
    ACTION, ACTION_NAMES, ACTION_UNITS, ARM_NAMES, CAMERAS, OBS_STATE,
    PolicySchema, STATE_NAMES, STATE_UNITS,
)
from xyc.learning.data_spec import SampleSpec, StatisticsSpec
from xyc.learning.dataset_metadata import DatasetMetadata
from xyc.learning.dataset import EpisodeDataset
from xyc.learning import deploy as deployment
from xyc.runtime.live_robot import LiveObservationClient
from xyc.runtime.policy_executor import CommandResult
from xyc.runtime.policy_runner import PolicyRunner
from xyc.runtime.strategies.base import ChunkExecutionSpec
from xyc.observation import ObservationBuffer


def test_adapter_declares_history_language_and_quantiles(dataset_root):
    meta = DatasetMetadata(dataset_root)
    meta.episodes[0].metadata['tasks'] = ['Put the blue block into the brown bowl.']
    request = SampleSpec(action_offsets=(0, 2, 4),
                         state_offsets=(-2, 0), image_offsets=(-1, 0), include_task=True)
    data = EpisodeDataset(meta, [0], spec=request)
    try:
        row = data[0]
        assert row[OBS_STATE].shape == (2, 18)
        assert row[OBS_STATE + '_is_pad'].tolist() == [True, False]
        assert row['observation.images.forward'].shape == (2, 3, 64, 64)
        assert row['observation.images.forward_is_pad'].tolist() == [True, False]
        np.testing.assert_allclose(row[ACTION][:, 0], [.5, 2.5, 4.5])
        assert data[5]['action_is_pad'].tolist() == [False, True, True]
        assert row['task'] == 'Put the blue block into the brown bowl.'
        request = StatisticsSpec({OBS_STATE: ('q01', 'q99'), ACTION: ('min', 'max')})
        stats = meta.statistics([0], request)
        np.testing.assert_allclose(stats[OBS_STATE]['q01'][0], .05)
        np.testing.assert_allclose(stats[OBS_STATE]['q99'][0], 4.95)
        assert stats[ACTION]['max'][0] == 5.5  # Validation episode never participates.
        assert 'images' not in stats and 'mean' not in stats[OBS_STATE]
    finally:
        data.close()


def test_per_frame_task_indices_resolve_pandas_index(dataset_root, tmp_path):
    # Use another root so other tests' shared fixture remains untouched.
    import shutil
    root = tmp_path / 'tasks'
    shutil.copytree(dataset_root, root)
    path = root / 'data/chunk-000/file-000.parquet'
    table = pq.read_table(path).append_column('task_index', pa.array([0, 0, 1, 1, 0, 1]))
    pq.write_table(table, path)
    import pandas as pd
    pd.DataFrame({'task_index': [0, 1]}, index=['pick', 'place']).to_parquet(root / 'meta/tasks.parquet')
    meta = DatasetMetadata(root)
    assert [meta.task_text(0, index) for index in (0, 2, 4, 5)] == ['pick', 'place', 'pick', 'place']


class PortableAdapter:
    """No ACT config or internal .policy object is exposed to shared runtime."""
    name = 'portable-test'

    def __init__(self, schema):
        self.schema, self.calls = schema, 0

    def reset(self):
        pass

    def eval(self):
        return self

    def chunk_execution_spec(self):
        return ChunkExecutionSpec(action_steps=2, horizon=4)

    def sample_spec(self, options, schema):
        return SampleSpec(action_offsets=(0, 1),
                          include_task=options.get('include_task', False))

    def predict_chunk(self, batch):
        self.calls += 1
        return torch.zeros(1, 4, 18)

    def live_batch(self, observation):
        self.observation = observation
        return {OBS_STATE: torch.tensor([observation['state']])}


def schema():
    return PolicySchema(CAMERAS, STATE_NAMES, ACTION_NAMES, STATE_UNITS, ACTION_UNITS, 30.)


def test_shared_queue_only_uses_declared_capability():
    adapter = PortableAdapter(schema())
    runner = PolicyRunner(adapter)
    for _ in range(3):
        assert runner.step({OBS_STATE: torch.zeros(1, 18)}).shape == (1, 18)
    assert adapter.calls == 2


def test_shared_ensemble_uses_adapter_method_without_model_internals():
    adapter = PortableAdapter(schema())
    adapter.chunk_execution_spec = lambda: ChunkExecutionSpec(1, 4, ensemble=True)
    seen = []
    def ensemble(chunk):
        seen.append(chunk)
        return chunk[:, 0]
    adapter.ensemble_action = ensemble
    runner = PolicyRunner(adapter)
    for _ in range(3):
        runner.step({OBS_STATE: torch.zeros(1, 18)})
    assert adapter.calls == len(seen) == 3


def sample(side, stamp, ids):
    index = 0 if side == 'left' else 1
    return {'stamp_ns': stamp, 'valid': True, 'state': [0.] * 7,
            'names': list(ARM_NAMES[index * 7:index * 7 + 7]),
            'state_stamps_ns': [stamp] * 7, 'calibration_id': ids[index], 'session': side + '-session',
            'mobile': {'valid': True, 'state': [.5, 0., 0., 0.]}}


class FakeCameras:
    cameras = CAMERAS

    def __init__(self, *_):
        self.stamp = 1_000_000_000
        ok, jpeg = cv2.imencode('.jpg', np.full((64, 64, 3), (10, 20, 200), dtype=np.uint8))
        assert ok
        self.jpeg = jpeg.tobytes()

    def capture(self):
        self.stamp += 34_000_000
        return dict.fromkeys(CAMERAS, self.jpeg), [self.stamp] * 4, .001


class FakeSink:
    def __init__(self, cameras, ids):
        self.cameras, self.ids = cameras, ids
        self.sent, self.stopped, self.started, self.faulted = [], False, False, False
        self.offset_ns = 0

    def status(self):
        stamp = self.cameras.stamp + self.offset_ns
        return {'ready': True, 'calibration_ids': self.ids, 'state': [0.] * 14 + [.5, 0., 0., 0.],
                'source_time_ns': self.cameras.stamp + 1_000_000,
                'observation_samples': {side: sample(side, stamp, self.ids) for side in ('left', 'right')},
                'chassis': {'homing': '', 'enabled': True}}

    def prepare(self, schema, ids):
        assert ids == self.ids

    def start(self):
        self.started = True
        return self.status()

    def heartbeat(self):
        return self.status()

    def hold(self):
        state = self.status()
        self.sent.append({'arm_position_normalized': state['state'][:14],
                          'mobile_command': dict.fromkeys(('lift', 'x', 'y', 'yaw'), 0.)})
        return state

    def dispatch(self, command, *, sequence, valid_for_s, before_send=None):
        if before_send is not None and not before_send():
            return None
        self.sent.append(command)
        return CommandResult(accepted=True, command_id=str(sequence), feedback_state=self.status()['state'],
                             source_time_ns=123, source_clock='pc_wall', feedback_time_ns=124,
                             feedback_clock='robot_monotonic')

    def stop(self):
        self.stopped = True

    def fault_stop(self):
        self.faulted = self.stopped = True


def live_fixture(monkeypatch):
    adapter = PortableAdapter(schema())
    from xyc.contracts import mobile_action_metadata
    assets = {'calibrations': {'calibration/' + side: {'arm': side} for side in ('left', 'right')},
              'mobile_action_scale': mobile_action_metadata()}
    manifest = {'policy': adapter.name, 'model_config': {}, 'robot_assets': assets, 'statistics': {},
                'weights_sha256': 'test', 'step': 2}
    cameras = FakeCameras()
    sink = FakeSink(cameras, follower_calibration_ids(assets))
    monkeypatch.setattr(deployment, 'load_checkpoint', lambda *a, **kw: (adapter, manifest, 'test-checkpoint'))
    monkeypatch.setattr(deployment, 'LiveCameraClient', lambda *a: cameras)
    monkeypatch.setattr(deployment, 'RobotControlSink', lambda *a: sink)
    return adapter, manifest, cameras, sink


def test_live_alignment_rejects_stale_state_and_camera_generation(monkeypatch):
    _, _, cameras, sink = live_fixture(monkeypatch)
    observations = LiveObservationClient(cameras, sink, sink.ids)
    row = observations.capture()
    assert row['state'][14] == .5 and row['state_stamps_ns'][0] == cameras.stamp
    assert row['camera_skew_s'] == 0 and row['source_clock'] == 'robot_ros'
    cameras.stamp += 1_000_000_000  # Expire the previously valid buffered state.
    sink.offset_ns = -200_000_000
    with pytest.raises(TimeoutError, match='时间未对齐'):
        observations.capture()


def test_live_alignment_does_not_reuse_camera_frames(monkeypatch):
    _, _, cameras, sink = live_fixture(monkeypatch)
    observations = LiveObservationClient(cameras, sink, sink.ids)
    observations.capture()
    frozen = (dict.fromkeys(CAMERAS, cameras.jpeg), [cameras.stamp] * 4, .001)
    monkeypatch.setattr(cameras, 'capture', lambda: frozen)
    with pytest.raises(TimeoutError, match='旧图像不会重复'):
        observations.capture()


def test_observation_end_includes_state_read_completion_time(monkeypatch):
    _, _, cameras, sink = live_fixture(monkeypatch)
    buffer = ObservationBuffer()
    stamp = cameras.stamp
    for name in CAMERAS:
        buffer.add_camera(name, stamp, cameras.jpeg)
    for side in ('left', 'right'):
        state = sample(side, stamp + 5_000_000, sink.ids)
        state['state_stamps_ns'] = [stamp] * 7
        buffer.add_arm(side, state)
    observation = buffer.pick(stamp + 10_000_000)
    assert observation.end_ns == stamp + 5_000_000


def test_live_deploy_records_prediction_sent_command_feedback_and_rgb(monkeypatch, tmp_path):
    from xyc.learning import inference_recording

    adapter, _, _, sink = live_fixture(monkeypatch)
    decode_jpeg = inference_recording.decode_jpeg
    decoding_threads = []

    def decode_in_writer(data):
        decoding_threads.append(threading.get_ident())
        return decode_jpeg(data)

    monkeypatch.setattr(inference_recording, 'decode_jpeg', decode_in_writer)
    result = deployment.deploy('test', interactive=False, device='cpu', steps=3, record_dir=tmp_path / 'trace', record_video=True)
    assert sink.started and sink.stopped and len(sink.sent) == 3
    assert len(adapter.observation['state_stamps_ns']) == 18
    meta = json.loads((tmp_path / 'trace/meta.json').read_text())
    assert meta['status'] == result['record_status'] == 'completed'
    assert meta['written']['prediction'] == 3  # One discarded warmup plus two executed chunks.
    assert all(meta['written'][kind] == 3 for kind in ('selection', 'execution', 'feedback'))
    assert meta['video_frames_per_camera'] == 3
    assert set(meta['video_shapes']) == set(CAMERAS)
    assert len(decoding_threads) == 3 * len(CAMERAS)
    assert threading.get_ident() not in decoding_threads
    rows = pq.read_table(tmp_path / 'trace/steps.parquet').to_pylist()
    selections = [row for row in rows if row['record_kind'] == 'selection']
    assert len(json.loads(selections[0]['observation_json'])['state_stamps_ns']) == 18
    assert all(row['driver_accepted'] for row in rows if row['record_kind'] == 'execution')


def test_deploy_cli_overrides_queue_length_without_changing_prediction_horizon(monkeypatch, tmp_path):
    from xyc.learning.cli import main

    adapter, _, _, sink = live_fixture(monkeypatch)
    trace = tmp_path / 'override'
    assert main(['deploy', '--checkpoint', 'test', '--device', 'cpu', '--auto-start',
                 '--steps', '4', '--action-steps', '3', '--record-dir', str(trace)]) == 0
    assert len(sink.sent) == 4 and sink.stopped
    assert adapter.chunk_execution_spec().action_steps == 2
    meta = json.loads((trace / 'meta.json').read_text())
    assert meta['context']['action_steps_override'] == 3
    predictions = pq.read_table(trace / 'predictions.parquet').to_pylist()
    assert len(predictions) == 3 and all(len(row['actions']) == 4 for row in predictions)
    steps = pq.read_table(trace / 'steps.parquet').to_pylist()
    assert [row['chunk_index'] for row in steps if row['record_kind'] == 'selection'] == [0, 1, 2, 0]


@pytest.mark.parametrize('action_steps,strategy,message', [
    (0, 'sync_chunk', '1 to 4'),
    (5, 'sync_chunk', '1 to 4'),
    (2, 'rtc', 'requires --strategy sync_chunk'),
])
def test_invalid_live_action_steps_fail_before_connect(monkeypatch, action_steps, strategy, message):
    live_fixture(monkeypatch)

    def unexpected_connection(*args):
        pytest.fail('Invalid execution steps tried to connect to the robot')

    monkeypatch.setattr(deployment, 'RobotControlSink', unexpected_connection)
    with pytest.raises(ValueError, match=message):
        deployment.deploy('test', interactive=False, device='cpu', record=False,
                          action_steps=action_steps, strategy=strategy)


def test_live_failure_stops_sink_and_finalizes_trace(monkeypatch, tmp_path):
    _, _, _, sink = live_fixture(monkeypatch)
    sink.offset_ns = -200_000_000
    with pytest.raises(TimeoutError, match='alignment'):
        deployment.deploy('test', interactive=False, device='cpu', steps=1, record_dir=tmp_path / 'failed')
    assert sink.stopped and not sink.sent
    meta = json.loads((tmp_path / 'failed/meta.json').read_text())
    assert meta['status'] == 'failed' and meta['writer_error'] is None


def test_live_rtc_worker_error_stops_shared_sink(monkeypatch, tmp_path):
    from xyc.runtime.strategies.base import RTCChunk

    adapter, _, _, sink = live_fixture(monkeypatch)
    adapter.chunk_execution_spec = lambda: ChunkExecutionSpec(2, 12)

    def predict_rtc(batch, *, previous, **kwargs):
        if previous is not None:
            raise RuntimeError('failed RTC inference')
        actions = torch.zeros(1, 12, 18)
        return RTCChunk(actions, actions)

    adapter.predict_rtc_chunk = predict_rtc
    trace = tmp_path / 'rtc-error'
    with pytest.raises(RuntimeError, match='failed RTC inference'):
        deployment.deploy('test', interactive=False, device='cpu', strategy='rtc',
                          steps=10, record_dir=trace)
    assert sink.started and sink.faulted and 2 <= len(sink.sent) < 10
    assert all(command['observation_time_ns'] > 0 for command in sink.sent)
    meta = json.loads((trace / 'meta.json').read_text())
    assert meta['status'] == 'failed' and meta['context']['strategy'] == 'rtc'
    assert meta['written']['prediction'] == 1


def test_live_rtc_dispatches_while_next_model_call_waits(monkeypatch, tmp_path):
    from xyc.runtime.strategies.base import RTCChunk

    adapter, _, _, sink = live_fixture(monkeypatch)
    adapter.chunk_execution_spec = lambda: ChunkExecutionSpec(2, 24)
    release = threading.Event()
    dispatch = sink.dispatch
    original_indices = torch.arange(24).float()

    guidance = []

    def predict_rtc(batch, *, previous, execution_horizon, **kwargs):
        values = original_indices
        if previous is not None:
            guidance.append(execution_horizon)
            if len(guidance) == 1:
                assert release.wait(3)
                values = values + 50  # This candidate missed the guided prefix.
            else:
                values = values + 75
        actions = torch.zeros(1, 24, 18)
        actions[0, :, 0] = values
        return RTCChunk(actions, actions)

    def dispatch_while_pending(command, *, sequence, valid_for_s, before_send=None):
        assert 0 < valid_for_s <= .25
        result = dispatch(command, sequence=sequence, valid_for_s=valid_for_s, before_send=before_send)
        if len(sink.sent) == 12:
            release.set()
        return result

    adapter.predict_rtc_chunk = predict_rtc
    monkeypatch.setattr(sink, 'dispatch', dispatch_while_pending)
    trace = tmp_path / 'rtc-overlap'
    try:
        result = deployment.deploy('test', interactive=False, device='cpu', strategy='rtc',
                                   steps=16, record_dir=trace)
    finally:
        release.set()
    assert sink.stopped and result['frames'] == 16
    sent = [command['arm_position_normalized'][0] for command in sink.sent]
    assert sent[:12] == list(range(12))
    assert all(value < 24 or value >= 75 for value in sent)
    first_new = next(i for i, value in enumerate(sent) if value >= 75)
    assert guidance[0] == 3 and guidance[1] > guidance[0]
    rows = pq.read_table(trace / 'steps.parquet').to_pylist()
    selected = [row for row in rows if row['record_kind'] == 'selection']
    assert selected[first_new]['chunk_index'] == sent[first_new] - 75
    assert selected[first_new]['prediction_id'] != selected[0]['prediction_id']


def test_language_requirement_and_trace_directory_checked_before_connect(monkeypatch, tmp_path):
    _, manifest, _, sink = live_fixture(monkeypatch)
    manifest['model_config'] = {'include_task': True}
    with pytest.raises(ValueError, match='--task'):
        deployment.deploy('test', interactive=False, device='cpu', steps=1, record=False)
    assert not sink.started
    result = deployment.deploy('test', interactive=False, device='cpu', steps=1, task='pick', record=False)
    assert result['record_dir'] is None
    with pytest.raises(ValueError, match='record=False'):
        deployment.deploy('test', interactive=False, record=False, record_video=True)


def test_existing_trace_is_not_overwritten_or_followed_by_robot_start(monkeypatch, tmp_path):
    _, _, _, sink = live_fixture(monkeypatch)
    trace = tmp_path / 'existing'
    trace.mkdir()
    with pytest.raises(FileExistsError):
        deployment.deploy('test', interactive=False, device='cpu', steps=1, record_dir=trace)
    assert not sink.started and not sink.sent


def test_live_interrupt_stops_and_preserves_unknown_dispatch_outcome(monkeypatch, tmp_path):
    _, _, _, sink = live_fixture(monkeypatch)
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()
    monkeypatch.setattr(sink, 'dispatch', interrupt)
    with pytest.raises(KeyboardInterrupt):
        deployment.deploy('test', interactive=False, device='cpu', steps=1, record_dir=tmp_path / 'interrupted')
    assert sink.stopped
    meta = json.loads((tmp_path / 'interrupted/meta.json').read_text())
    assert meta['status'] == 'interrupted'
    rows = pq.read_table(tmp_path / 'interrupted/steps.parquet').to_pylist()
    execution = next(row for row in rows if row['record_kind'] == 'execution')
    assert execution['driver_accepted'] is None  # No reply does not establish rejection.
