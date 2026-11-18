"""Platform boundaries tested with temporary datasets and fake input devices."""

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import subprocess
import sys

import pytest
import torch
from torch.utils.data import default_collate

from test_act_integration import dataset_root, config_for
from test_platform_interfaces import PortableAdapter
from test_evaluation import console
from xyc.contracts import OBS_STATE
from xyc.learning.data_spec import MetricValue, SampleSpec, ValidationSpec
from xyc.learning.dataset import EpisodeDataset
from xyc.learning.dataset_metadata import DatasetMetadata
from xyc.learning.trainer import train
from xyc.models.act.adapter import ACTAdapter
from xyc.runtime.policy_runner import PolicyRunner
from xyc.teleop.keyboard import MobileKeyboard


def test_validation_selects_declared_metric_direction_and_per_metric_weights(monkeypatch, dataset_root, tmp_path):
    values = iter((.2, .8, .3, .9))
    weights = iter((1., 3., 1., 3.))
    monkeypatch.setattr(ACTAdapter, 'validation_spec', staticmethod(lambda: ValidationSpec('success_score', 'max')))
    monkeypatch.setattr(ACTAdapter, 'validation_metrics', lambda self, batch: {
        'success_score': MetricValue(next(values), next(weights)), 'auxiliary': MetricValue(2., 1.)})
    config = replace(config_for(dataset_root, tmp_path / 'run', steps=2), eval_every=1,
                     eval_batches=2, save_every=10)
    result = train(config, progress=lambda _: None)
    assert result['validation_success_score'] == pytest.approx(.75)
    assert result['validation_auxiliary'] == 2.
    best = json.loads((tmp_path / 'run/best_checkpoint.json').read_text())
    assert best['path'] == 'checkpoint-00000002'
    manifest = json.loads((Path(result['checkpoint']) / 'manifest.json').read_text())
    assert manifest['validation_spec'] == {'primary': 'success_score', 'direction': 'max'}


def test_history_sampling_records_the_current_raw_rgb_observation(dataset_root):
    metadata = DatasetMetadata(dataset_root)
    data = EpisodeDataset(metadata, [0], spec=SampleSpec(action_offsets=(0, 1),
                          state_offsets=(-2, 0), image_offsets=(-2, -1)))
    try:
        batch = default_collate([data[3]])
        adapter = PortableAdapter(metadata.schema)
        captured = []
        recorder = SimpleNamespace(schema=metadata.schema, record_video=True,
            event=lambda *a, **k: None, prediction=lambda *a, **k: 1,
            selection=lambda *a, **k: captured.append(k) or 1)
        runner = PolicyRunner(adapter, recorder=recorder)
        assert runner.step(batch).shape == (1, 18)
        for camera in metadata.cameras:
            expected = batch['observation.current_images.' + camera][0].permute(1, 2, 0).numpy()
            assert (captured[0]['images'][camera] == expected).all()
        assert captured[0]['state'].shape == (18,)
    finally:
        data.close()


def test_act_owns_the_same_resize_for_live_and_training_images(dataset_root):
    from test_act_integration import batch_for, TINY
    from xyc.learning.registry import make_policy
    from xyc.images import decode_jpeg
    import cv2
    import numpy as np
    metadata, batch = batch_for(dataset_root)
    options = {**TINY, 'image_size': [32, 48]}
    adapter = make_policy('act', model_config=options, schema=metadata.schema,
                         statistics=metadata.statistics([0], ACTAdapter.statistics_spec(options, metadata.schema)))
    images = {}
    for camera in metadata.cameras:
        rgb = batch['observation.images.' + camera][0].permute(1, 2, 0).numpy()
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        assert ok
        images[camera] = encoded.tobytes()
        batch['observation.images.' + camera] = torch.from_numpy(
            np.ascontiguousarray(decode_jpeg(encoded.tobytes()))).permute(2, 0, 1).unsqueeze(0)
    live = adapter.live_batch({'state': batch[OBS_STATE][0].tolist(), 'images': images})
    prepared, live_prepared = adapter.prepare(batch), adapter.prepare(live)
    for camera in metadata.cameras:
        key = 'observation.images.' + camera
        assert batch[key].shape[-2:] == (64, 64)
        assert prepared[key].shape[-2:] == (32, 48)
        torch.testing.assert_close(prepared[key], live_prepared[key])


def test_evaluation_combines_axes_with_the_shared_keyboard_contract():
    ui, executor, sink, _ = console([['w', 'a', 'j'], ['p']])
    assert ui.prepare(executor)
    assert sink.sent[0]['mobile_command'] == {'lift': 0., 'x': 1., 'y': 1., 'yaw': 1.}


def test_held_keys_continue_together_and_release_without_repeat_delay():
    now = [1.]
    class Keys:
        held = {'w', 'a'}
        pending = ['w', 'a']
        def read(self):
            result, self.pending = self.pending, []
            return result
        def is_held(self, key): return key in self.held
    keys, keyboard = Keys(), MobileKeyboard(clock=lambda: now[0])
    for _ in range(3):
        for key in keyboard.poll(keys): keyboard.press(key)
        assert keyboard.values() == [0., 1., 1., 0.]
        now[0] += .4
    keys.held.clear()
    assert keyboard.poll(keys) == []
    assert keyboard.values() == [0.] * 4


def test_new_direction_and_stop_override_held_key_continuation():
    class Keys:
        held = {'w', 'a', 's'}
        pending = ['s']
        def read(self):
            result, self.pending = self.pending, []
            return result
        def is_held(self, key): return key in self.held
    keys, keyboard = Keys(), MobileKeyboard(clock=lambda: 1.)
    keyboard.press('w')
    keyboard.press('a')
    for key in keyboard.poll(keys): keyboard.press(key)
    assert keyboard.values() == [0., -1., 1., 0.]
    for key in keyboard.poll(keys): keyboard.press(key)
    assert keyboard.values() == [0., -1., 1., 0.]
    keys.held.remove('s')
    for key in keyboard.poll(keys): keyboard.press(key)
    assert keyboard.values() == [0., 0., 1., 0.]
    for stop_key in (' ', 'h', 'q', '\x03'):
        keyboard.press('w')
        keys.pending = [stop_key]
        for key in keyboard.poll(keys): keyboard.press(key)
        assert keyboard.values() == [0.] * 4


@pytest.mark.parametrize('values', [[-1., 0., 1., -1.], [0., 0., 0., 0.], [1., -1., 0., 1.]])
def test_every_mobile_contract_uses_the_same_direction_labels(values):
    from xyc.contracts import checked_action, checked_mobile_action, physical_mobile_action
    assert checked_mobile_action(values) == values
    assert checked_action([0.] * 14 + values)[14:] == values
    assert physical_mobile_action(values) == [value * speed for value, speed in zip(values, (.015, .15, .15, .60))]


@pytest.mark.parametrize('values', [[.5, 0., 0., 0.], [float('nan'), 0., 0., 0.],
                                   [float('inf'), 0., 0., 0.], [True, 0., 0., 0.],
                                   [0., 0., 0.], ['1', 0., 0., 0.]])
def test_strict_mobile_validation_is_shared_while_model_quantization_is_explicit(values):
    from xyc.contracts import checked_mobile_action, physical_mobile_action
    for method in (checked_mobile_action, physical_mobile_action):
        with pytest.raises(ValueError):
            method(values)


def test_recording_assets_get_their_speed_map_from_the_public_contract(tmp_path, monkeypatch):
    import ast
    from xyc import contracts
    from xyc.recording.alignment import RECORDING_CONTRACT
    from xyc.contracts import ACTION_SEMANTICS, ACTION_UNITS, STATE_UNITS
    monkeypatch.setattr(contracts, 'MOBILE_ACTION_SCALE', (.02, .2, .2, .8))
    for folder in ('calibration', 'leader_calibration'):
        (tmp_path / 'config' / folder).mkdir(parents=True)
        for side in ('left', 'right'):
            (tmp_path / 'config' / folder / (side + '.json')).write_text('{}')
    path = Path(__file__).parents[2] / 'ros_ws/src/xyc_recording/xyc_recording/recorder_node.py'
    tree = ast.parse(path.read_text())
    method = next(n for cls in tree.body if isinstance(cls, ast.ClassDef)
                  for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'assets')
    namespace = dict(json=json, ACTION_SEMANTICS=ACTION_SEMANTICS, RECORDING_CONTRACT=RECORDING_CONTRACT,
                     ACTION_UNITS=ACTION_UNITS, STATE_UNITS=STATE_UNITS,
                     mobile_action_metadata=contracts.mobile_action_metadata)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    node = SimpleNamespace(project=tmp_path)
    assert namespace['assets'](node)['mobile_action_scale'] == contracts.mobile_action_metadata()


def test_portable_core_has_no_ros_package_imports_or_ros_source_path_injection():
    root = Path(__file__).resolve().parents[2] / 'src/xyc'
    for path in root.rglob('*.py'):
        source = path.read_text()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(('xyc_hardware', 'xyc_recording', 'xyc_camera', 'rclpy'))
        assert "sys.path.insert" not in source


def test_profile_override_reaches_all_protocols_and_leader_discovery(tmp_path):
    from dataclasses import asdict
    from xyc.robot.profile import robot_profile
    import os
    import yaml
    profile = asdict(robot_profile())
    profile['ports'] = {'teleop': 18882, 'data': 18883, 'motion': 18886, 'recording': 18884}
    profile['leaders']['left'] = 'replacement-usb'
    profile['project_directory'] = str(tmp_path / 'robot')
    path = tmp_path / 'robot.yaml'
    path.write_text(yaml.safe_dump(profile))
    code = '''from xyc.robot.profile import robot_profile
from xyc.transport.teleop import PORT as teleop
from xyc.transport.motion import PORT as motion
from xyc.recording.stream import STREAM_PORT
from xyc.teleop.leader import USB_SERIALS
import sys
assert (teleop, motion, STREAM_PORT) == (18882, 18886, 18884)
assert USB_SERIALS['left'] == 'replacement-usb'
assert str(robot_profile().robot_root) == sys.argv[1]
assert 'torch' not in sys.modules and 'rclpy' not in sys.modules
'''
    result = subprocess.run([sys.executable, '-c', code, profile['project_directory']],
                            env={**os.environ, 'XYC_ROBOT_CONFIG': str(path)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
