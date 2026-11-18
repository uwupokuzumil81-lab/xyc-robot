"""Checkpoint-based action prediction; no serial, ROS or motor writes."""

from pathlib import Path
from torch.utils.data import default_collate

from .checkpoint import load_checkpoint
from xyc.io import write_json
from xyc.paths import project_root
from xyc.contracts import ACTION_NAMES, robot_action
from .dataset_metadata import DatasetMetadata
from .dataset import EpisodeDataset
from xyc.runtime.policy_runner import PolicyRunner


def validate_inputs(adapter, manifest, metadata):
    if metadata.schema != adapter.schema:
        raise ValueError('Camera order, state/action names, units or FPS differ from training')
    if manifest['robot_assets'] != metadata.assets:
        raise ValueError('Robot calibration/assets differ from training; explicitly adapt them before deployment')


def predict_dataset(checkpoint, dataset, *, episode=0, frame=0, device='cpu', output=None,
                    frames=1, record=True, record_dir=None, record_video=False, checkpoint_selection='latest'):
    if type(frames) is not int or frames < 1 or frame < 0:
        raise ValueError('frames must be positive and frame must be non-negative')
    if not record and (record_dir is not None or record_video):
        raise ValueError('Recording directory/video cannot be combined with record=False')
    adapter, manifest, path = load_checkpoint(checkpoint, device=device, selection=checkpoint_selection)
    metadata = DatasetMetadata(dataset, adapter.schema.cameras)
    validate_inputs(adapter, manifest, metadata)
    data = EpisodeDataset(metadata, [episode], spec=adapter.sample_spec(manifest['model_config'], metadata.schema))
    if frame + frames > len(data):
        raise ValueError('Offline replay cannot cross the end of its episode')
    recorder = None
    error = None
    try:
        if record:
            from .inference_recording import InferenceRecorder
            context = {'mode': 'offline_replay', 'execution': 'offline_prediction_only',
                       'policy': manifest['policy'], 'checkpoint': str(path),
                       'weights_sha256': manifest['weights_sha256'], 'training_step': manifest['step'],
                       'model_config': manifest['model_config'], 'statistics': manifest['statistics'],
                       'robot_assets': manifest['robot_assets'], 'device': str(device),
                       'tasks': metadata.episodes[episode].metadata.get('tasks', []),
                       'source': {'dataset': str(metadata.root), 'episode': episode, 'start_frame': frame,
                                  'frames': frames, 'fingerprint': metadata.fingerprint([episode]),
                                  'videos': {name: {'path': str(metadata.video(episode, name)[0]),
                                                    'from_timestamp': metadata.video(episode, name)[1]}
                                             for name in metadata.cameras}}}
            options = dict(schema=adapter.schema, metadata=context, record_video=record_video)
            recorder = (InferenceRecorder(record_dir, **options) if record_dir is not None else
                        InferenceRecorder.create(project_root() / 'outputs/inference', **options))
            if output is not None:
                recorder.check_export_path(output)
        runner = PolicyRunner(adapter, recorder=recorder)
        first_prediction = None
        for index in range(frame, frame + frames):
            sample = data[index]
            batch = default_collate([sample])
            runner.step(batch, observation={'episode': episode, 'frame': index,
                        'source_time_ns': round(index / adapter.schema.fps * 1e9),
                        'source_clock': 'dataset_nominal_video_time'})
            if first_prediction is None:
                first_prediction = runner.last_prediction[0].cpu().tolist()
        actions = first_prediction
        result = {'policy': manifest['policy'], 'checkpoint': str(path), 'training_step': manifest['step'],
                  'episode': episode, 'frame': frame, 'fps': adapter.schema.fps,
                  'action_names': list(adapter.schema.action_names), 'action_units': list(adapter.schema.action_units),
                  'shape': [len(actions), len(actions[0])], 'actions': actions,
                  'execution': 'offline_prediction_only', 'replayed_frames': frames,
                  'actions_scope': 'first_prediction_chunk',
                  'record_dir': str(recorder.path) if recorder is not None else None}
        if adapter.schema.action_names == ACTION_NAMES:
            result['first_robot_action'] = robot_action(actions[0], adapter.schema, bound_arms=True)
        if recorder is not None:
            result['record_status'] = recorder.close()['status']
        if output:
            output = Path(output).expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(output, result)
        return result
    except BaseException as exc:
        error = exc
        if recorder is not None and not recorder.closed:
            recorder.event('inference_error', error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        data.close()
        if recorder is not None:
            recorder.close(outcome='failed' if error is not None else 'completed',
                           reason=str(error) if error is not None else None)
