"""Run a trained policy against the unified real-robot controller."""

from xyc.robot.profile import robot_profile
from xyc.assets import follower_calibration_ids

from pathlib import Path
import sys
import time

from .checkpoint import load_checkpoint
from xyc.paths import project_root
from xyc.contracts import CAMERAS, COMMAND_VALIDITY_S, check_mobile_action_scale
from .inference_recording import InferenceRecorder
from xyc.runtime.live_robot import (
    LiveCameraClient, LiveObservationClient, RobotControlSink,
)
from xyc.runtime.policy_executor import PolicyExecutor
from xyc.runtime.policy_runner import PolicyRunner
from xyc.runtime.evaluation import EvaluationConsole


def deploy(checkpoint, *, host='localhost', port=robot_profile().ports['motion'], camera_url=None,
           device='cuda', home_lift=False, steps=None, strategy='sync_chunk', action_steps=None, task=None,
           record=True, record_dir=None, record_video=False, interactive=True, prepare_mode='keyboard',
           checkpoint_selection='latest'):
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError('steps must be a positive integer')
    if action_steps is not None and strategy != 'sync_chunk':
        raise ValueError('--action-steps requires --strategy sync_chunk')
    if not record and (record_dir is not None or record_video):
        raise ValueError('Recording directory/video cannot be combined with record=False')
    if prepare_mode not in ('keyboard', 'teleop') or not interactive and prepare_mode != 'keyboard':
        raise ValueError('Choose keyboard or teleop preparation; --auto-start skips preparation')
    adapter, manifest, path = load_checkpoint(checkpoint, device=device, selection=checkpoint_selection)
    schema = adapter.schema
    if tuple(schema.cameras) != CAMERAS:
        raise ValueError('Live deployment requires forward, chest, wrist_left and wrist_right cameras')
    live_preprocessor = getattr(adapter, 'live_batch', None)
    if live_preprocessor is None:
        raise ValueError(f'{manifest["policy"]} does not implement live observation preprocessing')
    spec = adapter.sample_spec(manifest['model_config'], schema)
    if spec.include_task and (not isinstance(task, str) or not task.strip()):
        raise ValueError('This policy requires a nonempty --task for live inference')
    check_mobile_action_scale(manifest['robot_assets'])
    calibration_ids = follower_calibration_ids(manifest['robot_assets'])
    recorder = sink = executor = runner = None
    arm_input = None
    if interactive and prepare_mode == 'teleop':
        from xyc.teleop.input import LeaderArms
        arm_input = LeaderArms()
    console = EvaluationConsole(schema, calibration_ids, arm_input=arm_input) if interactive else None
    console_open = False
    error = None
    completed = 0
    try:
        if console is not None:
            console.open()
            console_open = True
        if record:
            context = {'mode': 'live_deployment', 'execution': 'real_robot',
                       'policy': manifest['policy'], 'checkpoint': str(path),
                       'weights_sha256': manifest['weights_sha256'], 'training_step': manifest['step'],
                       'model_config': manifest['model_config'], 'statistics': manifest['statistics'],
                       'robot_assets': manifest['robot_assets'], 'device': str(device),
                       'strategy': strategy, 'task': task, 'interactive': interactive,
                       'prepare_mode': prepare_mode, 'action_steps_override': action_steps,
                       'source': {'host': host, 'port': port, 'camera_url': camera_url}}
            options = dict(schema=schema, metadata=context, record_video=record_video)
            recorder = (InferenceRecorder(record_dir, **options) if record_dir is not None else
                        InferenceRecorder.create(project_root() / 'outputs/inference', **options))
            print(f'实时推理记录：{recorder.path}', flush=True)
        strategy_options = {'action_steps': action_steps} if action_steps is not None else None
        runner = PolicyRunner(adapter, strategy=strategy, strategy_options=strategy_options, recorder=recorder)
        cameras = LiveCameraClient(camera_url or f"http://localhost:{robot_profile().ports['data']}", schema.cameras)
        sink = RobotControlSink(host, port)
        observations = LiveObservationClient(cameras, sink, calibration_ids)
        executor = PolicyExecutor(runner, sink)
        state = sink.status()
        deadline = time.monotonic() + 5.
        while not state.get('ready') and time.monotonic() < deadline:
            time.sleep(.05)
            state = sink.status()
        if not state.get('ready'):
            raise RuntimeError('Robot is not ready: ' + str(state.get('reason', 'missing feedback')))
        sink.prepare(schema, calibration_ids)
        state = sink.start()
        if state['state'][14] is None:
            if not home_lift and not interactive:
                raise ValueError('Lift needs a zero reference; use --home-lift only with a clear downward path')
            if home_lift:
                print('升降下行回零中；请确保下行路径畅通。', flush=True)
                state = sink.home_lift()
                deadline = time.monotonic() + 90.
                while state['state'][14] is None or state['chassis'].get('homing'):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Lift homing did not complete')
                    time.sleep(.05)
                    state = sink.heartbeat()
        deadline = time.monotonic() + 3.
        while state['chassis'].get('enabled') is False:
            if time.monotonic() >= deadline:
                raise TimeoutError('Base/lift enable was not confirmed in feedback')
            time.sleep(.05)
            state = sink.status()
        period = 1. / schema.fps
        next_tick = time.monotonic()
        preparing = console is not None
        needs_warmup = not runner.asynchronous
        quitting = False
        pending_key = None

        def apply_transition(choice):
            nonlocal preparing, quitting
            if choice == 'prepare':
                runner.reset()
                preparing = True
            elif choice == 'quit':
                quitting = True
            return choice == 'model'

        def allow_action():
            nonlocal pending_key
            pending_key = console.read_model_key()
            return pending_key is None

        while not quitting and (steps is None or completed < steps):
            if preparing:
                if not console.prepare(executor):
                    break
                next_tick = time.monotonic()
                preparing = False
                needs_warmup = not runner.asynchronous
            if console is not None and not apply_transition(console.model_keys(executor, completed)):
                continue
            tick_started = time.monotonic()
            observation = observations.capture()
            if task is not None:
                observation['task'] = task
            batch = live_preprocessor(observation)
            provenance = {key: value for key, value in observation.items() if key != 'images'}
            if needs_warmup:
                latency = runner.warmup(batch, observation=provenance)
                sink.heartbeat()
                needs_warmup = False
                next_tick = time.monotonic()
                if console is not None:
                    console.message = f'模型预热完成（{latency:.3f}s），正在读取新观测并开始执行。'
                else:
                    print(f'模型预热完成（{latency:.3f}s），预热动作未下发。', flush=True)
                continue
            videos = observation['images'] if record_video else None
            result = executor.step(batch, observation=provenance, video_frames=videos,
                                   deadline=tick_started + COMMAND_VALIDITY_S,
                                   allow_dispatch=allow_action if console is not None else None)
            if pending_key is not None:
                key, pending_key = pending_key, None
                apply_transition(console.apply_model_key(executor, completed, key))
                continue
            if result is not None:
                if result.accepted is not True:
                    raise RuntimeError(result.message or 'Model command was not accepted')
                completed += 1
                if console is None and (completed == 1 or completed % max(1, round(schema.fps)) == 0):
                    print(f'模型实时控制：{completed} 帧，目标 {schema.fps:g} Hz，后端已接受', flush=True)
            elif runner.asynchronous:
                sink.heartbeat()  # Main-loop progress while waiting in the initial hold.
            elapsed = time.monotonic() - tick_started
            if elapsed > COMMAND_VALIDITY_S:
                raise TimeoutError(f'Live policy tick took {elapsed:.3f}s; mobile watchdog cannot be renewed safely')
            next_tick = max(next_tick + period, time.monotonic())
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
    except BaseException as exc:
        error = exc
        if recorder is not None:
            recorder.event('inference_error', error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        try:
            if executor is not None:
                executor.close(fault=error is not None and not isinstance(error, KeyboardInterrupt))
            elif sink is not None:
                sink.stop()
            elif runner is not None:
                runner.close()
        except BaseException as exc:
            if error is None:
                error = exc
                raise
            if exc is not error and str(exc) not in str(error):
                print(f'停车未确认：{exc}', file=sys.stderr, flush=True)
                if recorder is not None:
                    recorder.event('stop_unconfirmed', message=str(exc))
        finally:
            try:
                if recorder is not None:
                    recorder.close(outcome=('interrupted' if isinstance(error, KeyboardInterrupt)
                                            else 'failed' if error is not None else 'completed'),
                                   reason=str(error) if error is not None else None)
            finally:
                if console_open:
                    console.close()
    return {'policy': manifest['policy'], 'checkpoint': str(Path(path)), 'frames': completed,
            'status': 'completed', 'execution': 'real_robot',
            'record_dir': str(recorder.path) if recorder is not None else None,
            'record_status': recorder.summary['status'] if recorder is not None else None,
            'evaluations': ({'started': console.rounds, 'success': console.successes,
                             'failure': console.failures} if console is not None else None)}
