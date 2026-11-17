"""Run a trained policy against the unified real-robot controller."""

from pathlib import Path
import time

from .checkpoint import load_checkpoint
from .contracts import CAMERAS
from xyc.runtime.live_robot import (
    LiveCameraClient, RobotControlSink, follower_calibration_ids, validate_live_state,
)
from xyc.runtime.policy_executor import PolicyExecutor
from xyc.runtime.policy_runner import PolicyRunner


def deploy(checkpoint, *, host='localhost', port=17886, camera_url='http://localhost:17883',
           device='cuda', home_lift=False, steps=None, strategy='sync_chunk'):
    if steps is not None and (type(steps) is not int or steps < 1):
        raise ValueError('steps must be a positive integer')
    adapter, manifest, path = load_checkpoint(checkpoint, device=device)
    schema = adapter.schema
    if tuple(schema.cameras) != CAMERAS:
        raise ValueError('Live deployment requires forward, chest, wrist_left and wrist_right cameras')
    live_preprocessor = getattr(adapter, 'live_batch', None)
    if live_preprocessor is None:
        raise ValueError(f'{manifest["policy"]} does not implement live observation preprocessing')
    calibration_ids = follower_calibration_ids(manifest['robot_assets'])
    cameras = LiveCameraClient(camera_url, schema.cameras)
    sink = RobotControlSink(host, port)
    runner = PolicyRunner(adapter, strategy=strategy)
    executor = PolicyExecutor(runner, sink)
    completed = 0
    try:
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
            if not home_lift:
                raise ValueError('Lift needs a zero reference; use --home-lift only with a clear downward path')
            state = sink.home_lift()
            deadline = time.monotonic() + 90.
            while state['state'][14] is None or state['chassis'].get('homing'):
                if time.monotonic() >= deadline:
                    raise TimeoutError('Lift homing did not complete')
                time.sleep(.05)
                state = sink.heartbeat()
        period = 1. / schema.fps
        next_tick = time.monotonic()
        while steps is None or completed < steps:
            tick_started = time.monotonic()
            state = sink.status()
            values = validate_live_state(state, calibration_ids)
            images, stamps, camera_latency = cameras.capture()
            batch = live_preprocessor({'state': values, 'images': images})
            result = executor.step(batch, observation={
                'source_time_ns': max(stamps), 'source_clock': 'robot_ros',
                'camera_stamps_ns': stamps, 'camera_fetch_s': camera_latency,
            })
            if result.accepted is not True:
                raise RuntimeError(result.message or 'Model command was not accepted')
            completed += 1
            if completed == 1 or completed % max(1, round(schema.fps)) == 0:
                print(f'模型实时控制：{completed} 帧，目标 {schema.fps:g} Hz，后端已接受', flush=True)
            elapsed = time.monotonic() - tick_started
            if elapsed > .25:
                raise TimeoutError(f'Live policy tick took {elapsed:.3f}s; mobile watchdog cannot be renewed safely')
            next_tick = max(next_tick + period, time.monotonic())
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        return {'policy': manifest['policy'], 'checkpoint': str(Path(path)), 'frames': completed,
                'status': 'completed', 'execution': 'real_robot'}
    finally:
        executor.close()
