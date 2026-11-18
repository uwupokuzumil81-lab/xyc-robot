"""Replay recorded actions on the real robot through its existing ROS drivers.

The whole selected sequence is checked before any connection. No model or
PyTorch is loaded. Default inspection does not open a control connection.
"""

from xyc.robot.profile import robot_profile
from xyc.assets import follower_calibration_ids
from xyc.io import write_json

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import numpy as np

from xyc.paths import project_root
from xyc.contracts import (
    ACTION, ACTION_NAMES, ACTION_SEMANTICS, ACTION_UNITS, LIFT_SOFT_STOP_MARGIN_M,
    OBS_STATE, STATE_NAMES, STATE_UNITS, check_mobile_action_scale, checked_action,
)

from xyc.transport.motion import MotionClient, PROTOCOL

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError as exc:
    raise ValueError('Replay needs PyArrow; install with: python -m pip install ".[replay]"') from exc

from .dataset_metadata import DatasetMetadata


class ReplayPlan:
    def __init__(self, dataset, *, episode=0, frame=0, frames=None, speed=1.):
        if not math.isfinite(speed) or not .1 <= speed <= 1.:
            raise ValueError('Replay speed must be in 0.1..1.0')
        self.metadata = metadata = DatasetMetadata(dataset)
        schema = metadata.schema
        if (schema.action_names != ACTION_NAMES or schema.state_names != STATE_NAMES
                or schema.action_units != ACTION_UNITS or schema.state_units != STATE_UNITS):
            raise ValueError('Real replay requires the existing 18-dimensional XYC dataset contract')
        if metadata.assets.get('action_semantics') != ACTION_SEMANTICS:
            raise ValueError('Unsupported recorded action semantics; refusing to guess actuator targets')
        check_mobile_action_scale(metadata.assets)
        if episode not in metadata.episodes:
            raise ValueError('Requested replay episode does not exist')
        length = metadata.episodes[episode].length
        frames = length - frame if frames is None else frames
        if type(frame) is not int or type(frames) is not int or frame < 0 or frames < 1 or frame + frames > length:
            raise ValueError('Replay frame range must stay inside one recorded episode')
        self.episode, self.frame, self.speed = episode, frame, speed
        self.ids = follower_calibration_ids(metadata.assets)
        numeric = metadata.numeric(episode)
        self.initial_state = numeric[OBS_STATE][frame].astype(np.float64).tolist()
        maximum = self.lift_max_height_m = robot_profile().lift_max_height_m
        # The dataset stores float32. Accept the nominal upper bound at its stored precision.
        if not 0 <= self.initial_state[14] <= max(maximum, float(np.float32(maximum))):
            raise ValueError(f'Recorded start lift height is outside the verified 0..{maximum:g} m travel')
        self.initial_state[14] = min(self.initial_state[14], maximum)
        self.actions = numeric[ACTION][frame:frame + frames].astype(np.float64).copy()
        if speed != 1. and np.any(self.actions[:, 14:] != 0.):
            raise ValueError(
                '底盘/升降动作只允许 -1、0、+1 的固定速度；'
                '包含移动动作的回放必须使用 --speed 1。')
        for index, action in enumerate(self.actions):
            self.actions[index] = checked_action(action.tolist())
        path = metadata.data_path(episode)
        self.timing = 'nominal_fps'
        times = np.arange(length, dtype=np.float64) / schema.fps
        if 'observation.source_time' in pq.read_schema(path).names:
            rows = pq.read_table(path, columns=['episode_index', 'frame_index', 'observation.source_time']).to_pylist()
            rows = sorted((r for r in rows if r['episode_index'] == episode), key=lambda r: r['frame_index'])
            values = [r['observation.source_time'] for r in rows]
            times = np.asarray([v[0] if isinstance(v, list) and len(v) == 1 else v for v in values], dtype=np.float64)
            if times.shape != (length,) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError('Recorded source time is missing, repeated or non-monotonic')
            self.timing = 'original_observation_source_time'
        original = times[frame:frame + frames]
        self.times = (original - original[0]) / speed
        self.period = float(np.median(np.diff(times))) / speed if length > 1 else 1 / schema.fps / speed
        self.duration = float(self.times[-1] + self.period)

    def summary(self):
        return {'dataset': str(self.metadata.root), 'episode': self.episode, 'start_frame': self.frame,
                'frames': len(self.actions), 'speed': self.speed, 'duration_s': self.duration,
                'timing': self.timing, 'dimensions': 18, 'calibration_ids': self.ids,
                'initial_state': self.initial_state,
                'velocity_semantics': 'strict ternary base/lift commands mapped to fixed physical speeds',
                'execution': 'checked_only_no_connection'}

    def lift_alignment_error(self, height):
        requested = self.initial_state[14]
        floor, ceiling = LIFT_SOFT_STOP_MARGIN_M, self.lift_max_height_m - LIFT_SOFT_STOP_MARGIN_M
        if (height <= floor and requested < height) or (height >= ceiling and requested > height):
            return 0.  # The requested direction is already at its software stop.
        return requested - height

    def check_start_pose(self, positions):
        """Use the full calibrated replay range, without an extra endpoint inset."""
        if len(positions) != 14:
            raise ValueError('Replay start needs all 14 current joint positions')
        for index, (current, target) in enumerate(zip(positions, self.initial_state[:14])):
            lower, upper = (0., 100.) if index % 7 == 6 else (-100., 100.)
            name = self.metadata.schema.state_names[index]
            if (not math.isfinite(current) or not math.isfinite(target)
                    or not lower <= current <= upper or not lower <= target <= upper):
                raise ValueError(
                    f'回放起点不可执行：{name} 当前 {current:.3f}，录制起点 {target:.3f}；'
                    f'必须位于标定范围 [{lower:g}, {upper:g}]，不自动裁剪目标。'
                )


def execute(plan, *, host='localhost', port=robot_profile().ports['motion'], home_lift=False, output_dir=None,
            client_factory=MotionClient, clock=time.monotonic, sleep=time.sleep, progress=print):
    # Disk imports and dataset hashing finish BEFORE opening the deadline-sensitive control link.
    summary = plan.summary()
    summary['dataset_fingerprint'] = plan.metadata.fingerprint([plan.episode])
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    root = Path(output_dir) if output_dir is not None else project_root() / 'outputs/replay' / name
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    summary.update(execution='real_robot', status='starting', started_at_utc=datetime.now(timezone.utc).isoformat(),
                   task_success=None, host=host, port=port)
    write_json(root / 'meta.json', summary)
    client, started, finished, sent = None, False, False, 0
    log = (root / 'steps.jsonl').open('w', encoding='utf-8')
    log_error = None

    def record(kind, **values):
        nonlocal log_error
        if log_error is not None:
            raise OSError('Replay log failed; replay has been stopped: ' + log_error)
        try:
            log.write(json.dumps({'kind': kind, 'pc_monotonic_s': clock(), **values}, allow_nan=False) + '\n')
            log.flush()
        except OSError as exc:
            log_error = str(exc)
            raise

    def command(action, phase, index=None):
        before = clock()
        snapshot = client.call('target', action=[float(v) for v in action])
        record(phase, frame=index, action=list(action), driver_accepted=True,
               round_trip_s=clock() - before, measured_snapshot=snapshot)
        return snapshot

    def wait_until(deadline):
        while (remaining := deadline - clock()) > 0:
            sleep(min(.05, remaining))
            if deadline - clock() > .05:
                client.call('heartbeat')  # Refresh mobile velocity watchdog even during a recorded gap.

    try:
        client = client_factory(host, port)
        deadline = clock() + 5.
        state = client.call('status')
        while not state.get('ready') and clock() < deadline:
            sleep(.05)
            state = client.call('status')
        if not state.get('ready'):
            raise RuntimeError('Robot is not ready: ' + state.get('reason', 'missing feedback'))
        if state['state'][14] is None and not home_lift:
            raise ValueError('Lift needs a zero reference; --home-lift explicitly permits downward homing')
        record('preflight', measured_snapshot=state)
        plan.check_start_pose(state['state'][:14])
        schema = plan.metadata.schema
        client.call('prepare', protocol=PROTOCOL, calibration_ids=plan.ids,
                    action_names=list(schema.action_names), action_units=list(schema.action_units),
                    state_names=list(schema.state_names), state_units=list(schema.state_units))
        started = True  # A lost start reply may conceal torque enable; always attempt stop on failure.
        state = client.call('start')
        deadline = clock() + 2.
        while not state['chassis'].get('enabled'):
            if clock() >= deadline:
                raise TimeoutError('Base/lift enable was not confirmed in feedback')
            sleep(.05)
            state = client.call('heartbeat')
        if state['state'][14] is None:
            progress('升降回零中；请确保下行路径畅通。')
            state = client.call('home')
            deadline = clock() + 90.
            while state['state'][14] is None or state['chassis'].get('homing'):
                if clock() >= deadline:
                    raise TimeoutError('Lift homing did not complete')
                sleep(.05)
                state = client.call('heartbeat')
                record('homing', measured_snapshot=state)
        progress('正在缓慢对齐首帧双臂位置与升降高度；底盘保持零速度。')
        target = np.asarray(plan.initial_state[:14], dtype=np.float64)
        reference = np.asarray(state['state'][:14], dtype=np.float64)
        deadline = clock() + 120.
        previous = clock()
        settled = 0
        while settled < 3:
            if clock() >= deadline:
                raise TimeoutError('Robot did not reach the replay start pose')
            now = clock()
            step = 8. * min(.1, max(.02, now - previous))
            previous = now
            reference += np.clip(target - reference, -step, step)
            height_error = plan.lift_alignment_error(state['state'][14])
            lift = (1. if height_error > .003 else -1. if height_error < -.003 else 0.)
            state = command([*reference, lift, 0., 0., 0.], 'align')
            height_arrived = abs(plan.lift_alignment_error(state['state'][14])) <= .003
            arrived = max(abs(a - b) for a, b in zip(target, state['state'][:14])) <= 3. and height_arrived
            settled = settled + 1 if arrived else 0
            sleep(.04)
        progress(f'首帧就绪，开始实机回放 {len(plan.actions)} 帧，约 {plan.duration:.1f} 秒。')
        origin = clock()
        for index, (offset, action) in enumerate(zip(plan.times, plan.actions)):
            wait_until(origin + float(offset))
            lag = clock() - origin - float(offset)
            if lag > max(.15, plan.period * 3.):
                raise TimeoutError(f'Replay fell behind by {lag:.3f}s; no frames skipped or commands retried')
            command(action.tolist(), 'replay', plan.frame + index)
            sent += 1
        wait_until(origin + plan.duration)
        client.call('finish')
        finished = True
        record('finished', frames=sent, measured_snapshot=client.snapshot)
        summary.update(status='completed', frames_sent=sent)
        return {**summary, 'record_dir': str(root)}
    except BaseException as exc:
        summary.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                       reason=f'{type(exc).__name__}: {exc}', frames_sent=sent)
        snapshot = getattr(exc, 'snapshot', None)
        if snapshot is None and client is not None:
            snapshot = client.snapshot
        summary['failure_snapshot'] = snapshot
        if log_error is None:
            try:
                record('failure', reason=summary['reason'], measured_snapshot=snapshot)
            except OSError:
                pass  # Keep the original control failure if its diagnostic write also fails.
        raise
    finally:
        if client is not None:
            if started and not finished:
                try:
                    if not getattr(client, 'last_stop_confirmed', False):
                        client.call('stop')
                    summary['stop_confirmed'] = True
                except Exception as exc:
                    summary['stop_confirmed'] = False
                    summary['stop_error'] = str(exc)
            try:
                client.close()
            except OSError as exc:
                summary['connection_close_error'] = str(exc)
        log.close()
        summary['ended_at_utc'] = datetime.now(timezone.utc).isoformat()
        if log_error is not None:
            summary['log_error'] = log_error
        write_json(root / 'meta.json', summary)
