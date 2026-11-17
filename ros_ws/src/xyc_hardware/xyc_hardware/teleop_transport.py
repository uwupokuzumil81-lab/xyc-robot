# SPDX-License-Identifier: Apache-2.0
"""Timing and framing shared by the PC reader and the Pi state bridge."""

import json
import time

from xyc_hardware.replay_transport import MOBILE_ACTION_SCALE


SIDES = ('left', 'right')
PERIOD = 1 / 70
PORT = 17882
REPLY_DEADLINE = .18
INPUT_LOSS_DEADLINE = .35
MAX_SAMPLE_AGE = .15
TARGET_COMMAND_EXPIRED = 'Target command expired'
TELEOP_TARGET_DROPPED = 'Teleop target expired before commit'


def running_backend_pids(project, *, with_motors, with_cameras, recording, proc_root='/proc'):
    """Identify this project's existing components without opening a device."""
    import os
    from pathlib import Path
    install = Path(project).resolve() / 'ros_ws/install'
    executables = {str(install / 'xyc_hardware/lib/xyc_hardware' / name)
                   for name in ('leader_state_bridge', 'teleop_controller', 'motion_replay')}
    if with_motors:
        executables.update(str(install / 'xyc_hardware/lib/xyc_hardware' / name)
                           for name in ('motor_state_node', 'arm_group_coordinator'))
    if with_cameras:
        executables.update(str(install / 'xyc_camera/lib/xyc_camera' / name)
                           for name in ('camera_node', 'camera_preview'))
    if recording:
        executables.add(str(install / 'xyc_recording/lib/xyc_recording/recorder'))
    executables.update(('/tmp/xyc_fps_composed_controller.py', '/tmp/xyc_fps_coordinator_resume.py'))
    result = []
    for directory in Path(proc_root).glob('[0-9]*'):
        try:
            pid = int(directory.name)
            args = {item.decode(errors='replace') for item in (directory / 'cmdline').read_bytes().split(b'\x00')}
            if pid != os.getpid() and args & executables:
                result.append(pid)
        except (OSError, ValueError):
            continue
    return sorted(result)


class BackendLease:
    """One Linux launch owns the unified backend until its shutdown."""
    def __init__(self, project):
        from pathlib import Path
        self.project = Path(project).resolve()
        self.handle = None

    def acquire(self, *, with_motors, with_cameras, recording):
        import fcntl
        import os
        directory = self.project / '.runtime'
        directory.mkdir(parents=True, exist_ok=True)
        handle = (directory / 'teleop.lock').open('a+')
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('XYC后端已在运行；不要重复启动。直接在电脑启动遥操客户端即可。') from exc
            self.handle = handle
            existing = running_backend_pids(self.project, with_motors=with_motors,
                                           with_cameras=with_cameras, recording=recording)
            if existing:
                raise RuntimeError(f'已有XYC组件正在运行，PID={existing}；请先停止旧后台，避免重复控制。')
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({'pid': os.getpid()}))
            handle.flush()
        except BaseException:
            if self.handle is not None:
                self.close()
            else:
                handle.close()
            raise

    def close(self):
        if self.handle is not None:
            import fcntl
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def cadence(wait=time.sleep, *, period=PERIOD):
    scheduled = time.monotonic()
    while True:
        while (delay := scheduled - time.monotonic()) > 0:
            if wait(delay):
                return
        # If a read overruns the period, the next read can start immediately.
        # Base the next deadline on the actual start to avoid catch-up bursts.
        scheduled = time.monotonic() + period
        yield


def send_frame(channel, frame):
    channel.write(json.dumps(frame, allow_nan=False, separators=(',', ':')).encode('ascii') + b'\n')
    channel.flush()


def read_frame(channel):
    data = channel.readline(8192)
    if not data:
        raise EOFError('连接已关闭')
    if not data.endswith(b'\n'):
        raise ValueError('消息过长或不完整')
    frame = json.loads(data)
    if not isinstance(frame, dict) or type(frame.get('sequence')) is not int:
        raise ValueError('消息缺少序号')
    return frame


def check_arm_positions(side, values):
    if len(values) != 7:
        raise ValueError(f'{side} 应有 7 路关节位置')
    for i, value in enumerate(values):
        lower = 0 if i == 6 else -100
        if type(value) not in (float, int) or not lower <= value <= 100:
            raise ValueError(f'{side} ID {i + 1} 位置无效')


def checked_positions(frame):
    sides = frame.get('positions')
    if not isinstance(sides, dict) or not sides or sides.keys() - set(SIDES):
        raise ValueError('主臂消息缺少有效侧别')
    for side, values in sides.items():
        if not isinstance(values, list):
            raise ValueError(f'{side} 主臂位置格式无效')
        check_arm_positions(side, values)
    return sides


def checked_chassis(payload):
    """Validate PC intent independently of leader encoder freshness."""
    import math

    if not isinstance(payload, dict):
        raise ValueError('底盘指令格式无效')
    base, lift = payload.get('base'), payload.get('lift_m_s')
    if (not isinstance(base, list) or len(base) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in base)
            or type(lift) not in (int, float) or not math.isfinite(lift)):
        raise ValueError('底盘或升降速度无效')
    physical = (float(lift), float(base[0]), float(base[1]), float(base[2]))
    for value, speed in zip(physical, MOBILE_ACTION_SCALE):
        if min(abs(value - candidate) for candidate in (-speed, 0., speed)) > 1e-6:
            raise ValueError('底盘和升降只允许 -1/0/+1 对应的固定速度')
    operation = payload.get('operation')
    if operation is not None and (
            not isinstance(operation, dict) or type(operation.get('id')) is not int
            or not 0 < operation['id'] < 2**64
            or operation.get('name') not in ('enable', 'stop', 'home_lift')):
        raise ValueError('底盘操作序号或名称无效')
    return tuple(base), float(lift), operation
