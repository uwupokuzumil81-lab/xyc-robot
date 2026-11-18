"""Exclusive ownership of a ROS backend, without importing ROS."""

import json


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
