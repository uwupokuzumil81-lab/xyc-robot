"""Stream calibrated leader positions over an SSH forward to the Pi."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import threading
import time
import unicodedata


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ros_ws' / 'src' / 'xyc_hardware'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ros_ws' / 'src' / 'xyc_recording'))
from xyc_hardware.teleop_transport import (
    MAX_SAMPLE_AGE, PORT, SIDES, cadence, read_frame, send_frame,
)

# Sample independently of the bridge's requests so each request can use
# the latest serial result. The cap also bounds polling on faster adapters.
LEADER_SAMPLE_PERIOD = 1 / 80
LEADER_STARTUP_MARGIN_TICKS = 256
LEADER_VOLTAGE_HOLD_NS = 80_000_000
PREVIEW_URL = 'http://localhost:17883'
DATA_PORT = 17883


class SSHDataTunnel:
    """Own a dedicated SSH connection for preview and recording bytes."""

    def __init__(self, host, user, *, port=DATA_PORT, wait=time.sleep):
        self.host, self.user, self.port, self.wait = host, user, port, wait
        self.process = None

    @staticmethod
    def port_open(port, timeout=.15):
        try:
            with socket.create_connection(('localhost', port), timeout=timeout):
                return True
        except OSError:
            return False

    def start(self):
        if self.port_open(self.port):
            raise ValueError(
                f'本机{self.port}端口已被占用；后端SSH只转发17882，'
                f'{self.port}必须由桌面客户端单独建立，避免视频阻塞控制'
            )
        executable = shutil.which('ssh')
        if executable is None:
            raise ValueError('未找到ssh命令；Windows请启用OpenSSH客户端')
        command = [
            executable, '-N', '-T',
            '-o', 'ExitOnForwardFailure=yes',
            '-o', 'ServerAliveInterval=15',
            '-o', 'ServerAliveCountMax=3',
            '-o', 'StrictHostKeyChecking=accept-new',
            '-L', f'{self.port}:127.0.0.1:{self.port}',
            f'{self.user}@{self.host}',
        ]
        print(f'正在建立独立数据通道 {self.user}@{self.host}，请输入树莓派密码……', flush=True)
        self.process = subprocess.Popen(command)
        deadline = time.monotonic() + 60.
        while time.monotonic() < deadline:
            code = self.process.poll()
            if code is not None:
                raise ConnectionError(f'独立数据SSH通道启动失败，退出码 {code}')
            if self.port_open(self.port):
                print(f'独立数据通道已就绪：localhost:{self.port}', flush=True)
                return self
            self.wait(.1)
        self.close()
        raise TimeoutError('等待独立数据SSH通道超时')

    def close(self):
        process, self.process = self.process, None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=3.)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3.)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()


class InvalidLeaderFrame(ValueError):
    """A frame unsuitable for follower targets, despite a live connection."""


class RecordingRemote:
    """Recording controls/status only; failures never stop the master reader."""

    def __init__(self, config):
        self.lock = threading.RLock()
        self.config = config
        self.operation = None
        self.next_id = 1
        self.status = {}
        self.status_at = 0.
        self.notice = '采集待机；完成升降回零和姿态准备后按 R 开录'
        self.supported = False
        self.error = ''
        self.local = None

    def start_local(self, datasets_dir):
        from xyc_recording.local_recorder import LocalDatasetReceiver
        self.local = LocalDatasetReceiver(datasets_dir, self.config).start()
        print(f'数据保存到本地电脑：{self.local.datasets_dir / self.config["dataset"]}', flush=True)
        return self.local

    def key(self, key):
        with self.lock:
            if self.operation is not None:
                self.notice = '正在等待上一个采集操作回执'
                return
            operation = {'r': 'start', 'b': 'save', 'n': 'discard', 'q': 'save', '\x03': 'save'}.get(key.lower())
            if operation is None:
                return
            recovering = (self.status.get('phase') == 'error'
                          and self.status.get('buffering') == 'robot_episode_cache'
                          and self.config.get('resume') is True)
            if operation == 'start' and self.status.get('phase') not in (None, 'idle') and not recovering:
                self.notice = '当前回合尚未结束，请B保存或N丢弃'
                return
            self.operation = {'id': self.next_id, 'operation': operation,
                              'config': self.config if operation == 'start' else {}}
            self.next_id += 1
            self.error = ''
            self.notice = {'start': '正在请求开始录制', 'save': '正在请求保存回合',
                           'discard': '正在请求丢弃当前未保存回合'}[operation]

    def receive(self, request):
        with self.lock:
            self.supported = request.get('recording_supported') is True
            if not self.supported:
                self.status = {}
                self.notice = '后端未开启采集：请加 recording:=true；遥操继续'
                return
            status = request.get('recording_status')
            if isinstance(status, dict) and status:
                self.status, self.status_at = status, time.monotonic()
            if self.local is not None and status:
                if status.get('storage') != 'pc':
                    self.notice = '后端仍是旧版树莓派写盘模式，请更新采集后端；不会开始录制'
                    return
                if not self.local.ready.is_set():
                    self.notice = '等待电脑采集数据通道，请检查17883端口转发'
                    return
            ack = request.get('recording_ack')
            if isinstance(ack, dict) and self.operation is not None and ack.get('id') == self.operation['id']:
                self.notice = str(ack.get('message', ''))
                if not ack.get('accepted'):
                    self.error = self.notice
                self.operation = None

    def command(self):
        with self.lock:
            return dict(self.operation) if self.operation is not None else None

    def home_allowed(self):
        with self.lock:
            return (self.status.get('phase') not in ('waiting', 'recording', 'saving')
                    and (self.operation is None or self.operation['operation'] != 'start'))

    def summary(self):
        with self.lock:
            status = self.status if time.monotonic() - self.status_at < 2. else {}
            phase = {'idle': '待机', 'waiting': '等待数据', 'recording': '录制中',
                     'saving': '保存中', 'paused': '暂停', 'error': '错误'}.get(status.get('phase'), '未就绪')
            local = self.local.snapshot() if self.local is not None else {}
            if local:
                status = {**status, 'saved_episodes': local['saved_episodes']}
            notice = self.notice if (self.operation is not None
                                     or not status.get('dataset')
                                     or self.local is not None and status.get('storage') != 'pc') else ''
            receiving = f'电脑已接收{local["frames"]}帧 | ' if local and status.get('phase') == 'saving' else ''
            return (f'采集{phase} {status.get("frames", 0)}/{status.get("target_frames", "-")} '
                    f'已存{status.get("saved_episodes", 0)}回合 | '
                    + receiving + str(local.get('error') or self.error or notice or status.get('message') or self.notice))


def open_camera_preview(wait=time.sleep):
    """Open the read-only dashboard independently of the control connection."""
    import urllib.error
    import urllib.request
    import webbrowser

    # Match the control connection: ssh -6 may listen on ::1 only.
    url = PREVIEW_URL
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for attempt in range(20):
        try:
            with opener.open(url + '/api/status', timeout=1.0) as response:
                if response.status != 200:
                    raise OSError('预览服务未就绪')
            opened = webbrowser.open(url, new=2)
            if not opened:
                print(f'\n未找到桌面浏览器，请打开 {url} 查看四路相机。', flush=True)
            return opened
        except (OSError, urllib.error.URLError):
            if attempt < 19:
                wait(.25)
    print('\n相机预览未连接：请确认后端相机已启动，独立17883数据通道保持在线。', flush=True)
    return False


class LocalChassisKeyboard:
    """PC-only operator state. Windows input and serial sampling run independently."""

    def __init__(self, recording=None):
        self.recording = recording
        self.lock = threading.Lock()
        self.base, self.lift = (0., 0., 0.), 0.
        self.base_until = self.lift_until = 0.
        self.state = {}
        self.received = 0.
        self.operation = {'id': 1, 'name': 'enable'}
        self.next_operation = 2
        self.control_error = ''
        self.quit_at = None
        self.quit_confirmed = False
        self.notice = '连接后自动准备；主臂直接跟随，升降先 H 回零'

    def handle_key(self, key, now=None):
        key = key.lower()
        if key == 'h' and self.recording is not None and not self.recording.home_allowed():
            with self.lock:
                self.notice = '当前采集回合尚未结束；先 B 保存或 N 丢弃，再按 H 回零'
            return
        if self.recording is not None and key in ('r', 'b', 'n', 'q', '\x03'):
            self.recording.key(key)
        now = time.monotonic() if now is None else now
        with self.lock:
            if key in ('h', ' ', 'q', '\x03'):
                self.base, self.lift = (0., 0., 0.), 0.
                self.base_until = self.lift_until = 0.
                name = {'h': 'home_lift', ' ': 'stop',
                        'q': 'stop', '\x03': 'stop'}[key]
                if self.operation is not None and name != 'stop':
                    self.notice = '正在等待上一个操作结果'
                    return
                self.operation = {'id': self.next_operation, 'name': name}
                self.next_operation += 1
                if key in ('q', '\x03'):
                    self.quit_at = now
            elif key in 'wasdjl':
                self.base = {
                    'w': (.15, 0., 0.), 's': (-.15, 0., 0.),
                    'a': (0., .15, 0.), 'd': (0., -.15, 0.),
                    'j': (0., 0., .60), 'l': (0., 0., -.60),
                }[key]
                self.base_until = now + .25
            elif key in 'ik':
                if self.state.get('homing'):
                    self.lift, self.lift_until = 0., 0.
                    self.notice = '升降正在回零，请等待完成后再按 I/K；空格可停止'
                    return
                self.lift = .015 * (1 if key == 'i' else -1)
                self.lift_until = now + .25

    def command(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            active = (now - self.received <= .5 and self.state.get('enabled')
                      and not self.state.get('homing') and not self.state.get('feedback_paused'))
            return {
                'base': list(self.base if active and now < self.base_until else (0., 0., 0.)),
                'lift_m_s': self.lift if active and now < self.lift_until else 0.,
                'operation': dict(self.operation) if self.operation is not None else None,
            }

    def receive(self, request, now=None):
        now = time.monotonic() if now is None else now
        if request.get('chassis_supported') is not True:
            raise ValueError('树莓派启动命令需要加 chassis_mode:=true')
        with self.lock:
            status = request.get('chassis_status')
            if isinstance(status, dict) and status:
                self.state, self.received = status, now
            ack = request.get('chassis_ack')
            if isinstance(ack, dict) and self.operation is not None and ack.get('id') == self.operation['id']:
                if ack.get('pending'):
                    self.notice = str(ack.get('message', '正在等待底盘准备'))
                    return
                self.notice = ('已接受：' if ack.get('accepted') else '已拒绝：') + str(ack.get('message', ''))
                self.control_error = '' if ack.get('accepted') else self.notice
                if self.quit_at is not None and self.operation['name'] == 'stop':
                    self.quit_confirmed = ack.get('accepted') is True
                self.operation = None

    def should_quit(self):
        with self.lock:
            return self.quit_at is not None and (
                self.quit_confirmed or time.monotonic() - self.quit_at >= 1.)

    def set_notice(self, notice):
        with self.lock:
            self.notice = notice

    def run_console(self, stop):
        print('本地电脑控制：主臂连接后直接跟随；W/S 前后，A/D 左右，J/L 旋转，I/K 升降。')
        print('H 升降下行回零，确保底部路径畅通；空格停车，Q 停车并退出主臂发送。')
        print('固定速度：前后/平移 0.15 m/s，旋转 0.60 rad/s，升降 0.015 m/s。')
        if self.recording is not None:
            print('采集：连接后待机，准备好按 R 开录；B 保存；N 丢弃；达到帧数自动保存。')
            print('默认整机18维；录制时树莓派临时缓存，录完传到电脑保存并校验，完成后才能R开下一回合。')
        with ConsoleKeys() as console:
            self._console_loop(stop, console)
        print()

    def _console_loop(self, stop, console):
        last_draw = 0.
        printed_error = ''
        held_base = held_lift = None
        while not stop.is_set():
            for key in console.poll():
                self.handle_key(key)
                if key in 'wasdjl':
                    held_base = key
                elif key in 'ik':
                    held_lift = key
                elif key in (' ', 'q', '\x03', 'h'):
                    held_base = held_lift = None
            for axis, key in (('base', held_base), ('lift', held_lift)):
                if key is None:
                    continue
                held = console.is_held(key)
                if held is True:
                    self.handle_key(key)
                elif held is False:
                    with self.lock:
                        if axis == 'base':
                            self.base_until = 0.
                            held_base = None
                        else:
                            self.lift_until = 0.
                            held_lift = None
            now = time.monotonic()
            if now - last_draw >= .2:
                with self.lock:
                    state = self.state if now - self.received <= .5 else {}
                    height = state.get('height_m')
                    lift_label = {'seeking': '回零中', 'settling': '等待回零稳定'}.get(
                        state.get('homing'),
                        f'{height * 1000:.1f}mm' if height is not None else '未回零',
                    )
                    if state.get('feedback_paused'):
                        lift_label = '反馈暂缺'
                    error = state.get('fault') or self.control_error
                    if error and error != printed_error:
                        sys.stdout.write('\n底盘/升降操作失败：' + error + '\n')
                        printed_error = error
                    line = ('固定速度 | 底盘 '
                            f'{"已使能" if state.get("enabled") else "未使能"} | 升降 '
                            + lift_label
                            + ' | ' + (error or self.notice if state else '等待树莓派状态'))
                    if self.recording is not None:
                        line = self.recording.summary() + ' | ' + line
                width = max(20, shutil.get_terminal_size((100, 20)).columns - 1)
                text, used = [], 0
                for char in line:
                    columns = 2 if unicodedata.east_asian_width(char) in ('F', 'W') else 1
                    if used + columns > width:
                        break
                    text.append(char)
                    used += columns
                sys.stdout.write('\r' + ''.join(text) + ' ' * (width - used))
                sys.stdout.flush()
                last_draw = now
            stop.wait(.01)


class ConsoleKeys:
    """Native held-key polling on Windows; restored cbreak input on Linux."""

    def __init__(self, stream=None):
        self.stream = sys.stdin if stream is None else stream
        self.escape = False
        self.previous = None

    def __enter__(self):
        if sys.platform == 'win32':
            import ctypes
            import msvcrt
            from ctypes import wintypes
            self.msvcrt = msvcrt
            self.key_state = ctypes.windll.user32.GetAsyncKeyState
            self.key_state.argtypes, self.key_state.restype = [ctypes.c_int], ctypes.c_short
            self.foreground = ctypes.windll.user32.GetForegroundWindow
            self.foreground.argtypes, self.foreground.restype = [], wintypes.HWND
            self.window = self.foreground()
        else:
            import termios
            import tty
            self.termios = termios
            self.previous = termios.tcgetattr(self.stream.fileno())
            tty.setcbreak(self.stream.fileno())
        return self

    def __exit__(self, *_):
        if self.previous is not None:
            self.termios.tcsetattr(self.stream.fileno(), self.termios.TCSADRAIN, self.previous)

    def poll(self):
        keys = []
        if sys.platform == 'win32':
            while self.msvcrt.kbhit():
                key = self.msvcrt.getwch()
                if key in ('\x00', '\xe0'):
                    self.msvcrt.getwch()
                else:
                    self.window = self.foreground()
                    keys.append(key.lower())
        else:
            import select
            if select.select([self.stream], [], [], 0)[0]:
                for key in os.read(self.stream.fileno(), 128).decode('utf-8', errors='ignore').lower():
                    if key == '\x1b':
                        self.escape = True
                    elif self.escape:
                        if key.isalpha() or key == '~':
                            self.escape = False
                    else:
                        keys.append(key)
        return keys

    def is_held(self, key):
        if sys.platform != 'win32':
            return None
        return self.foreground() == self.window and bool(self.key_state(ord(key.upper())) & 0x8000)


def leader_position(entry, tick, is_gripper, previous_delta=None):
    from leader_arm import PERIOD as encoder_period, integer, normalize_tick

    integer(tick, 0, encoder_period - 1)
    low, high = entry['min_delta_ticks'], entry['max_delta_ticks']
    reference = entry['reference_tick']
    raw_delta = tick - reference
    anchor = (low + high) / 2 if previous_delta is None else previous_delta
    delta = raw_delta + encoder_period * round((anchor - raw_delta) / encoder_period)
    if previous_delta is None:
        # Start close enough to the recorded arc to pick a unique encoder turn.
        margin = min(LEADER_STARTUP_MARGIN_TICKS, (encoder_period - 1 - (high - low)) // 2)
        if not low - margin <= delta <= high + margin:
            raise ValueError('启动位置离标定范围过远')
    elif abs(delta - previous_delta) > encoder_period // 4:
        raise ValueError('相邻反馈跳变过大')

    # The leader's recorded arc maps hand motion to commands. Outside that
    # arc, keep its endpoint until the leader moves back. Keep the unwrapped
    # reading so an encoder rollover cannot change which endpoint is held.
    bounded_tick = (reference + min(high, max(low, delta))) % encoder_period
    settings = dict(entry, direction=1, normalized_min=0 if is_gripper else -100,
                    normalized_max=100)
    return normalize_tick(settings, bounded_tick), delta


def leader_stream_sample(reader, ranges, names, last_good, *, clock_ns=time.monotonic_ns):
    from leader_arm import VOLTAGE_ALARM, decode_feedback, servo_error_detail

    started = oldest = clock_ns()
    try:
        replies, alarms = reader.read_stream_feedback()
    except (ConnectionError, ValueError) as exc:
        raise InvalidLeaderFrame(str(exc)) from exc
    sampled_at = clock_ns()
    values, fresh, notices = [], {}, []
    for motor_id, name in enumerate(names, 1):
        reply = replies[motor_id]
        previous = last_good.get(name)
        if reply.servo_error:
            detail = servo_error_detail(motor_id, reply)
            if (reply.servo_error != VOLTAGE_ALARM or previous is None
                    or sampled_at - previous[2] > LEADER_VOLTAGE_HOLD_NS):
                raise InvalidLeaderFrame(f'{name}: {detail}')
            values.append(previous[1])
            oldest = min(oldest, previous[2])
            notices.append(f'{detail}，短暂保持上一有效位置')
            continue
        tick = decode_feedback(reply.data)['position_ticks']
        try:
            value, delta = leader_position(
                ranges[name], tick, motor_id == 7, previous[0] if previous is not None else None,
            )
        except ValueError as exc:
            raise InvalidLeaderFrame(f'{name}: {exc}（编码器 {tick}）') from exc
        values.append(value)
        fresh[name] = (delta, value, started)
        if motor_id in alarms:
            volts = decode_feedback(reply.data)['voltage_v']
            notices.append(f'{alarms[motor_id]}，复读正常 {volts:.1f} V')
    last_good.update(fresh)
    return values, oldest, notices


def run_leader(args):
    sys.setswitchinterval(.001)  # Two serial readers, network and console share one interpreter.
    from leader_arm import (ROOT, USB_SERIALS, LeaderCalibrationReader, arm_names, identity,
                            load_calibration, open_spec, require_torque_off)

    recording = RecordingRemote(args.record_config) if getattr(args, 'record', False) else None
    keyboard = LocalChassisKeyboard(recording) if getattr(args, 'keyboard', False) else None
    if keyboard is not None and (sys.platform not in ('win32', 'linux') or not sys.stdin.isatty()):
        raise ValueError('--keyboard 请在本地 Windows 或 Linux 交互终端运行')
    with ExitStack() as stack:
        if getattr(args, 'preview', True) or recording is not None:
            stack.enter_context(SSHDataTunnel(
                getattr(args, 'robot_host', 'raspberrypi.local'),
                getattr(args, 'robot_user', 'pi5'),
            ))
        # Native video/Arrow/pandas imports can take seconds on a cold desktop.
        # Finish them BEFORE opening the control socket: the Pi expects its
        # first reply within REPLY_DEADLINE, just like every later control frame.
        if recording is not None:
            print('正在准备本地数据保存，完成后连接控制通道……', flush=True)
            local = recording.start_local(getattr(args, 'data_dir', ROOT / 'data' / 'datasets'))
            stack.callback(local.close)
        inputs = {}
        calibration_ids = {}
        last_voltage_notice = {}
        for side in SIDES if args.side == 'both' else (args.side,):
            spec = open_spec(side)
            reader = LeaderCalibrationReader(spec, allow_release=False)
            stack.callback(reader.close)
            reader.open()
            settings = reader.settings()
            require_torque_off(settings)
            ranges = load_calibration(ROOT / 'config' / 'leader_calibration' / f'{side}.json',
                                      side, settings)
            inputs[side] = (reader, ranges, arm_names(side), {})
            cal = {'usb': USB_SERIALS[side], 'hardware': identity(settings), 'ranges': ranges}
            calibration_ids[side] = hashlib.sha256(json.dumps(cal, sort_keys=True).encode()).hexdigest()
            print(f'{side} 主臂已就绪 ({spec["port"]})', flush=True)

        def sample_side(side):
            reader, ranges, names, last_good = inputs[side]
            try:
                values, oldest, notices = leader_stream_sample(reader, ranges, names, last_good)
            except InvalidLeaderFrame as exc:
                raise InvalidLeaderFrame(f'{side} 主臂：{exc}') from exc
            now = time.monotonic()
            if notices and now - last_voltage_notice.get(side, -float('inf')) >= 1.:
                message = f'{side} 主臂：' + '、'.join(notices)
                if keyboard is None:
                    print(message, flush=True)
                else:
                    keyboard.set_notice(message)
                last_voltage_notice[side] = now
            return values, oldest

        pool = stack.enter_context(ThreadPoolExecutor(max_workers=2)) if len(inputs) == 2 else None

        def frames(wait=time.sleep):
            for sample_id, _ in enumerate(cadence(wait, period=LEADER_SAMPLE_PERIOD), 1):
                started = time.monotonic_ns()
                try:
                    if pool is None:
                        samples = {args.side: sample_side(args.side)}
                    else:
                        jobs = {side: pool.submit(sample_side, side) for side in inputs}
                        errors, samples = [], {}
                        for side, job in jobs.items():
                            try:
                                samples[side] = job.result()
                            except (ValueError, OSError) as exc:
                                errors.append(exc)
                        if errors:
                            raise errors[0]
                except InvalidLeaderFrame as exc:
                    yield started, {'sample_id': sample_id, 'error': str(exc)}
                else:
                    positions = {side: sample[0] for side, sample in samples.items()}
                    started = min(started, *(sample[1] for sample in samples.values()))
                    yield started, {'sample_id': sample_id, 'positions': positions,
                                    'calibration_ids': calibration_ids}

        if args.monitor:
            start = time.monotonic()
            valid = 0
            for _, frame in frames():
                sequence = frame['sample_id']
                if 'error' in frame:
                    print(f'第 {sequence} 帧跳过：{frame["error"]}', flush=True)
                    continue
                valid += 1
                if sequence == 1 or sequence % 15 == 0:
                    print(f'{sequence} 帧，有效 {valid} / {time.monotonic()-start:.2f}s  {frame["positions"]}', flush=True)
            return

        stop = threading.Event()
        lock = threading.Lock()
        latest = (0, {'error': '正在等待主臂首帧'})

        def collect():
            nonlocal latest
            try:
                for item in frames(stop.wait):
                    if stop.is_set():
                        return
                    with lock:
                        latest = item
            except OSError as exc:
                with lock:
                    latest = (0, {'error': str(exc)})

        worker = threading.Thread(target=collect, name='leader-sampling')
        worker.start()
        stack.callback(worker.join)
        stack.callback(stop.set)
        try:
            connection = stack.enter_context(socket.create_connection(('localhost', PORT), timeout=3))
        except OSError as exc:
            raise ConnectionError(
                f'无法连接本机 localhost:{PORT}，请确认 SSH 转发已启动（{exc}）'
            ) from exc
        channel = stack.enter_context(connection.makefile('rwb'))
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print('已连接树莓派，开始连续读取主臂位置。', flush=True)
        if getattr(args, 'preview', True):
            threading.Thread(target=open_camera_preview, daemon=True,
                             name='camera-preview-browser').start()
        if keyboard is not None:
            keyboard_worker = threading.Thread(target=keyboard.run_console, args=(stop,),
                                               name='pc-keyboard')
            keyboard_worker.start()
            stack.callback(keyboard_worker.join)
            stack.callback(stop.set)
        while True:
            request = read_frame(channel)
            if recording is not None:
                recording.receive(request)
            if keyboard is not None:
                keyboard.receive(request)
            with lock:
                started, sample = latest
            age = time.monotonic_ns() - started
            if 'error' not in sample and age > MAX_SAMPLE_AGE * 1e9:
                sample = {'error': '主臂最新采样已过期'}
            reply = {**sample, 'sequence': request['sequence'], 'sample_age_ns': age}
            if keyboard is not None:
                reply['chassis'] = keyboard.command()
            if recording is not None:
                reply['recording'] = recording.command()
            send_frame(channel, reply)
            if keyboard is not None and keyboard.should_quit():
                return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    leader = sub.add_parser('leader', help='在 Windows/Linux 桌面电脑上读取主臂并遥操')
    leader.add_argument('--side', choices=(*SIDES, 'both'), default='right')
    leader.add_argument('--monitor', action='store_true', help='本地显示读数，不连接树莓派')
    leader.add_argument('--keyboard', action='store_true', help='Windows/Linux 本地键盘控制底盘、升降；连接后直接跟随')
    leader.add_argument('--no-preview', dest='preview', action='store_false', default=True,
                        help='不自动弹出四路预览；默认由客户端建立独立17883数据通道')
    leader.add_argument('--robot-host', default=os.environ.get('XYC_ROBOT_HOST', 'raspberrypi.local'),
                        help='树莓派SSH主机；默认raspberrypi.local')
    leader.add_argument('--robot-user', default=os.environ.get('XYC_ROBOT_USER', 'pi5'),
                        help='树莓派SSH用户；默认pi5')
    leader.add_argument('--record', action='store_true', help='遥操同时采集LeRobot v3；后端需 recording:=true')
    leader.add_argument('--data-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'data' / 'datasets',
                        help='本地电脑的数据根目录，默认项目下 data/datasets；支持Windows/Linux路径')
    leader.add_argument('--dataset', help='数据集名称，例如 xyc/pick_toys；默认按电脑当前时间新建')
    leader.add_argument('--task', default='', help='采集任务描述，可使用中文或英文')
    leader.add_argument('--fps', type=int, default=30, help='数据集采样频率1～30，默认30')
    limit = leader.add_mutually_exclusive_group()
    limit.add_argument('--episode-seconds', type=float, default=30., help='每回合名义时长，默认30秒')
    limit.add_argument('--episode-frames', type=int, help='每回合有效帧数，优先直接指定帧数')
    leader.add_argument('--episodes', type=int, default=0, help='本次最多保存多少回合；0为手动不限')
    leader.add_argument('--record-scope', choices=('mobile', 'arms'), default='mobile',
                        help='默认整机18维；arms仅双臂14维')
    leader.add_argument('--resume', action='store_true', help='显式追加到同名同配置数据集，保留已保存回合')
    args = parser.parse_args()
    if args.mode == 'leader' and args.keyboard and args.monitor:
        parser.error('--keyboard 请连接树莓派使用，不能同时指定 --monitor')
    if args.record:
        if args.monitor:
            parser.error('--record 不能与 --monitor 同时使用')
        import math
        if (not args.task.strip() or len(args.task.encode()) > 512 or not 1 <= args.fps <= 30
                or not 0 <= args.episodes <= 10000 or not math.isfinite(args.episode_seconds)):
            parser.error('--record 需要 --task；fps为1..30，episodes为0..10000')
        frames = args.episode_frames if args.episode_frames is not None else round(args.episode_seconds * args.fps)
        if not 2 <= frames <= 108000:
            parser.error('每回合有效帧数必须在2..108000之间')
        args.record_config = {'dataset': args.dataset or time.strftime('xyc/demonstrations_%Y%m%d_%H%M%S'),
                              'task': args.task, 'fps': args.fps, 'episode_frames': frames,
                              'scope': args.record_scope, 'resume': args.resume,
                              'num_episodes': args.episodes}
        args.keyboard = True
    try:
        run_leader(args)
    except KeyboardInterrupt:
        print('已停止遥操流。')
        return 130
    except (OSError, ValueError, EOFError) as exc:
        print(f'遥操流已停止：{exc}', flush=True)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
