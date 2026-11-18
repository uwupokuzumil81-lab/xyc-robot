"""Stream calibrated leader positions over an SSH forward to the Pi."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import socket
import shutil
import subprocess
import sys
import threading
import time
import unicodedata


from xyc.transport.teleop import MAX_SAMPLE_AGE, PORT, SIDES, read_frame, send_frame


from xyc.terminal import TerminalKeys as ConsoleKeys
from xyc.contracts import MOBILE_ACTION_SCALE
from xyc.teleop.keyboard import MobileKeyboard, MOTIONS
from xyc.teleop.input import LeaderArms
from xyc.robot.profile import robot_profile
from xyc.paths import project_root

DATA_PORT = robot_profile().ports['data']
PREVIEW_URL = f'http://localhost:{DATA_PORT}'


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
                f'本机{self.port}端口已被占用；后端SSH只转发{PORT}，'
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
        from xyc.recording.local_recorder import LocalDatasetReceiver
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
                    self.notice = f'等待电脑采集数据通道，请检查{DATA_PORT}端口转发'
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
    print(f'\n相机预览未连接：请确认后端相机已启动，独立{DATA_PORT}数据通道保持在线。', flush=True)
    return False


class LocalChassisKeyboard:
    """PC-only operator state. Windows input and serial sampling run independently."""

    def __init__(self, recording=None):
        self.recording = recording
        self.lock = threading.Lock()
        self.mobile = MobileKeyboard()
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
                self.mobile.reset()
                name = {'h': 'home_lift', ' ': 'stop',
                        'q': 'stop', '\x03': 'stop'}[key]
                if self.operation is not None and name != 'stop':
                    self.notice = '正在等待上一个操作结果'
                    return
                self.operation = {'id': self.next_operation, 'name': name}
                self.next_operation += 1
                if key in ('q', '\x03'):
                    self.quit_at = now
            elif key in MOTIONS:
                if key in 'ik' and self.state.get('homing'):
                    self.notice = '升降正在回零，请等待完成后再按 I/K；空格可停止'
                    return
                self.mobile.press(key, now)

    def command(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            active = (now - self.received <= .5 and self.state.get('enabled')
                      and not self.state.get('homing') and not self.state.get('feedback_paused')
                      and not self.state.get('fault'))
            return {
                'commands': self.mobile.values(now) if active else [0.] * 4,
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
        lift, forward, lateral, yaw = MOBILE_ACTION_SCALE
        print(f'固定轴速度：前后 {forward:g} m/s，平移 {lateral:g} m/s，'
              f'旋转 {yaw:g} rad/s，升降 {lift:g} m/s；组合超轮速时等比例降速。')
        if self.recording is not None:
            print('采集：连接后待机，准备好按 R 开录；B 保存；N 丢弃；达到帧数自动保存。')
            print('默认整机18维；录制时树莓派临时缓存，录完传到电脑保存并校验，完成后才能R开下一回合。')
        try:
            with ConsoleKeys() as console:
                self._console_loop(stop, console)
        except KeyboardInterrupt:
            self.handle_key('\x03')  # The control thread sends STOP and waits for its acknowledgement.
        print()

    def _console_loop(self, stop, console):
        last_draw = 0.
        printed_error = ''
        while not stop.is_set():
            for key in self.mobile.poll(console):
                self.handle_key(key)
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




def run_leader(args):
    sys.setswitchinterval(.001)  # Two serial readers, network and console share one interpreter.
    recording = RecordingRemote(args.record_config) if getattr(args, 'record', False) else None
    keyboard = LocalChassisKeyboard(recording) if getattr(args, 'keyboard', False) else None
    if keyboard is not None and (sys.platform not in ('win32', 'linux') or not sys.stdin.isatty()):
        raise ValueError('--keyboard 请在本地 Windows 或 Linux 交互终端运行')
    with ExitStack() as stack:
        if getattr(args, 'preview', True) or recording is not None:
            stack.enter_context(SSHDataTunnel(
                getattr(args, 'robot_host', robot_profile().ssh_host),
                getattr(args, 'robot_user', robot_profile().ssh_user),
            ))
        # Native video/Arrow/pandas imports can take seconds on a cold desktop.
        # Finish them BEFORE opening the control socket: the Pi expects its
        # first reply within REPLY_DEADLINE, just like every later control frame.
        if recording is not None:
            print('正在准备本地数据保存，完成后连接控制通道……', flush=True)
            local = recording.start_local(getattr(args, 'data_dir', project_root() / 'data' / 'datasets'))
            stack.callback(local.close)
        leaders = LeaderArms(SIDES if args.side == 'both' else (args.side,))
        leaders.open()
        stack.callback(leaders.close)
        leaders.resume()
        if args.monitor:
            last = valid = 0
            began = time.monotonic()
            while True:
                frame = leaders.wait_next(last)
                if frame is None or frame.sample_id == last:
                    continue
                last = frame.sample_id
                if frame.error:
                    print(f'第 {last} 帧跳过：{frame.error}', flush=True)
                else:
                    valid += 1
                    if last == 1 or last % 15 == 0:
                        print(f'{last} 帧，有效 {valid} / {time.monotonic()-began:.2f}s {frame.positions}', flush=True)
        stop = threading.Event()
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
            frame = leaders.snapshot()
            age = time.monotonic_ns() - frame.stamp_ns if frame else 0
            if frame is None:
                sample = {'error': '正在等待主臂首帧'}
            elif frame.error or age > MAX_SAMPLE_AGE * 1e9:
                sample = {'error': frame.error or '主臂最新采样已过期'}
            else:
                sample = {'sample_id': frame.sample_id, 'positions': frame.positions,
                          'calibration_ids': frame.calibration_ids}
                if frame.notice and keyboard is not None:
                    keyboard.set_notice(frame.notice)
            reply = {**sample, 'sequence': request['sequence'], 'sample_age_ns': age}
            if keyboard is not None:
                reply['chassis'] = keyboard.command()
            if recording is not None:
                reply['recording'] = recording.command()
            send_frame(channel, reply)
            if keyboard is not None and keyboard.should_quit():
                return


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    leader = sub.add_parser('leader', help='在 Windows/Linux 桌面电脑上读取主臂并遥操')
    leader.add_argument('--side', choices=(*SIDES, 'both'), default='right')
    leader.add_argument('--monitor', action='store_true', help='本地显示读数，不连接树莓派')
    leader.add_argument('--keyboard', action='store_true', help='Windows/Linux 本地键盘控制底盘、升降；连接后直接跟随')
    leader.add_argument('--no-preview', dest='preview', action='store_false', default=True,
                        help='不自动弹出四路预览；数据通道端口由机器人配置决定')
    leader.add_argument('--robot-host', default=robot_profile().ssh_host,
                        help='树莓派SSH主机；默认读取机器人配置或XYC_ROBOT_HOST')
    leader.add_argument('--robot-user', default=robot_profile().ssh_user,
                        help='树莓派SSH用户；默认读取机器人配置或XYC_ROBOT_USER')
    leader.add_argument('--record', action='store_true', help='遥操同时采集LeRobot v3；后端需 recording:=true')
    leader.add_argument('--data-dir', type=Path,
                        default=project_root() / 'data' / 'datasets',
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
    args = parser.parse_args(argv)
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
