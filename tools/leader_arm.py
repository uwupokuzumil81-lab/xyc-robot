"""Read and calibrate USB leader arms on Windows or Linux, independently of ROS."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ros_ws' / 'src' / 'xyc_hardware'))
sys.path.insert(0, str(ROOT / '.setup' / 'windows-python'))

from serial.tools.list_ports import comports

from xyc_hardware.arm_calibration import (
    PERIOD, CalibrationReader, RangeRecorder, arm_names, integer, normalize_tick,
    stable_reference,
)
from xyc_hardware.feetech_reader import decode_feedback


USB_SERIALS = {'left': '5B79017577', 'right': '5B79018606'}
PROFILE = 'am-leader-6dof'
JOINT_LABELS = ('肩水平', '肩抬', '肘', '腕俯仰', '腕偏航', '腕旋转', '夹爪')
MIN_SPAN = 128
VOLTAGE_ALARM = 0x01


def servo_error_detail(motor_id, reply):
    detail = f'ID {motor_id}: 0x{reply.servo_error:02x}'
    if reply.servo_error & VOLTAGE_ALARM and len(reply.data) == 15:
        volts = decode_feedback(reply.data)['voltage_v']
        detail += f'（电压状态异常，反馈 {volts:.1f} V）'
    return detail


class LeaderCalibrationReader(CalibrationReader):
    def read_frame(self):
        return super().read_feedback()

    def read_feedback(self):
        for _ in range(3):
            replies = self.read_frame()
            errors = [servo_error_detail(i, r) for i, r in replies.items() if r.servo_error]
            if errors:
                raise ValueError('舵机报告错误：' + '、'.join(errors))
            missing = [name for name, (motor_id, _) in self.spec['motors'].items()
                       if motor_id not in replies]
            if not missing:
                return replies
        raise ConnectionError(f'连续 3 次未收到完整反馈（{self.spec["port"]}），最后缺少：'
                              + '、'.join(missing))

    def read_stream_feedback(self):
        """Reread each pure voltage alarm once on its own servo ID."""
        for _ in range(3):
            replies = self.read_frame()
            if all(i in replies for i in self.motor_ids):
                break
        else:
            missing = [str(i) for i in self.motor_ids if i not in replies]
            raise ConnectionError('连续 3 次未收到完整反馈，最后缺少 ID ' + '、'.join(missing))
        alarms = {}
        for motor_id, reply in replies.items():
            if reply.servo_error == VOLTAGE_ALARM:
                alarms[motor_id] = servo_error_detail(motor_id, reply)
                self._request(motor_id, 2, b'\x38\x0f')
                fresh = self._receive((motor_id,), 15, .012).get(motor_id)
                if fresh is not None:
                    replies[motor_id] = fresh
        return replies, alarms


def open_spec(arm):
    serial = USB_SERIALS[arm]
    matches = [port.device for port in comports()
               if (port.vid, port.pid, port.serial_number) == (0x1A86, 0x55D3, serial)]
    if len(matches) != 1:
        raise ConnectionError(f'{arm} 主臂 USB 序列号 {serial} 匹配到 {len(matches)} 个串口')
    return {
        'arm': arm, 'port': matches[0], 'baudrate': 1_000_000,
        'motors': {name: (motor_id, 777) for motor_id, name in enumerate(arm_names(arm), 1)},
    }


def require_torque_off(settings):
    enabled = [name for name, info in settings.items() if info['torque_enabled']]
    if enabled:
        raise ValueError(f'检测到关节已上力，停止手动标定：{", ".join(enabled)}')


def identity(settings):
    return {name: {key: value for key, value in info.items() if key != 'torque_enabled'}
            for name, info in settings.items()}


def load_calibration(path, arm, settings):
    if not path.is_file():
        raise ValueError(f'请先标定 {arm} 主臂，未找到 {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    if (data['schema_version'] != 1 or data['profile'] != PROFILE or data['arm'] != arm
            or data['usb_serial'] != USB_SERIALS[arm] or data['ticks_per_revolution'] != PERIOD
            or data['hardware'] != identity(settings) or set(data['ranges']) != set(arm_names(arm))):
        raise ValueError('标定文件与当前主臂或电机设置不匹配')
    for name, entry in data['ranges'].items():
        integer(entry['reference_tick'], 0, PERIOD - 1)
        low = integer(entry['min_delta_ticks'], 1 - PERIOD, 0)
        high = integer(entry['max_delta_ticks'], 0, PERIOD - 1)
        if not MIN_SPAN <= high - low < PERIOD:
            raise ValueError(f'{name}: 标定跨度无效')
    return data['ranges']


def save_calibration(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         suffix='.tmp', delete=False) as output:
            temp = Path(output.name)
            json.dump(data, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write('\n')
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


class TerminalTable:
    """Redraw in place on Windows consoles with virtual terminal support."""

    def __enter__(self):
        import ctypes
        from ctypes import wintypes
        import msvcrt

        self.line_count = 0
        self.restore = None
        if sys.stdout.isatty() and shutil.get_terminal_size().columns >= 74:
            console = ctypes.WinDLL('kernel32', use_last_error=True)
            console.GetConsoleMode.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
            console.SetConsoleMode.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            handle = msvcrt.get_osfhandle(sys.stdout.fileno())
            mode = wintypes.DWORD()
            if (console.GetConsoleMode(handle, ctypes.byref(mode))
                    and console.SetConsoleMode(handle, mode.value | 0x0004)):
                self.restore = (console.SetConsoleMode, handle, mode.value)
        return self

    def show(self, lines):
        if self.restore is None:
            print('\n'.join(lines), flush=True)
        else:
            if self.line_count:
                sys.stdout.write(f'\x1b[{self.line_count}A')
            sys.stdout.write(''.join('\r\x1b[2K' + line + '\n' for line in lines))
            sys.stdout.flush()
        self.line_count = len(lines)

    def __exit__(self, *_):
        if self.restore is not None:
            set_mode, handle, old_mode = self.restore
            set_mode(handle, old_mode)


def calibration_sample(reader, names):
    replies = reader.read_frame()
    positions, rows = {}, {}
    for motor_id, name in enumerate(names, 1):
        reply = replies.get(motor_id)
        if reply is None:
            rows[name] = ('--', '--', 'TIMEOUT')
            continue
        feedback = decode_feedback(reply.data)
        tick = feedback['position_ticks']
        status = 'OK'
        if reply.servo_error:
            kind = 'VOLT' if reply.servo_error & 0x01 else 'ERROR'
            status = f'{kind} 0x{reply.servo_error:02x}'
        elif not 0 <= tick < PERIOD:
            status = 'RANGE'
        else:
            positions[name] = tick
        rows[name] = (str(tick), f'{feedback["voltage_v"]:.1f}', status)
    return positions, rows


def calibration_table(recorder, rows, note, last_alarm=''):
    lines = ['ID  JOINT              RAW    MIN    NOW    MAX   SPAN     V  STATUS']
    for motor_id, name in enumerate(recorder.reference, 1):
        joint = name.split('_', 2)[2]
        tick, volts, status = rows[name]
        lo, hi = recorder.minimum[name], recorder.maximum[name]
        lines.append(f'{motor_id:>2}  {joint:<15} {tick:>6} {lo:>6} {recorder.delta[name]:>6} '
                     f'{hi:>6} {hi-lo:>6} {volts:>5}  {status}')
    lines.extend((note, last_alarm))
    return lines


def calibrate(reader, arm, settings, path, overwrite):
    import msvcrt

    if not sys.stdin.isatty():
        raise ValueError('请在 Windows PowerShell 终端中运行标定')
    if path.exists() and not overwrite:
        raise ValueError(f'已有标定 {path}。明确重做时加 --overwrite')
    require_torque_off(settings)
    input('把主臂放到自然中位，夹爪保持半开，扶稳后按回车记录参考姿态：')
    reference = stable_reference(reader)
    recorder = RangeRecorder(reference)
    names = arm_names(arm)
    print('依次慢慢拨动 6 个关节和夹爪，覆盖安全可用范围。记录完按回车查看汇总，Ctrl+C 取消。')
    print('RAW 是当前原始编码器值；MIN/NOW/MAX 是相对参考姿态的有效记录。')
    last_valid = time.monotonic()
    deadline = last_valid + 600
    next_display = 0.0
    interrupted = False
    hint = ''
    last_alarm = ''
    previous_status = ()
    with TerminalTable() as display:
        while time.monotonic() < deadline:
            positions, rows = calibration_sample(reader, names)
            now = time.monotonic()
            complete = len(positions) == len(names)
            statuses = tuple(row[2] for row in rows.values())
            for motor_id, (_, volts, status) in enumerate(rows.values(), 1):
                if status.startswith(('VOLT', 'ERROR')):
                    last_alarm = f'最近告警：ID {motor_id} {status} ({volts} V)'
            interrupted |= now - last_valid > 1.0
            if complete and not interrupted:
                recorder.update(positions)
                last_valid = now

            note = hint or '采集中。扫完按回车。'
            if interrupted:
                note = '反馈中断超过 1 秒，范围已冻结。Ctrl+C 后重新标定。'
            elif not complete:
                note = '反馈异常，暂停记录。恢复后请补扫该段。'
            if now >= next_display or statuses != previous_status:
                display.show(calibration_table(recorder, rows, note, last_alarm))
                next_display = now + .5
                previous_status = statuses
            if msvcrt.kbhit():
                key = msvcrt.getwch()
                if key == '\x03':
                    raise KeyboardInterrupt
                if key == '\r' and complete and not interrupted:
                    incomplete = [str(i) for i, name in enumerate(names, 1)
                                  if recorder.maximum[name] - recorder.minimum[name] < MIN_SPAN]
                    if not incomplete:
                        display.show(calibration_table(recorder, rows, '范围记录完成。', last_alarm))
                        break
                    hint = '活动范围不足，请继续扫 ID ' + '、'.join(incomplete)
                    next_display = 0.0
            time.sleep(.025)
        else:
            raise TimeoutError('标定超过 10 分钟，本次结果不保存')

    if input('确认已覆盖安全可用范围后，输入 save 并回车保存：').strip().lower() != 'save':
        raise KeyboardInterrupt

    final_settings = reader.settings()
    require_torque_off(final_settings)
    if identity(final_settings) != identity(settings):
        raise ValueError('标定过程中电机设置变化，请重新标定')
    data = {
        'schema_version': 1, 'profile': PROFILE, 'arm': arm, 'usb_serial': USB_SERIALS[arm],
        'ticks_per_revolution': PERIOD, 'hardware': identity(settings),
        'ranges': {name: {'reference_tick': reference[name],
                          'min_delta_ticks': recorder.minimum[name],
                          'max_delta_ticks': recorder.maximum[name]} for name in names},
    }
    save_calibration(path, data)
    print(f'标定已保存：{path}')


def show_positions(reader, arm, ranges, watch):
    names = arm_names(arm)
    if ranges is not None:
        print('标定读数按主臂编码器递增方向显示，单位为标定单位。')
    while True:
        raw = reader.positions()
        display = []
        for label, name in zip(JOINT_LABELS, names):
            tick = raw[name]
            if ranges is None:
                display.append(f'{label}={tick}')
                continue
            entry = dict(ranges[name], direction=1, normalized_max=100,
                         normalized_min=0 if name.endswith('_gripper') else -100)
            try:
                value = f'{normalize_tick(entry, tick):.1f}'
            except ValueError:
                value = '超出标定范围'
            display.append(f'{label}={value}')
        print('  '.join(display), flush=True)
        if not watch:
            return
        time.sleep(.25)




def main():
    parser = argparse.ArgumentParser(description='Windows/Linux 主臂只读检查和软件标定')
    parser.add_argument('arm', choices=USB_SERIALS)
    parser.add_argument('action', choices=('read', 'calibrate', 'preview'))
    parser.add_argument('--watch', action='store_true', help='持续显示读数，Ctrl+C 退出')
    parser.add_argument('--overwrite', action='store_true', help='重新标定并替换该侧的标定文件')
    args = parser.parse_args()
    path = ROOT / 'config' / 'leader_calibration' / f'{args.arm}.json'
    reader = None
    try:
        spec = open_spec(args.arm)
        reader = LeaderCalibrationReader(spec, allow_release=False)
        reader.open()
        settings = reader.settings()
        on = sum(info['torque_enabled'] for info in settings.values())
        print(f'{args.arm} 主臂 {spec["port"]}，USB {USB_SERIALS[args.arm]}，7/7 舵机在线，上力 {on}/7')
        if args.action == 'calibrate':
            calibrate(reader, args.arm, settings, path, args.overwrite)
        else:
            ranges = load_calibration(path, args.arm, settings) if args.action == 'preview' else None
            show_positions(reader, args.arm, ranges, args.watch)
    except KeyboardInterrupt:
        print('\n已退出。')
        return 130
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f'操作失败：{exc}', file=sys.stderr)
        return 1
    finally:
        if reader is not None:
            reader.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
