# SPDX-License-Identifier: Apache-2.0
"""Arm-only passive calibration; retain the servo's existing EEPROM zero.

The feedback node remains read-only. This separate, exclusive-port tool can
release torque on IDs 1..7 after an interactive prompt. It cannot enable torque,
send goals, or write EEPROM. Software reference/ranges are not URDF calibration.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import select
import shutil
import statistics
import sys
import tempfile
import time
from uuid import uuid4

from xyc_hardware.feetech_reader import FeetechReader, decode_feedback, signed_magnitude


JOINTS = (
    'shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex',
    'wrist_yaw', 'wrist_roll', 'gripper',
)
PERIOD = 4096
# Measured positions can drift slightly beyond a passively recorded endpoint.
FEEDBACK_ENDPOINT_TOLERANCE_TICKS = 16


def arm_names(arm):
    if arm not in ('left', 'right'):
        raise ValueError('arm must be left or right')
    return tuple(f'arm_{arm}_{joint}' for joint in JOINTS)


def integer(value, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'Expected integer in [{minimum}, {maximum}], got {value!r}')
    return value


def load_arm_spec(path, arm):
    import yaml

    names = arm_names(arm)
    config = yaml.safe_load(Path(path).read_text())
    params = config[f'/motors/{arm}/motor_state']['ros__parameters']
    ids, configured_names, models = (
        params['motor_ids'], params['motor_names'], params['model_numbers']
    )
    if not len(ids) == len(configured_names) == len(models):
        raise ValueError('Motor configuration arrays must have equal lengths')
    if len(set(ids)) != len(ids) or len(set(configured_names)) != len(configured_names):
        raise ValueError('Duplicate motor names or IDs in configuration')
    by_name = dict(zip(configured_names, zip(ids, models)))
    motors = {name: by_name[name] for name in names}
    if tuple(value[0] for value in motors.values()) != tuple(range(1, 8)):
        raise ValueError('This arm profile requires exactly arm IDs 1..7')
    if tuple(value[1] for value in motors.values()) != (
        2825, 2569, 2569, 2825, 2825, 2825, 2825
    ):
        raise ValueError('This calibration profile is for the verified AlohaMini2Pro arms')
    return {
        'arm': arm, 'port': params['port'],
        'baudrate': params.get('baudrate', 1_000_000), 'motors': motors,
    }


class CalibrationReader(FeetechReader):
    """Restricted calibration transport: extra reads plus arm torque-OFF only."""

    def __init__(self, spec, *, allow_release=False):
        self.spec = spec
        if tuple(spec['motors']) != arm_names(spec['arm']):
            raise ValueError('Expected all seven named arm motors, in order')
        ids = [pair[0] for pair in spec['motors'].values()]
        if ids != list(range(1, 8)):
            raise ValueError('Calibration must never include wheel/lift IDs')
        super().__init__(spec['port'], ids, baudrate=spec['baudrate'], timeout_s=0.03)
        self.allow_release = allow_release

    def _request(self, motor_id, instruction, parameters):
        extra_read = instruction == 2 and parameters in (b'\x1f\x03', b'\x28\x01')
        torque_off = (
            self.allow_release and instruction == 3 and parameters == b'\x28\x00'
        )
        if motor_id not in self.motor_ids or not (extra_read or torque_off):
            return super()._request(motor_id, instruction, parameters)
        body = bytes([motor_id, len(parameters) + 2, instruction]) + parameters
        packet = b'\xff\xff' + body + bytes([(~sum(body)) & 255])
        self.port.reset_input_buffer()
        if self.port.write(packet) != len(packet):
            raise OSError('Incomplete calibration request transmission')

    def _read(self, motor_id, address, size):
        self._request(motor_id, 2, bytes([address, size]))
        reply = self._receive((motor_id,), size, 0.1).get(motor_id)
        if reply is None or reply.servo_error:
            raise ConnectionError(f'ID {motor_id}: failed to read register {address}')
        return reply.data

    def settings(self):
        models = self.read_models()
        result = {}
        for name, (motor_id, model) in self.spec['motors'].items():
            if models[motor_id] != model:
                raise ValueError(f'{name}: motor model does not match configuration')
            data = self._read(motor_id, 31, 3)
            offset = signed_magnitude(int.from_bytes(data[:2], 'little'), 11)
            integer(offset, -2047, 2047)
            if data[2] != 0:
                raise ValueError(f'{name}: expected single-turn position mode 0')
            torque = self._read(motor_id, 40, 1)[0]
            if torque not in (0, 1):
                raise ValueError(f'{name}: unsupported torque state {torque}')
            result[name] = {
                'id': motor_id, 'model_number': model,
                'hardware_homing_offset': offset, 'operating_mode': 0,
                'torque_enabled': bool(torque),
            }
        return result

    def release_arm(self):
        if not self.allow_release:
            raise ValueError('Read-only inspection cannot release torque')
        failures = []
        for motor_id in self.motor_ids:
            try:
                self._request(motor_id, 3, b'\x28\x00')
                # Some status-return configurations omit write acknowledgements.
                # Readback, not an absent acknowledgement, determines success.
                reply = self._receive((motor_id,), 0, 0.03).get(motor_id)
                if reply is not None and reply.servo_error:
                    raise ConnectionError(f'Servo error {reply.servo_error}')
                if self._read(motor_id, 40, 1) != b'\x00':
                    raise ConnectionError('Torque-off readback failed')
            except (OSError, ValueError, ConnectionError) as exc:
                failures.append(f'ID {motor_id}: {exc}')
        if failures:
            raise RuntimeError('Arm torque release incomplete: ' + '; '.join(failures))

    def positions(self):
        replies = self.read_feedback()
        positions = {}
        for name, (motor_id, _) in self.spec['motors'].items():
            reply = replies.get(motor_id)
            if reply is None or reply.servo_error:
                raise ConnectionError(f'{name}: no valid fresh feedback')
            positions[name] = integer(decode_feedback(reply.data)['position_ticks'], 0, 4095)
        return positions


class RangeRecorder:
    """Track encoder wrap without mistaking it for a nearly-full-turn range."""

    def __init__(self, reference):
        self.reference = dict(reference)
        self.last = dict(reference)
        self.delta = dict.fromkeys(reference, 0)
        self.minimum = dict(self.delta)
        self.maximum = dict(self.delta)

    def update(self, positions):
        if positions.keys() != self.reference.keys():
            raise ValueError('A sample must contain all seven arm motors')
        # Validate the entire sample before updating any joint.
        steps = {}
        for name, tick in positions.items():
            integer(tick, 0, 4095)
            step = (tick - self.last[name] + PERIOD // 2) % PERIOD - PERIOD // 2
            if abs(step) > 512:
                raise ValueError(f'{name}: large sample jump; restart and move more slowly')
            delta = self.delta[name] + step
            if max(self.maximum[name], delta) - min(self.minimum[name], delta) >= PERIOD:
                raise ValueError(f'{name}: range spans a full encoder turn; ambiguous position')
            steps[name] = step
        for name, step in steps.items():
            self.last[name] = positions[name]
            self.delta[name] += step
            self.minimum[name] = min(self.minimum[name], self.delta[name])
            self.maximum[name] = max(self.maximum[name], self.delta[name])


def validate_document(document):
    if document['schema_version'] != 1 or document['ticks_per_revolution'] != PERIOD:
        raise ValueError('Unsupported arm calibration schema')
    names = arm_names(document['arm'])
    if set(document['motors']) != set(names):
        raise ValueError('Calibration must contain exactly the seven selected arm motors')
    for motor_id, name in enumerate(names, 1):
        entry = document['motors'][name]
        if integer(entry['id'], 1, 7) != motor_id:
            raise ValueError(f'{name}: incorrect motor ID')
        expected_model = 2569 if motor_id in (2, 3) else 2825
        if entry['model_number'] != expected_model or entry['operating_mode'] != 0:
            raise ValueError(f'{name}: unsupported motor model/mode')
        integer(entry['hardware_homing_offset'], -2047, 2047)
        integer(entry['reference_tick'], 0, 4095)
        minimum = integer(entry['min_delta_ticks'], -4095, 0)
        maximum = integer(entry['max_delta_ticks'], 0, 4095)
        if not 128 <= maximum - minimum < PERIOD:
            raise ValueError(f'{name}: range must span 128..4095 ticks')
        if type(entry['direction']) is not int or entry['direction'] not in (-1, 1):
            raise ValueError(f'{name}: direction must be +1 or -1')
        expected_min = 0 if name.endswith('_gripper') else -100
        if entry['normalized_min'] != expected_min or entry['normalized_max'] != 100:
            raise ValueError(f'{name}: unsupported normalization range')
    return document


def check_hardware(document, settings, spec):
    validate_document(document)
    if document['arm'] != spec['arm'] or document['port'] != spec['port']:
        raise ValueError('Calibration belongs to a different arm/serial device')
    for name in arm_names(spec['arm']):
        for key in ('id', 'model_number', 'hardware_homing_offset', 'operating_mode'):
            if document['motors'][name][key] != settings[name][key]:
                raise ValueError(f'{name}: saved {key} differs from hardware; do not use this file')


def normalize_tick(entry, tick, *, feedback_tolerance_ticks=0):
    """Convert a measured tick, optionally accepting small feedback excursions."""
    integer(tick, 0, 4095)
    delta = tick - entry['reference_tick']
    minimum, maximum = entry['min_delta_ticks'], entry['max_delta_ticks']
    first = math.ceil((minimum - delta) / PERIOD)
    last = math.floor((maximum - delta) / PERIOD)
    if first > last and feedback_tolerance_ticks:
        first = math.ceil((minimum - feedback_tolerance_ticks - delta) / PERIOD)
        last = math.floor((maximum + feedback_tolerance_ticks - delta) / PERIOD)
    if first != last:
        raise ValueError('Position is outside the calibrated range or is ambiguous')
    bounded_delta = min(max(delta + first * PERIOD, minimum), maximum)
    fraction = (bounded_delta - minimum) / (maximum - minimum)
    if entry['direction'] == -1:
        fraction = 1 - fraction
    return entry['normalized_min'] + fraction * (entry['normalized_max'] - entry['normalized_min'])


def denormalize_tick(entry, value):
    """Inverse conversion only; this function never transmits a motor command."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('Model action must be a finite number')
    if not entry['normalized_min'] <= value <= entry['normalized_max']:
        raise ValueError('Model action is outside the calibrated normalization range')
    fraction = (value - entry['normalized_min']) / (entry['normalized_max'] - entry['normalized_min'])
    if entry['direction'] == -1:
        fraction = 1 - fraction
    delta = entry['min_delta_ticks'] + fraction * (
        entry['max_delta_ticks'] - entry['min_delta_ticks']
    )
    return round(entry['reference_tick'] + delta) % PERIOD


def position_target_tick(entry, current_tick, position):
    """Preserve the complete absolute goal and reject an ambiguous encoder path."""
    target = denormalize_tick(entry, position)
    current = normalize_tick(entry, current_tick,
                             feedback_tolerance_ticks=FEEDBACK_ENDPOINT_TOLERANCE_TICKS)
    span = entry['max_delta_ticks'] - entry['min_delta_ticks']
    expected_delta = ((position - current) * span * entry['direction'] /
                      (entry['normalized_max'] - entry['normalized_min']))
    # An actual crossing of the encoder wrap differs by one full revolution.
    # Endpoint feedback tolerance and rounding do not imply such a crossing.
    if abs((target - current_tick) - expected_delta) > FEEDBACK_ENDPOINT_TOLERANCE_TICKS + 1:
        raise ValueError('Target crosses encoder wrap')
    return target


def make_document(spec, settings, reference, minimum, maximum, *, source):
    motors = {}
    for name in arm_names(spec['arm']):
        motors[name] = {
            key: value for key, value in settings[name].items() if key != 'torque_enabled'
        }
        motors[name].update(
            reference_tick=reference[name], min_delta_ticks=minimum[name],
            max_delta_ticks=maximum[name], direction=1,
            normalized_min=0 if name.endswith('_gripper') else -100, normalized_max=100,
        )
    return validate_document({
        'schema_version': 1, 'arm': spec['arm'], 'port': spec['port'],
        'ticks_per_revolution': PERIOD,
        'coordinate_system': 'servo_reference_and_range_not_urdf',
        'source': source,
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'motors': motors,
    })


def import_legacy(spec, legacy):
    """Retain existing LeRobot-style encoder ranges and direction conventions."""
    settings, reference, minimum, maximum = {}, {}, {}, {}
    for name, (motor_id, model) in spec['motors'].items():
        item = legacy[name]
        if item['id'] != motor_id:
            raise ValueError(f'{name}: legacy ID mismatch')
        lower = integer(item['range_min'], 0, 4095)
        upper = integer(item['range_max'], 0, 4095)
        if lower >= upper:
            raise ValueError(f'{name}: invalid legacy range')
        integer(item['drive_mode'], 0, 1)
        settings[name] = {
            'id': motor_id, 'model_number': model, 'operating_mode': 0,
            'hardware_homing_offset': integer(item['homing_offset'], -2047, 2047),
        }
        # A reference inside the range is sufficient; this is not a physical zero.
        reference[name] = (lower + upper) // 2
        minimum[name], maximum[name] = lower - reference[name], upper - reference[name]
    document = make_document(spec, settings, reference, minimum, maximum, source='legacy_import')
    for name in spec['motors']:
        document['motors'][name]['direction'] = -1 if legacy[name]['drive_mode'] else 1
    return validate_document(document)


def save_document(path, document):
    validate_document(document)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError('Calibration output must not be a symbolic link')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            backup = path.with_name(f'{path.stem}.backup-{stamp}-{uuid4().hex[:8]}.json')
            shutil.copy2(path, backup)
            print(f'旧文件已备份：{backup}')
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def stable_reference(reader):
    samples = [reader.positions()]
    for _ in range(9):
        time.sleep(0.03)
        samples.append(reader.positions())
    reference = {}
    for name, first in samples[0].items():
        deltas = [(sample[name] - first + 2048) % PERIOD - 2048 for sample in samples]
        if max(deltas) - min(deltas) > 8:
            raise ValueError(f'{name}: reference posture moved; restart while holding it still')
        reference[name] = (first + round(statistics.median(deltas))) % PERIOD
    return reference


class RangeDisplay:
    """Redraw a small table in place; redirected output gets summaries only."""

    def __init__(self, arm):
        self.prefix = f'arm_{arm}_'
        self.interactive = (
            sys.stdout.isatty() and os.environ.get('TERM') != 'dumb'
            and shutil.get_terminal_size().columns > 43
        )
        self.rendered = False

    def show(self, recorder, *, final=False):
        if self.rendered and not self.interactive and not final:
            return
        lines = [f'{"JOINT":<15} {"MIN":>6} {"NOW":>6} {"MAX":>6} {"SPAN":>6}']
        for name in recorder.reference:
            span = recorder.maximum[name] - recorder.minimum[name]
            lines.append(
                f'{name.removeprefix(self.prefix):<15} '
                f'{recorder.minimum[name]:>6} {recorder.delta[name]:>6} '
                f'{recorder.maximum[name]:>6} {span:>6}'
            )
        if self.interactive:
            if self.rendered:
                sys.stdout.write(f'\033[{len(lines)}A')
            sys.stdout.write(''.join('\r\033[2K' + line + '\n' for line in lines))
            sys.stdout.flush()
        else:
            print('\n'.join(lines), flush=True)
        self.rendered = True


def record(reader, spec, before):
    print('只处理所选手臂 ID 1..7。请先托住手臂，脱力后可能下落。')
    if input('准备好后输入 release 并回车，让这条手臂脱力：').strip() != 'release':
        raise KeyboardInterrupt
    reader.release_arm()
    print('该臂已脱力；本程序不会重新使能扭矩。')
    input('将手臂放在安全、远离限位的自然姿态并保持不动（无需精确中位），回车记录参考姿态：')
    reference = stable_reference(reader)
    recorder = RangeRecorder(reference)
    print('依次缓慢活动这条手臂的 7 个关节（含夹爪），记录安全可用范围。')
    print('不要强推机械限位或缠绕线缆；记录全部关节后，按一次回车结束。')
    print('MIN/NOW/MAX 为相对参考姿态的计数；SPAN 为已记录的活动跨度。')
    display = RangeDisplay(spec['arm'])
    next_display, deadline = 0.0, time.monotonic() + 600
    while time.monotonic() < deadline:
        recorder.update(reader.positions())
        if time.monotonic() >= next_display:
            display.show(recorder)
            next_display = time.monotonic() + 0.5
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line == '':
                raise EOFError('标定终端已断开；本次结果不保存')
            if line.strip() == '':
                break
        time.sleep(0.025)
    else:
        raise TimeoutError('标定超过 10 分钟，已取消；本次结果不保存')
    if not display.interactive:
        display.show(recorder, final=True)
    after = reader.settings()
    for name in reference:
        for key in ('id', 'model_number', 'hardware_homing_offset', 'operating_mode'):
            if before[name][key] != after[name][key]:
                raise ValueError(f'{name}: hardware coordinates changed during calibration')
        if after[name]['torque_enabled']:
            raise ValueError(f'{name}: torque unexpectedly enabled; cancel calibration')
    return make_document(
        spec, before, reference, recorder.minimum, recorder.maximum, source='passive_recording',
    )


def main(args=None):
    parser = argparse.ArgumentParser(description='左右臂标定：默认只检查，--record 才交互采集')
    parser.add_argument('--arm', required=True, choices=('left', 'right'))
    parser.add_argument('--config', type=Path, help='默认使用本包已安装的 motors.yaml')
    parser.add_argument('--output', type=Path, help='默认 ~/pi05_from_zero/config/calibration/left.json 或 right.json')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true', help='只读检查硬件与已有标定（默认）')
    modes.add_argument('--record', action='store_true', help='交互脱力、记录参考姿态与活动范围')
    modes.add_argument('--import-file', type=Path, help='仅导入旧标定 JSON，不打开串口')
    options = parser.parse_args(args)
    reader = None
    try:
        if options.record and not sys.stdin.isatty():
            raise ValueError('实际标定必须在实机旁的交互终端运行')
        if options.config is None:
            from ament_index_python.packages import get_package_share_directory
            options.config = Path(get_package_share_directory('xyc_hardware')) / 'config/motors.yaml'
        spec = load_arm_spec(options.config, options.arm)
        output = options.output or (
            Path.home() / 'pi05_from_zero/config/calibration' / f'{options.arm}.json'
        )
        if options.import_file is not None:
            legacy = json.loads(options.import_file.expanduser().read_text())
            document = import_legacy(spec, legacy)
            document['legacy_source'] = str(options.import_file.expanduser().resolve())
            save_document(output, document)
            print(f'已导入 {options.arm} 臂 7 个电机：{output}')
            print('尚未实读核对 EEPROM；使用前运行同一命令的 --check 模式。')
            return
        reader = CalibrationReader(spec, allow_release=options.record)
        reader.open()
        before = reader.settings()
        print(f'{options.arm} 臂：{spec["port"]}')
        for name, item in before.items():
            print(f'{name}: ID={item["id"]}, offset={item["hardware_homing_offset"]}, '
                  f'torque={int(item["torque_enabled"])}')
        if not options.record:
            if output.exists():
                check_hardware(json.loads(output.read_text()), before, spec)
                print(f'已有标定与硬件匹配：{output}')
            else:
                print(f'硬件读取正常；尚无本项目标定文件：{output}')
            return
        document = record(reader, spec, before)
        print(json.dumps(document, ensure_ascii=False, indent=2))
        if input('确认已覆盖安全可用范围后，输入 save 保存：').strip() != 'save':
            raise KeyboardInterrupt
        save_document(output, document)
        print(f'已保存：{output}；舵机 EEPROM 零点保持原值。')
        print('手臂仍处于脱力状态，请托住或放稳。')
    except KeyboardInterrupt:
        print('\n已取消；若已执行脱力，手臂保持脱力。', file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, EOFError) as exc:
        parser.exit(1, f'标定未完成：{exc}\n已有文件保持原值；若已脱力，请托住手臂。\n')
    finally:
        if reader is not None:
            reader.close()


if __name__ == '__main__':
    main()
