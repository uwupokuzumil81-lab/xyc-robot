"""USB leader discovery, calibration and read-only streaming shared by all frontends."""

from xyc.robot.profile import robot_profile

import json
import time

from xyc.paths import project_root

ROOT = project_root()

from serial.tools.list_ports import comports

from xyc.robot.calibration import (
    PERIOD, CalibrationReader, arm_names, integer, normalize_tick,
)
from xyc.robot.feetech import decode_feedback


USB_SERIALS = robot_profile().leaders
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
        raise ValueError(f'主臂关节仍在上力，请先让主臂脱力：{", ".join(enabled)}')


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


# Same stream policy for dataset collection and evaluation preparation.
LEADER_SAMPLE_PERIOD = 1 / 80
LEADER_STARTUP_MARGIN_TICKS = 256
LEADER_VOLTAGE_HOLD_NS = 80_000_000


class InvalidLeaderFrame(ValueError):
    """One unusable leader frame; no follower command should be generated."""


def leader_position(entry, tick, is_gripper, previous_delta=None):
    encoder_period = PERIOD

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
