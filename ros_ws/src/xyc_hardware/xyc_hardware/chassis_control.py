"""Velocity axes on the existing left bus, with a local command watchdog."""

from xyc.robot.profile import robot_profile
# SPDX-License-Identifier: Apache-2.0

from collections import deque
import json
import math
import time

from rclpy.qos import QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from xyc_interfaces.msg import MobileCommand, MobileState

from xyc.robot.calibration import integer
from xyc.transport.teleop import MotionSessionEnded
from xyc_hardware.arm_joint_control import JointDriver
from xyc_hardware.motor_safety import (
    CurrentProtection, FEEDBACK_LOSS_GRACE_S,
    MOTOR_TEMPERATURE_LIMIT_C, confirm_temperature,
)
from xyc_hardware.base_kinematics import OmniGeometry
from xyc.robot.feetech import decode_feedback
from xyc_hardware.lift_tracking import LiftGeometry, LiftTracker
from xyc.contracts import (
    COMMAND_VALIDITY_S, LIFT_SOFT_STOP_MARGIN_M, MOBILE_ACTION_SCALE, STATE_NAMES, STATE_UNITS, CommandExpired,
    checked_mobile_action, physical_mobile_action,
)


AXIS_IDS = (8, 9, 10, 11)
COMMAND_TIMEOUT_S = .30
ENABLE_STATIONARY_WINDOW_S = .15
ENABLE_STATIONARY_RANGE_TICKS = 2
HOMING_SPEED_M_S = .010
HOMING_CONTACT_WINDOW_S = .10
HOMING_CONTACT_RANGE_TICKS = 2
STATIONARY_MAX_SAMPLE_GAP_S = .060
COMMAND_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
COMMAND_RESULT_QOS = QoSProfile(depth=16, reliability=ReliabilityPolicy.RELIABLE)
CHASSIS_NOT_READY = 'chassis_not_ready: '


def mobile_message(node, commands, *, stamp=None, expires_ns=None):
    """Build the sole mobile command format used by every frontend."""
    message = MobileCommand()
    message.header.stamp = stamp if stamp is not None else node.get_clock().now().to_msg()
    message.header.frame_id = 'base_link'
    generated = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
    expires_ns = generated + int(COMMAND_VALIDITY_S * 1e9) if expires_ns is None else expires_ns
    message.expires_at.sec, message.expires_at.nanosec = divmod(expires_ns, 1_000_000_000)
    message.commands = [int(v) for v in checked_mobile_action(commands)]
    return message


class ChassisNotReady(ValueError):
    """Preparation must wait; no new torque-enable attempt was dispatched."""


class ChassisDriver(JointDriver):
    """Extend the bus owner with mode/torque/velocity access to IDs 8..11."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.bus != 'left' or self.motor_ids != tuple(range(1, 12)):
            raise ValueError('Chassis requires the complete left bus')
        self.velocity_owned = set()

    def axis_read(self, motor_id, address, size):
        if motor_id not in AXIS_IDS or (address, size) not in (
                (18, 1), (33, 1), (40, 1), (41, 1), (46, 2), (55, 1), (56, 15), (63, 1), (85, 1)):
            raise ValueError('Unsupported chassis register read')
        detail = '未收到回复'
        for _ in range(3):
            self._send_joint_packet(motor_id, 2, bytes([address, size]))
            reply = self._receive((motor_id,), size, .025).get(motor_id)
            if reply is None:
                continue
            if reply.servo_error:
                raise ConnectionError(
                    f'底盘/升降 ID {motor_id} 寄存器 {address} 舵机告警 0x{reply.servo_error:02x}')
            if len(reply.data) != size or address == 40 and reply.data not in (b'\x00', b'\x01'):
                detail = '回复内容无效：' + reply.data.hex()
                continue
            return reply.data
        raise ConnectionError(f'底盘/升降 ID {motor_id} 寄存器 {address} 读取失败（3次只读请求）：{detail}')

    def axis_write(self, motor_id, address, data):
        allowed = (address in (33, 40, 55) and data in (b'\x00', b'\x01'))
        allowed |= address in (41, 85) and data == b'\xfe'
        if motor_id not in AXIS_IDS or not allowed:
            raise ValueError('Unsupported chassis configuration write')
        self._send_joint_packet(motor_id, 3, bytes([address]) + data, moves=address == 40 and data == b'\x01')
        reply = self._receive((motor_id,), 0, .02).get(motor_id)
        if reply is not None and reply.servo_error:
            raise ConnectionError(f'底盘/升降 ID {motor_id} 写入报错')
        actual = self.axis_read(motor_id, address, len(data))
        if actual != data:
            raise ConnectionError(
                f'底盘/升降 ID {motor_id} 寄存器 {address} 核对失败：目标 {data.hex()}，实际 {actual.hex()}'
            )

    def write_velocities(self, values, *, deadline=None):
        if not values or any(i not in AXIS_IDS for i in values):
            raise ValueError('Velocity writes require chassis IDs 8..11')
        encoded = {}
        for motor_id, speed in values.items():
            limit = 1300 if motor_id == 11 else 3000
            integer(speed, -limit, limit)
            encoded[motor_id] = (abs(speed) | (0x8000 if speed < 0 else 0)).to_bytes(2, 'little')
        ids = tuple(sorted(encoded))
        payload = b''.join(bytes([i]) + encoded[i] for i in ids)
        if deadline is not None and time.monotonic() >= deadline and any(values.values()):
            raise CommandExpired('Mobile command expired before serial dispatch')
        self._send_joint_packet(254, 0x83, b'\x2e\x02' + payload, deadline=deadline, moves=any(values.values()))
        self._send_joint_packet(254, 0x82, b'\x2e\x02' + bytes(ids))
        replies = self._receive(ids, 2, .025)
        failed = [i for i in ids if i not in replies or replies[i].servo_error
                  or replies[i].data != encoded[i]]
        if failed:
            raise ConnectionError(f'底盘/升降速度目标核对失败，ID {failed}')

    def enable_velocity_axes(self):
        if self.velocity_owned:
            raise ValueError('底盘/升降已经由本节点上力')
        # Complete the torque-off check before changing any configuration.
        for motor_id in AXIS_IDS:
            if self.axis_read(motor_id, 40, 1) != b'\x00':
                raise ValueError(f'ID {motor_id} 扭矩寄存器为1，但本节点尚未认领；请核对上次启动清理结果')
            if not self.axis_read(motor_id, 18, 1)[0] & 0x04:
                raise ValueError(f'ID {motor_id} 需要精细速度单位，请先核对硬件配置')
        for motor_id in AXIS_IDS:
            if self.axis_read(motor_id, 33, 1) != b'\x01':
                self.axis_write(motor_id, 55, b'\x00')
                try:
                    self.axis_write(motor_id, 33, b'\x01')
                finally:
                    self.axis_write(motor_id, 55, b'\x01')
            self.axis_write(motor_id, 85, b'\xfe')
            self.axis_write(motor_id, 41, b'\xfe')
        self.write_velocities(dict.fromkeys(AXIS_IDS, 0))
        try:
            for motor_id in AXIS_IDS:
                self.velocity_owned.add(motor_id)
                self.axis_write(motor_id, 40, b'\x01')
        except BaseException as exc:
            try:
                self.release_velocity_axes()
            except Exception as cleanup:
                raise RuntimeError(f'使能未完成：{exc}；清理未确认：{cleanup}；'
                                   f'待清理ID {sorted(self.velocity_owned)}') from exc
            raise

    def enable_lift(self):
        if 11 not in self.velocity_owned:
            if self.axis_read(11, 40, 1) != b'\x00':
                raise ValueError('升降已被其他控制器上力')
            self.write_velocities({11: 0})
            self.velocity_owned.add(11)
            self.axis_write(11, 40, b'\x01')

    def release_velocity_axes(self, ids=None):
        selected = sorted(self.velocity_owned if ids is None else self.velocity_owned.intersection(ids))
        errors = []
        for motor_id in selected:
            try:
                self.write_velocities({motor_id: 0})
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
            try:
                self.axis_write(motor_id, 40, b'\x00')
                self.velocity_owned.discard(motor_id)
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError('; '.join(errors))


class ChassisControl:
    def __init__(self, driver, geometry, lift_geometry, *, clock=time.monotonic):
        self.driver, self.geometry, self.lift_geometry = driver, geometry.validate(), lift_geometry.validate()
        self.clock = clock
        self.enabled = False
        self.fault = ''
        self.reason = '等待本地控制自动准备，底盘和升降初始速度为零'
        self.samples = {}
        self.standstill_samples = deque()
        self.last_feedback = None
        self.feedback_paused = False
        self.missing_feedback_ids = ()
        self.missing_feedback_detail = ''
        self.base = (0., 0., 0.)
        self.lift_speed = 0.
        self.mobile_until = 0.
        self.tracker = None
        self.height = None
        self.homing = ''
        self.home_started = self.home_stationary = self.settle_started = None
        self.home_tick = None
        self.home_until = 0.
        self.lift_progress_tick = None
        self.lift_progress_at = 0.
        self.home_travel = self.settle_position = 0
        self.home_position = 0
        self.home_samples = deque()
        self.settle_samples = deque()
        self.sent = {}
        self.current_protection = {
            i: CurrentProtection(2569 if i == 11 else 2825) for i in AXIS_IDS
        }
        self.next_torque_check = 0.
        self.max_base_speed = math.hypot(*MOBILE_ACTION_SCALE[1:3])
        self.max_yaw_speed = .80
        self.max_lift_speed = .020

    def enable(self):
        if self.enabled:
            return
        if self.last_feedback is None or self.clock() - self.last_feedback > FEEDBACK_LOSS_GRACE_S:
            raise ChassisNotReady('等待底盘和升降的新鲜反馈')
        if self.driver.velocity_owned:
            # Only clean axes this same driver attempted to enable. Never adopt
            # an unowned motor merely because its torque register reads one.
            self.driver.release_velocity_axes()
        ready, reason = self.stationary_status()
        if not ready:
            raise ChassisNotReady(reason)
        for motor_id in AXIS_IDS:
            sample = self.samples[motor_id]
            self.check_temperature(motor_id, sample)
        try:
            self.driver.enable_velocity_axes()
        except (OSError, ValueError, RuntimeError) as exc:
            self.fail('底盘/升降使能未完成：' + str(exc))
            raise RuntimeError(self.reason) from exc
        self.enabled, self.fault = True, ''
        self.base = (0., 0., 0.)
        self.lift_speed = 0.
        self.sent = dict.fromkeys(AXIS_IDS, 0)
        self.reset_current_protection()
        self.reason = '已使能；升降先按 H 回零，再用 I/K 控制'
        self.next_torque_check = self.clock() + .5

    def reset_current_protection(self):
        for monitor in self.current_protection.values():
            monitor.reset()

    def stationary_status(self):
        rows = list(self.standstill_samples)
        if (len(rows) < 2 or self.clock() - rows[-1][0] > FEEDBACK_LOSS_GRACE_S
                or rows[-1][0] - rows[0][0] < ENABLE_STATIONARY_WINDOW_S):
            return False, '等待连续0.15秒的完整底盘/升降静止反馈'
        for motor_id in AXIS_IDS:
            origin = rows[0][1][motor_id]
            positions = [(row[1][motor_id] - origin + 2048) % 4096 - 2048 for row in rows]
            span = max(positions) - min(positions)
            if span > ENABLE_STATIONARY_RANGE_TICKS:
                return False, f'使能前ID {motor_id}仍在移动：编码器窗口跨度 {span} 刻度'
        return True, ''

    def set_mobile(self, commands, *, deadline):
        """Preflight and apply a single four-axis target under one deadline."""
        lift, x, y, yaw = physical_mobile_action(commands)
        now = self.clock()
        if not math.isfinite(deadline) or now >= deadline:
            raise CommandExpired('Mobile command expired before acceptance')
        if not self.enabled or self.fault or self.feedback_paused:
            raise ValueError(self.reason or '底盘/升降未就绪')
        if self.last_feedback is None or now - self.last_feedback >= FEEDBACK_LOSS_GRACE_S:
            raise ValueError('底盘/升降反馈已过期')
        if self.homing:
            self.home_until = min(deadline, now + COMMAND_TIMEOUT_S)
            if any((lift, x, y, yaw)):
                self.reason = '升降回零中，方向目标已忽略，空格可停止'
            return
        if lift and self.tracker is None:
            raise ValueError('升降零点尚未建立，请先按 H 回零')
        if (math.hypot(x, y) > self.max_base_speed or abs(yaw) > self.max_yaw_speed
                or abs(lift) > self.max_lift_speed):
            raise ValueError('移动目标超过配置上限')
        base = self.geometry.constrain_body_velocity(x, y, yaw)
        if lift:
            self.driver.enable_lift()
        now = self.clock()
        if now >= deadline:
            self.clear_target()
            raise CommandExpired('Mobile command expired during preparation')
        if lift and lift * self.lift_speed <= 0:
            self.lift_progress_tick = self.samples[11]['position_ticks']
            self.lift_progress_at = now
        self.base, self.lift_speed = base, lift
        self.mobile_until = min(deadline, now + COMMAND_TIMEOUT_S)

    def clear_target(self):
        self.base, self.lift_speed = (0., 0., 0.), 0.
        self.mobile_until = 0.

    def zero_targets_verified(self):
        return (self.enabled and not (self.fault or self.feedback_paused or self.homing)
                and not any(self.base) and self.lift_speed == 0.
                and all(self.sent.get(axis) == 0 for axis in AXIS_IDS))

    def stop(self):
        self.clear_target()
        if self.homing:
            self.homing = ''
            self.tracker, self.height = None, None
        if self.driver.velocity_owned:
            values = dict.fromkeys(self.driver.velocity_owned, 0)
            if values:
                self.driver.write_velocities(values)
                self.sent.update(values)
        self.reason = '已发送零速度'

    def begin_home(self):
        if not self.enabled or self.fault:
            raise ValueError('先使能并检查底盘/升降状态')
        if self.feedback_paused:
            raise ValueError(self.reason)
        if self.homing:
            self.reason = ('升降正在下行回零，请等待完成；空格可停止'
                           if self.homing == 'seeking' else '升降正在等待回零稳定；空格可停止')
            return  # A repeated H must not restart motion or reset its deadline.
        if self.last_feedback is None or self.clock() - self.last_feedback > FEEDBACK_LOSS_GRACE_S:
            raise ValueError('升降反馈过期')
        self.stop()
        self.driver.enable_lift()
        self.tracker, self.height = None, None
        self.homing = 'seeking'
        self.home_started = self.clock()
        self.home_until = self.clock() + COMMAND_TIMEOUT_S
        self.home_stationary = None
        self.home_tick = self.samples[11]['position_ticks']
        self.home_travel = 0
        self.home_position = 0
        self.home_samples.clear()
        self.reason = '升降正在下行回零，确保底部无障碍；空格可停止'

    def fail(self, reason):
        self.fault = str(reason)
        try:
            self.stop()
        except (OSError, ValueError, RuntimeError) as exc:
            self.fault += '; 零速度未确认：' + str(exc)
        try:
            self.driver.release_velocity_axes()
        except (OSError, ValueError, RuntimeError) as exc:
            self.fault += '; 脱力未确认：' + str(exc)
        self.enabled = False
        self.tracker, self.height = None, None
        self.reason = '底盘/升降保护停止：' + self.fault

    def feedback_timeout_reason(self):
        age = self.clock() - self.last_feedback if self.last_feedback is not None else float('inf')
        if not self.missing_feedback_ids:
            return f'底盘/升降采样监测超时：距完整反馈 {age:.3f}s；未记录到具体缺回复的电机ID'
        ids = ','.join(map(str, self.missing_feedback_ids))
        return (f'底盘/升降反馈持续缺失：ID [{ids}]，距完整反馈 {age:.3f}s'
                + (f'；{self.missing_feedback_detail}' if self.missing_feedback_detail else ''))

    def observe(self, samples, *, missing_detail=''):
        now = self.clock()
        missing = tuple(i for i in AXIS_IDS if i not in samples)
        if missing:
            self.standstill_samples.clear()
            self.missing_feedback_ids = missing
            self.missing_feedback_detail = missing_detail
            if self.enabled:
                if self.last_feedback is None or now - self.last_feedback >= FEEDBACK_LOSS_GRACE_S:
                    self.fail(self.feedback_timeout_reason())
                    return
                # Do not drive from a stale sample. Pause velocities immediately
                # without throwing away a still-unambiguous lift reference.
                self.clear_target()
                self.home_stationary = None
                self.home_samples.clear()
                self.settle_samples.clear()
                if not self.feedback_paused:
                    zero = {i: 0 for i in self.driver.velocity_owned if self.sent.get(i) != 0}
                    try:
                        if zero:
                            self.driver.write_velocities(zero)
                            self.sent.update(zero)
                    except (OSError, ValueError, RuntimeError) as exc:
                        self.fail(f'反馈暂缺时零速度未确认：{exc}')
                        return
                self.feedback_paused = True
                self.reason = f'反馈暂缺 ID {list(missing)}，速度已暂停，等待新反馈'
            return
        if self.enabled and self.feedback_paused and now - self.last_feedback >= FEEDBACK_LOSS_GRACE_S:
            self.fail(self.feedback_timeout_reason())
            return
        recovered = self.feedback_paused
        self.feedback_paused = False
        self.missing_feedback_ids = ()
        self.missing_feedback_detail = ''
        if self.last_feedback is not None and now - self.last_feedback > STATIONARY_MAX_SAMPLE_GAP_S:
            self.home_samples.clear()
            self.standstill_samples.clear()
        self.standstill_samples.append((now, {i: samples[i]['position_ticks'] for i in AXIS_IDS}))
        while (len(self.standstill_samples) > 2
               and self.standstill_samples[1][0] <= now - ENABLE_STATIONARY_WINDOW_S):
            self.standstill_samples.popleft()
        self.samples, self.last_feedback = samples, now
        if not self.enabled:
            return
        if recovered:
            self.reason = '完整反馈已恢复，等待新的速度指令'
        try:
            for motor_id in AXIS_IDS:
                sample = samples[motor_id]
                self.check_temperature(motor_id, sample)
                reason = self.current_protection[motor_id].observe(sample['current_a'], now)
                if reason:
                    raise ValueError(
                        f'ID {motor_id} 持续过流：{reason}；回零阶段 {self.homing or "无"}，'
                        f'位置 {sample["position_ticks"]}，速度读数 {sample["velocity_raw"]}'
                    )
            if now >= self.next_torque_check:
                for motor_id in self.driver.velocity_owned:
                    if self.driver.axis_read(motor_id, 40, 1) != b'\x01':
                        raise ValueError(f'ID {motor_id} 上力意外关闭')
                self.next_torque_check = now + .5
            if self.homing:
                if now >= self.home_until:
                    raise ValueError('回零控制端断开或命令超时，升降已停止')
                self._observe_home(now)
            elif self.tracker is not None:
                self.height = self.tracker.update(samples[11]['position_ticks'], now)
                if self.lift_speed and now < self.mobile_until and self.sent.get(11):
                    tick = samples[11]['position_ticks']
                    delta = (tick - self.lift_progress_tick + 2048) % 4096 - 2048
                    direction = self.lift_geometry.direction * (1 if self.lift_speed > 0 else -1)
                    if delta * direction >= 3:
                        self.lift_progress_tick, self.lift_progress_at = tick, now
                    elif now - self.lift_progress_at >= .75:
                        raise ValueError('升降持续未向指令方向移动，请检查行程端部和机构')
            base = self.base if now < self.mobile_until and not self.homing else (0., 0., 0.)
            speeds = dict(zip((8, 9, 10), (
                round(v * 4096 / math.tau) for v in self.geometry.body_to_wheels(*base)
            )))
            lift_speed = self.lift_speed if now < self.mobile_until else 0.
            if self.homing == 'seeking':
                speeds[11] = round(-self.lift_geometry.direction * HOMING_SPEED_M_S * 4096 /
                                   self.lift_geometry.lead_m_per_revolution)
            elif self.homing == 'settling' or self.height is None:
                speeds[11] = 0
            else:
                if (lift_speed < 0 and self.height <= LIFT_SOFT_STOP_MARGIN_M
                        or lift_speed > 0 and self.height >= self.lift_geometry.max_height_m - LIFT_SOFT_STOP_MARGIN_M):
                    lift_speed = 0.
                    self.reason = '升降已到软件行程边界'
                speeds[11] = round(lift_speed * 4096 /
                                   (self.lift_geometry.direction * self.lift_geometry.lead_m_per_revolution))
            changed = {i: v for i, v in speeds.items()
                       if i in self.driver.velocity_owned and self.sent.get(i) != v}
            if changed:
                deadline = None
                if any(changed.values()):
                    deadline = self.home_until if self.homing == 'seeking' else self.mobile_until
                try:
                    self.driver.write_velocities(changed, deadline=deadline)
                    if deadline is not None and self.clock() >= deadline and any(changed.values()):
                        raise CommandExpired('Mobile command expired during serial confirmation')
                except CommandExpired:
                    self.clear_target()
                    self.reason = '移动命令已过期，已停车；等待新指令'
                    changed = dict.fromkeys(self.driver.velocity_owned, 0)
                    self.driver.write_velocities(changed)
                self.sent.update(changed)
        except MotionSessionEnded:
            self.clear_target()  # The shared protection path owns the stop write.
        except (OSError, ValueError, RuntimeError) as exc:
            self.fail(str(exc))

    def check_temperature(self, motor_id, sample):
        if not all(math.isfinite(sample[k]) for k in ('current_a', 'temperature_c')):
            raise ValueError(f'ID {motor_id} 电流/温度反馈无效')
        if sample['temperature_c'] >= MOTOR_TEMPERATURE_LIMIT_C:
            confirm_temperature(
                sample, read_temperature=lambda: self.driver.axis_read(motor_id, 63, 1)[0],
                read_feedback=lambda: decode_feedback(self.driver.axis_read(motor_id, 56, 15)),
                report=self.temperature_notice,
            )

    def temperature_notice(self, message):
        self.reason = message

    def _observe_home(self, now):
        sample = self.samples[11]
        tick = sample['position_ticks']
        delta = (tick - self.home_tick + 2048) % 4096 - 2048
        if abs(delta) == 2048:
            raise ValueError('升降编码器半圈位置不明确')
        self.home_tick = tick
        self.home_travel += abs(delta)
        self.home_position += delta
        max_travel = self.lift_geometry.max_height_m - self.lift_geometry.min_height_m + .05
        max_duration = max_travel / HOMING_SPEED_M_S + 5.
        if (now - self.home_started >= max_duration
                or self.home_travel * self.lift_geometry.lead_m_per_revolution / 4096 > max_travel):
            raise ValueError('升降回零超时或超出行程，请检查机构')
        if self.homing == 'seeking':
            # Position history is authoritative for standstill. A noisy velocity
            # register must not keep driving into the lower stop. Require both
            # a bounded encoder window and continuous contact current; a slow
            # but steadily moving encoder does not establish a zero reference.
            self.home_samples.append((now, self.home_position, abs(sample['current_a'])))
            while (len(self.home_samples) > 2
                   and self.home_samples[1][0] <= now - HOMING_CONTACT_WINDOW_S):
                self.home_samples.popleft()
            rows = list(self.home_samples)
            positions = [row[1] for row in rows]
            stationary = (abs(delta) <= HOMING_CONTACT_RANGE_TICKS
                          and max(positions) - min(positions) <= HOMING_CONTACT_RANGE_TICKS)
            complete_window = rows[-1][0] - rows[0][0] >= HOMING_CONTACT_WINDOW_S
            if stationary:
                if self.home_stationary is None:
                    self.home_stationary = now
            else:
                self.home_stationary = None
            if stationary and complete_window and all(row[2] >= .3 for row in rows):
                self.driver.release_velocity_axes((11,))
                self.sent[11] = 0
                self.homing = 'settling'
                self.settle_started = now
                self.settle_position = 0
                self.settle_samples.clear()
            else:
                if self.home_stationary is not None and now - self.home_stationary >= .5:
                    raise ValueError('升降停止但没有接触电流，回零未建立')
            return
        self.settle_position += delta
        self.settle_samples.append((now, self.settle_position, abs(sample['velocity_raw'])))
        while len(self.settle_samples) > 2 and self.settle_samples[1][0] <= now - 1.:
            self.settle_samples.popleft()
        rows = list(self.settle_samples)
        if rows[-1][0] - rows[0][0] >= 1.:
            positions = [r[1] for r in rows]
            if max(positions) - min(positions) <= 10:
                self.tracker = LiftTracker(self.lift_geometry, reference_id=str(time.time_ns()),
                                           zero_tick=tick, reference_time_s=now)
                self.height, self.homing = 0., ''
                self.reason = '升降回零完成，可以按 I/K 升降'
                return
        if now - self.settle_started >= 2.5:
            raise ValueError('升降回零后未稳定，零点未建立')


class ChassisRos:
    """Attach to MotorStateNode. All serial work stays in that node's executor."""

    def __init__(self, node, driver, *, teleop_link=None):
        self.node = node
        self.teleop_link = teleop_link
        def parameter(name, default):
            return node.declare_parameter(name, default).value
        defaults = robot_profile().geometry
        geometry = OmniGeometry(
            parameter('wheel_radius_m', defaults['wheel_radius_m']), parameter('base_radius_m', defaults['base_radius_m']),
            tuple(math.radians(v) for v in parameter('wheel_angles_deg', defaults['wheel_angles_deg'])),
            tuple(parameter('wheel_signs', defaults['wheel_signs'])), 3000 * math.tau / 4096,
        )
        lift = LiftGeometry(parameter('lift_lead_m_per_revolution', defaults['lift_lead_m_per_revolution']),
                            parameter('lift_direction', defaults['lift_direction']), 4000., .2,
                            0., parameter('lift_max_height_m', defaults['lift_max_height_m']))
        self.control = ChassisControl(driver, geometry, lift)
        self.status = node.create_publisher(String, '/robot/chassis/status', 1)
        self.state_publisher = node.create_publisher(
            MobileState, '/robot/chassis/state',
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.recording_command_publisher = node.create_publisher(
            String, '/robot/chassis/recording_command', COMMAND_RESULT_QOS)
        self.pending_command = None
        self.latest_command_ns = 0
        self.motion_fault = ''
        self.command_sub = node.create_subscription(MobileCommand, '/robot/mobile/command',
                                                    self.mobile_command, COMMAND_QOS)
        self.services = [
            node.create_service(Trigger, '/robot/chassis/' + name, callback)
            for name, callback in (('enable', self.enable), ('stop', self.stop), ('home_lift', self.home))
        ]
        self.watchdog = node.create_timer(.05, self.check_feedback)
        self.last_status = 0.

    def session_fault(self):
        if self.teleop_link is not None and (fault := self.teleop_link.status()) is not None:
            return fault['reason']
        return self.motion_fault

    def check_feedback(self):
        if self.session_fault():
            return
        c = self.control
        if c.enabled and (c.last_feedback is None or c.clock() - c.last_feedback >= FEEDBACK_LOSS_GRACE_S):
            detail = (f'；上一总线操作 {getattr(self.node, "last_transaction_kind", "无")} '
                      f'{getattr(self.node, "last_transaction_duration_s", 0.):.3f}s，'
                      f'上一采样处理 {getattr(self.node, "last_feedback_cycle_duration_s", 0.):.3f}s')
            c.fail(c.feedback_timeout_reason() + detail)
            self.node.get_logger().error(c.reason)

    def valid_command(self, message):
        received = self.control.clock()
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        expires = message.expires_at.sec * 1_000_000_000 + message.expires_at.nanosec
        now = self.node.get_clock().now().nanoseconds
        if (message.header.frame_id != 'base_link' or stamp <= 0 or stamp > now
                or not 0 < expires - stamp <= round(COMMAND_VALIDITY_S * 1e9)):
            raise ValueError('移动消息时间或坐标系无效')
        if now >= expires:
            raise CommandExpired('Mobile command expired in the ROS queue')
        return received + (expires - now) / 1e9

    def mobile_command(self, message):
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if fault := self.session_fault():
            self.publish_mobile_result(message, error=fault)
            return
        if 0 < stamp <= self.latest_command_ns:
            return  # A prior command cannot undo a stop or supersede a newer target.
        try:
            deadline = self.valid_command(message)
            self.latest_command_ns = stamp
            if not self.control.enabled and not any(message.commands):
                self.publish_mobile_result(message, error=self.control.fault or self.control.reason)
                return
            self.control.set_mobile(message.commands, deadline=deadline)
        except (OSError, ValueError, RuntimeError) as exc:
            previous, self.pending_command = self.pending_command, None
            self.control.clear_target()
            self.control.reason = str(exc)
            self.publish_mobile_result(message, error=self.control.reason)
            if previous is not None:
                self.publish_mobile_result(previous, error='A rejected newer command cleared the target')
            return
        previous = self.pending_command
        self.pending_command = message if not self.control.homing else None
        if previous is not None:
            self.publish_mobile_result(previous, error='Mobile command superseded before velocity confirmation')
        if self.control.homing:
            self.publish_mobile_result(message, error='Lift homing is active')
        elif not any(message.commands) and self.control.zero_targets_verified():
            # `sent` contains readback-verified targets. Another zero needs
            # neither a bus write nor another feedback tick to confirm it.
            self.record_mobile_command()

    def record_mobile_command(self):
        """Acknowledge only after the shared controller has applied the command."""
        message, self.pending_command = self.pending_command, None
        if message is None:
            return
        if self.control.fault or self.control.feedback_paused:
            self.publish_mobile_result(message, error=self.control.fault or self.control.reason)
            return
        if self.control.clock() >= self.control.mobile_until:
            self.publish_mobile_result(message, error='Mobile command expired before velocity confirmation')
            return
        try:
            self.valid_command(message)
        except (OSError, ValueError, RuntimeError) as exc:
            self.publish_mobile_result(message, error=str(exc))
        else:
            self.publish_mobile_result(message)

    def publish_mobile_result(self, message, *, error=None):
        stamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        if stamp <= 0:
            return
        event = {'kind': 'mobile', 'sequence': stamp, 'generated_ns': stamp, 'sent_ns': stamp,
                 'targets': [int(value) for value in message.commands], 'session': self.node.group_session_id,
                 'accepted': error is None}
        if error is None:
            event['accepted_ns'] = self.node.get_clock().now().nanoseconds
        else:
            event['error'] = error
        try:
            self.recording_command_publisher.publish(String(data=json.dumps(event, separators=(',', ':'), allow_nan=False)))
        except (RuntimeError, TypeError, ValueError) as exc:
            self.node.get_logger().warning(f'Mobile command result unavailable: {exc}', throttle_duration_sec=1.)

    def protective_stop(self, reason):
        self.motion_fault = self.motion_fault or reason
        response = self.stop(None, Trigger.Response())
        if not response.success:
            raise RuntimeError(response.message)

    def enable(self, request, response):
        return self.service(self.control.enable, response)

    def stop(self, request, response):
        return self.service(self.control.stop, response, invalidate_commands=True, allow_fault=True)

    def home(self, request, response):
        return self.service(self.control.begin_home, response, invalidate_commands=True)

    def service(self, operation, response, *, invalidate_commands=False, allow_fault=False):
        if not allow_fault and (fault := self.session_fault()):
            response.message = fault
            return response
        if invalidate_commands:
            self.pending_command = None
        try:
            operation()
            response.success = True
            response.message = self.control.reason
        except (OSError, ValueError, RuntimeError) as exc:
            response.message = (CHASSIS_NOT_READY if isinstance(exc, ChassisNotReady) else '') + str(exc)
            self.control.reason = '操作被拒绝：' + str(exc)
            self.node.get_logger().warning(self.control.reason)
        finally:
            if invalidate_commands:
                self.latest_command_ns = max(self.latest_command_ns, self.node.get_clock().now().nanoseconds)
        return response

    def observe(self, message):
        samples = {m.motor_id: {k: getattr(m, k) for k in (
            'position_ticks', 'velocity_raw', 'current_a', 'temperature_c', 'moving')}
            for m in message.motors if m.motor_id in AXIS_IDS and m.valid and not m.servo_error and not m.error}
        if not self.session_fault():
            previous_fault = self.control.fault
            alarms = [m for m in message.motors if m.motor_id in AXIS_IDS and m.servo_error]
            if self.control.enabled and alarms:
                self.control.fail('; '.join(f'ID {m.motor_id} 舵机告警 0x{m.servo_error:02x}' for m in alarms))
            detail = '; '.join(f'ID {m.motor_id}: {m.error}' for m in message.motors
                               if m.motor_id in AXIS_IDS and m.motor_id not in samples and m.error)
            self.control.observe(samples, missing_detail=detail)
            self.record_mobile_command()
            if self.control.fault and self.control.fault != previous_fault:
                self.node.get_logger().error(self.control.reason)
        # The measured-state topic is an observation output, not part of the
        # motor safety loop.  A transient ROS publisher error must not tear
        # down the chassis owner and stop an otherwise healthy control path.
        try:
            self.publish_state(message, samples)
        except (RuntimeError, TypeError, ValueError) as exc:
            self.node.get_logger().warning(
                f'移动状态发布失败，底盘控制继续运行：{exc}',
                throttle_duration_sec=1.0,
            )
        now = time.monotonic()
        if now - self.last_status >= .1:
            self.last_status = now
            c = self.control
            speeds = [samples[i]['velocity_raw'] * math.tau / 4096 for i in (8, 9, 10) if i in samples]
            body = self.control.geometry.wheels_to_body(speeds) if len(speeds) == 3 else None
            payload = {'enabled': c.enabled, 'fault': c.fault or self.motion_fault, 'reason': c.reason,
                       'motion_fault': self.motion_fault,
                       'enable_ready': not self.motion_fault and c.stationary_status()[0],
                       'enable_wait_reason': c.stationary_status()[1],
                       'velocity_owned': sorted(c.driver.velocity_owned),
                       'feedback_paused': c.feedback_paused,
                       'zero_targets_verified': not self.motion_fault and c.zero_targets_verified(),
                       'missing_motor_ids': list(c.missing_feedback_ids),
                       'homing': c.homing, 'height_m': c.height, 'base_measured': body,
                       'wheel_ticks_s': [samples[i]['velocity_raw'] if i in samples else None for i in (8, 9, 10)],
                       'lift_ticks': samples[11]['position_ticks'] if 11 in samples else None}
            self.status.publish(String(data=json.dumps(payload, ensure_ascii=False, allow_nan=False)))

    def publish_state(self, feedback, samples):
        """Publish timestamped measured mobile state without enabling any motor."""
        c = self.control
        output = MobileState()
        output.header.stamp = feedback.header.stamp
        output.header.frame_id = 'base_link'
        output.names = list(STATE_NAMES[-4:])
        output.units = list(STATE_UNITS[-4:])
        output.enabled = c.enabled
        output.homing = bool(c.homing)
        problems = []
        if c.fault or self.motion_fault:
            problems.append(c.fault or self.motion_fault)
        if c.feedback_paused:
            problems.append(c.reason or 'mobile feedback paused')
        missing = [motor_id for motor_id in AXIS_IDS if motor_id not in samples]
        if missing:
            problems.append('missing motor feedback IDs ' + ','.join(map(str, missing)))
        if c.height is None:
            problems.append('lift height has no verified zero reference')
        if c.homing:
            problems.append('lift homing is active')
        body = None
        if not missing:
            body = c.geometry.wheels_to_body([
                samples[motor_id]['velocity_raw'] * math.tau / 4096
                for motor_id in (8, 9, 10)
            ])
        values = [c.height, *(body if body is not None else (None, None, None))]
        if not all(value is not None and math.isfinite(value) for value in values):
            problems.append('mobile state contains missing or nonfinite values')
        output.valid = not problems
        output.state = [float(value) if value is not None else float('nan') for value in values]
        output.error = '; '.join(problems)
        self.state_publisher.publish(output)

    def recording_snapshot(self, message):
        """Measured body velocities and height, independent of action events."""
        c = self.control
        samples = {m.motor_id: m for m in message.motors
                   if m.motor_id in AXIS_IDS and m.valid and not m.servo_error and not m.error}
        if (not c.enabled or c.fault or self.motion_fault or c.feedback_paused or c.homing
                or c.height is None or len(samples) != 4):
            return {'valid': False, 'reason': c.fault or self.motion_fault or c.reason}
        measured = c.geometry.wheels_to_body([samples[i].velocity_raw * math.tau / 4096 for i in (8, 9, 10)])
        return {'valid': True, 'state': [c.height, *measured]}

    def close(self):
        try:
            self.control.stop()
        finally:
            self.control.driver.release_velocity_axes()
