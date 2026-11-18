"""Timing and framing shared by the PC reader and the Pi state bridge."""

from xyc.robot.profile import robot_profile
# SPDX-License-Identifier: Apache-2.0

import json
import time

from xyc.contracts import SIDES, checked_mobile_action


PERIOD = 1 / 70
PORT = robot_profile().ports['teleop']
REPLY_DEADLINE = .18
INPUT_LOSS_DEADLINE = .35
MAX_SAMPLE_AGE = .15
TARGET_COMMAND_EXPIRED = 'Target command expired'
TELEOP_TARGET_DROPPED = 'Teleop target expired before commit'


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
    if not isinstance(payload, dict):
        raise ValueError('底盘指令格式无效')
    commands = payload.get('commands')
    if not isinstance(commands, list):
        raise ValueError('底盘指令必须使用四轴三值命令，请确认遥操两端版本一致')
    commands = checked_mobile_action(commands)
    operation = payload.get('operation')
    if operation is not None and (
            not isinstance(operation, dict) or type(operation.get('id')) is not int
            or not 0 < operation['id'] < 2**64
            or operation.get('name') not in ('enable', 'stop', 'home_lift')):
        raise ValueError('底盘操作序号或名称无效')
    return commands, operation
