"""Model-independent manual preparation and repeated live evaluations."""

from xyc.teleop.keyboard import MobileKeyboard, MOTIONS

import math
import time

from xyc.terminal import TerminalKeys
from .live_robot import validate_live_state




class EvaluationConsole:
    """Manual controls share PolicyExecutor; no model, serial or ROS logic."""

    def __init__(self, schema, calibration_ids, *, arm_input=None, keys=None, clock=time.monotonic, sleep=time.sleep):
        self.schema, self.calibration_ids = schema, calibration_ids
        self.keys = keys if keys is not None else TerminalKeys()
        self.clock, self.sleep = clock, sleep
        self.arm_input = arm_input
        self.mobile = MobileKeyboard(clock=clock)
        self.message = '先调整机器人与场景，准备好后按 P 开始模型推理。'
        self.drawn_at = -math.inf
        self.rounds = self.successes = self.failures = 0

    def open(self):
        self.keys.open()
        try:
            if self.arm_input is not None:
                self.arm_input.open()
        except BaseException:
            self.keys.close()
            raise

    def close(self):
        try:
            if self.arm_input is not None:
                self.arm_input.close()
        finally:
            self.keys.close()
        print('\n已退出评估界面。', flush=True)

    def event(self, executor, name, **details):
        recorder = executor.runner.recorder
        if recorder is not None:
            recorder.event(name, **details)

    def show(self, state, *, model=False, steps=0):
        now = self.clock()
        if now - self.drawn_at < .2:
            return
        self.drawn_at = now
        values = state.get('state', [None] * 18)
        labels = ('肩水平', '肩抬', '肘', '腕俯仰', '腕偏航', '腕旋转', '夹爪')
        mode = '模型推理' if model else '遥操准备' if self.arm_input is not None else '键盘准备'
        lines = ['\x1b[2J\x1b[H=== XYC 通用机器人评估 ===',
                 f'模式：{mode} | 模型执行 {steps} 帧 | 已开始 {self.rounds} 轮',
                 '关节位置为标定单位；升降高度为毫米。', '编号  关节            左臂        右臂']
        def number(value):
            return f'{value:9.1f}' if isinstance(value, (int, float)) and math.isfinite(value) else '       --'
        for index, label in enumerate(labels):
            lines.append(f'  {index + 1}   {label:<6}'
                         f'{number(values[index])} {number(values[index + 7])}')
        height = values[14]
        lift = ('升降：回零中，空格可停止' if state.get('chassis', {}).get('homing') else
                f'升降：{height * 1000:.1f} mm' if height is not None else '升降：未回零，按 H 回零')
        preparation = ('遥操准备：双臂跟随主臂。' if self.arm_input is not None else
                       '纯键盘准备：双臂保持当前位置，不需要连接主臂。')
        lines += [lift, preparation,
                  '主臂：' + self.arm_input.notice if self.arm_input is not None else '',
                  'W/S 前后，A/D 平移，J/L 旋转，I/K 升降；空格停车，H 下行回零。',
                  'P 开始/暂停模型；推理中 Y 标成功、N 标失败并回准备；Q 退出。',
                  '切换模式不重启驱动；回零前确保下行路径畅通。',
                  f'人工评估：成功 {self.successes} / 失败 {self.failures}（未标注轮次不计成功率）',
                  '提示：' + self.message]
        print('\n'.join(lines), flush=True)

    def _hold(self, executor, state):
        validate_live_state(state, self.calibration_ids)
        result = executor.execute_action([*state['state'][:14], 0., 0., 0., 0.], record=False)
        if result.accepted is not True:
            raise RuntimeError('手动目标未被驱动确认接受')

    def prepare(self, executor):
        """Return True on deliberate start, False on quit. No model call here."""
        if self.arm_input is not None:
            self.arm_input.resume()
        try:
            return self._prepare(executor)
        finally:
            if self.arm_input is not None:
                self.arm_input.pause()

    def _prepare(self, executor):
        sink = executor.command_sink
        state = sink.status()
        targets = list(state['state'][:14])
        self.mobile.reset()
        next_tick = self.clock()
        period = 1. / self.schema.fps
        self.event(executor, 'manual_preparation_started')
        while True:
            now = self.clock()
            state = sink.status()
            homing = bool(state['chassis'].get('homing'))
            if self.arm_input is not None:
                leader_targets = self.arm_input.targets(state['state'][:14])
                if leader_targets is not None:
                    targets = leader_targets
            for key in self.mobile.poll(self.keys):
                if key == 'q':
                    return False
                if key == 'p':
                    if homing or not state['chassis'].get('enabled'):
                        self.message = '请先等待回零完成和底盘使能反馈。'
                        continue
                    if state['state'][14] is None:
                        self.message = '升降尚未建立零点；先按 H 回零，再调整高度，最后按 P。'
                        continue
                    try:
                        self._hold(executor, state)
                    except ValueError as exc:
                        self.message = str(exc)
                        continue
                    self.rounds += 1
                    self.event(executor, 'evaluation_started', round=self.rounds, state=state['state'])
                    return True
                if key == 'h' and not homing:
                    self.mobile.reset()
                    sink.home_lift()
                    homing = True
                    self.message = '升降下行回零中；空格可停止。'
                elif key == ' ':
                    self.mobile.reset()
                    if homing:
                        sink.halt()
                        sink.start()
                        homing = False
                    self.message = '底盘和升降已停车。'
                elif key in MOTIONS and not homing:
                    if state['state'][14] is None:
                        self.message = '先按 H 完成升降回零。'
                        continue
                    self.mobile.press(key, now)
            if homing:
                sink.heartbeat()
            elif state['state'][14] is not None and state['chassis'].get('enabled'):
                validate_live_state(state, self.calibration_ids)
                command = self.mobile.values(now)
                result = executor.execute_action([*targets, *command], record=False)
                if result.accepted is not True:
                    raise RuntimeError('手动目标未被驱动确认接受')
            self.show(state)
            next_tick = max(next_tick + period, self.clock())
            self.sleep(max(0., next_tick - self.clock()))

    def model_keys(self, executor, state, steps):
        """Pause and freeze on P/Y/N, quit on Q/space; no concurrent owner."""
        self.show(state, model=True, steps=steps)
        for key in self.mobile.poll(self.keys):
            if key in ('p', 'y', 'n', 'q', ' '):
                self._hold(executor, state)
                success = True if key == 'y' else False if key == 'n' else None
                self.successes += int(success is True)
                self.failures += int(success is False)
                self.event(executor, 'evaluation_finished', round=self.rounds,
                           task_success=success, model_frames=steps)
                self.message = '模型已暂停，保持当前位置；可调整场景，再按 P 开始下一轮。'
                return 'quit' if key == 'q' else 'prepare'
        return 'model'
