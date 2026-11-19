"""Model-independent manual preparation and repeated live evaluations."""

from xyc.teleop.keyboard import MobileKeyboard, MOTIONS
from xyc.teleop.mapping import AlignedFollow, ALIGNMENT_FRAMES

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
        self.follow = self.paused_targets = None
        self.mobile = MobileKeyboard(clock=clock)
        self.model_start_key = 'p'
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
        waiting = not model and self.follow is not None and not self.follow.engaged
        mode = ('模型推理' if model else '等待主臂对齐' if waiting else
                '主臂遥操' if self.arm_input is not None else '键盘准备')
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
        if waiting:
            errors = self.follow.errors
            lines.append(f'主臂对齐 {self.follow.matches}/{ALIGNMENT_FRAMES} 帧。差值表示主臂还应移动多少。')
            if errors is not None:
                lines.append('关节          左主臂        右主臂')
                for index, label in enumerate(labels):
                    lines.append(f'{label:<6} {errors[index]:+9.1f} {errors[index + 7]:+12.1f}')
        lines += [lift,
                  '主臂：' + self.arm_input.notice if self.arm_input is not None else '',
                  'W/S 前后，A/D 平移，J/L 旋转，I/K 升降；空格停车，H 下行回零。',
                  'P 开始或暂停模型，R 从 P 暂停恢复；Y/N 标成功或失败；Q 退出。',
                  '切换模式不重启驱动；回零前确保下行路径畅通。',
                  f'人工评估：成功 {self.successes} / 失败 {self.failures}（未标注轮次不计成功率）',
                  '提示：' + self.message]
        print('\n'.join(lines), flush=True)

    @staticmethod
    def _send_manual(executor, targets, mobile):
        result = executor.execute_action([*targets, *mobile], record=False)
        if result.accepted is not True:
            raise RuntimeError('手动目标未被驱动确认接受')

    def _hold(self, executor):
        state = executor.command_sink.hold()
        return validate_live_state(state, self.calibration_ids)[:14]

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
        start_key = self.model_start_key
        freeze_mobile = self.paused_targets is not None
        targets = self.paused_targets if freeze_mobile else list(state['state'][:14])
        self.paused_targets = None
        self.follow = AlignedFollow() if self.arm_input is not None else None
        self.mobile.reset()
        next_tick = self.clock()
        period = 1. / self.schema.fps
        self.event(executor, 'manual_preparation_started')
        while True:
            now = self.clock()
            state = sink.status()
            homing = bool(state['chassis'].get('homing'))
            aligning = self.follow is not None and not self.follow.engaged
            keys = self.mobile.poll(self.keys)
            if 'q' in keys:
                return False
            for key in keys:
                if key == start_key:
                    if homing or not state['chassis'].get('enabled'):
                        self.message = '请先等待回零完成和底盘使能反馈。'
                        continue
                    if state['state'][14] is None:
                        self.message = f'升降尚未建立零点；先按 H 回零，再调整高度，最后按 {start_key.upper()}。'
                        continue
                    try:
                        self._hold(executor)
                    except ValueError as exc:
                        self.message = str(exc)
                        continue
                    self.rounds += 1
                    self.event(executor, 'evaluation_started', round=self.rounds, state=state['state'])
                    return True
                if freeze_mobile and aligning and (key == 'h' or key in MOTIONS):
                    self.message = f'对齐期间底盘和升降保持停止。对齐后可继续遥操，或按 {start_key.upper()} 恢复模型。'
                    continue
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
                follower = validate_live_state(state, self.calibration_ids)[:14]
                if self.follow is not None:
                    leader_targets = self.follow.update(self.arm_input.fresh_snapshot(), follower)
                    if leader_targets is not None:
                        targets = leader_targets
                        if aligning:
                            self.message = f'主臂已对齐，开始绝对跟随。按 {start_key.upper()} 回到模型推理。'
                            self.event(executor, 'takeover_engaged', round=self.rounds,
                                       alignment_errors=self.follow.errors)
                    elif not aligning:
                        targets, freeze_mobile = follower, True
                        self.mobile.reset()
                        self.message = '主臂反馈中断，机器人原位保持。等待重新对齐。'
                self._send_manual(executor, targets, self.mobile.values(now))
            else:
                sink.heartbeat()  # The interactive preparation loop is alive, even before lift homing.
            self.show(state)
            next_tick = max(next_tick + period, self.clock())
            self.sleep(max(0., next_tick - self.clock()))

    def model_keys(self, executor, steps, *, display=True):
        """Handle the operator key before the next model action."""
        sink = executor.command_sink
        if display and self.clock() - self.drawn_at >= .2:
            self.show(sink.status(), model=True, steps=steps)
        key = self.read_model_key()
        return self.apply_model_key(executor, steps, key) if key is not None else 'model'

    def read_model_key(self):
        keys = self.mobile.poll(self.keys)
        if 'q' in keys:
            return 'q'
        return next((key for key in keys if key in ('p', 'y', 'n', ' ')), None)

    def apply_model_key(self, executor, steps, key):
        if key != 'q':
            targets = self._hold(executor)
            if self.arm_input is not None:
                self.paused_targets = targets
        success = True if key == 'y' else False if key == 'n' else None
        self.successes += int(success is True)
        self.failures += int(success is False)
        self.event(executor, 'evaluation_finished', round=self.rounds, task_success=success, model_frames=steps)
        if key == 'q':
            return 'quit'  # The deployment cleanup sends the single stop.
        # Terminals can report repeated P presses without a separate key-up event.
        self.model_start_key = 'r' if key == 'p' else 'p'
        start_key = self.model_start_key.upper()
        self.message = (f'模型已暂停。请摆动主臂对齐；按 {start_key} 可直接恢复模型。'
                        if self.arm_input is not None else
                        f'模型已暂停，保持当前位置；可调整场景，再按 {start_key} 开始推理。')
        return 'prepare'
