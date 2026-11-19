"""One read-only leader sampler shared by collection and evaluation preparation."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
import threading
import time

from xyc.io import content_id
from xyc.contracts import SIDES
from xyc.transport.teleop import MAX_SAMPLE_AGE
from .leader import (ROOT, USB_SERIALS, LEADER_SAMPLE_PERIOD, LeaderCalibrationReader,
                     arm_names, identity, leader_stream_sample, load_calibration,
                     open_spec, require_torque_off)


@dataclass(frozen=True)
class LeaderFrame:
    sample_id: int
    stamp_ns: int
    positions: dict
    calibration_ids: dict
    notice: str = ''
    error: str = ''
    reused_positions: bool = False


class LeaderArms:
    def __init__(self, sides=SIDES):
        self.sides = tuple(sides)
        if not self.sides or len(set(self.sides)) != len(self.sides) or set(self.sides) - set(SIDES):
            raise ValueError('Choose left, right or both leader arms')
        self.inputs, self.calibration_ids = {}, {}
        self.lock = threading.Condition()
        self.active, self.stopping = threading.Event(), threading.Event()
        self.latest = None
        self.stack = self.worker = None
        self.generation = self.sequence = 0

    @property
    def notice(self):
        with self.lock:
            return (self.latest.error or self.latest.notice or '主臂反馈正常') if self.latest else '等待主臂首帧'

    def open(self):
        self.stack = ExitStack()
        try:
            for side in self.sides:
                reader = LeaderCalibrationReader(open_spec(side), allow_release=False)
                self.stack.callback(reader.close)
                reader.open()
                settings = reader.settings()
                require_torque_off(settings)
                ranges = load_calibration(ROOT / 'config/leader_calibration' / f'{side}.json', side, settings)
                self.inputs[side] = reader, ranges, arm_names(side), {}
                self.calibration_ids[side] = content_id(
                    {'usb': USB_SERIALS[side], 'hardware': identity(settings), 'ranges': ranges})
                print(f'{side} 主臂已就绪 ({reader.spec["port"]})', flush=True)
            self.pool = self.stack.enter_context(ThreadPoolExecutor(max_workers=len(self.sides)))
            self.worker = threading.Thread(target=self._collect, name='leader-sampling', daemon=True)
            self.worker.start()
        except BaseException:
            self.stack.close()
            raise

    def _sample(self, side):
        try:
            return leader_stream_sample(*self.inputs[side])
        except (OSError, ValueError) as exc:
            raise ValueError(f'{side} 主臂：{exc}') from exc

    def _collect(self):
        while not self.stopping.is_set():
            if not self.active.wait(.05):
                continue
            started = time.monotonic()
            with self.lock:
                generation = self.generation
            positions, notices, error = {}, [], ''
            batch_started_ns = stamp = time.monotonic_ns()
            jobs = {side: self.pool.submit(self._sample, side) for side in self.sides}
            for side, job in jobs.items():
                try:
                    values, sampled_at, messages = job.result()
                    positions[side] = values
                    stamp = min(stamp, sampled_at)
                    notices.extend(f'{side} 主臂：{message}' for message in messages)
                except (OSError, ValueError) as exc:
                    error = str(exc)
            with self.lock:
                if generation == self.generation and self.active.is_set() and not self.stopping.is_set():
                    self.sequence += 1
                    self.latest = LeaderFrame(self.sequence, stamp, positions,
                                              dict(self.calibration_ids), '；'.join(notices), error,
                                              reused_positions=stamp < batch_started_ns)
                    self.lock.notify_all()
            self.stopping.wait(max(0., LEADER_SAMPLE_PERIOD - (time.monotonic() - started)))

    def resume(self):
        with self.lock:
            self.generation += 1
            self.latest = None
        self.active.set()

    def pause(self):
        self.active.clear()
        with self.lock:
            self.generation += 1

    def snapshot(self):
        with self.lock:
            return self.latest

    def wait_next(self, sample_id, timeout=.5):
        with self.lock:
            self.lock.wait_for(lambda: self.stopping.is_set() or
                               self.latest is not None and self.latest.sample_id > sample_id, timeout)
            return self.latest

    def fresh_snapshot(self):
        frame = self.snapshot()
        if (not self.active.is_set() or frame is None or frame.error
                or not 0 <= time.monotonic_ns() - frame.stamp_ns <= MAX_SAMPLE_AGE * 1e9):
            return None
        return frame

    def close(self):
        self.stopping.set()
        self.active.set()
        with self.lock:
            self.lock.notify_all()
        if self.worker is not None:
            self.worker.join(timeout=2.)
        if self.stack is not None:
            self.stack.close()
        self.active.clear()
        with self.lock:
            self.latest = None
