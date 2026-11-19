"""The evaluation reader reuses collection math without owning follower motors."""

import time
import threading

from xyc.teleop import input as module


def test_leaders_are_only_sampled_during_preparation_and_ports_close(monkeypatch):
    readers, samples = [], []
    class Reader:
        def __init__(self, spec, **kwargs):
            assert kwargs['allow_release'] is False
            self.spec, self.closed = spec, False
            readers.append(self)
        def open(self): pass
        def close(self): self.closed = True
        def settings(self): return {}
    monkeypatch.setattr(module, 'LeaderCalibrationReader', Reader)
    monkeypatch.setattr(module, 'open_spec', lambda side: {'side': side, 'port': 'fake-' + side})
    monkeypatch.setattr(module, 'load_calibration', lambda *args: {})
    def sample(reader, *args):
        samples.append(reader.spec['side'])
        values = [10.] * 7 if reader.spec['side'] == 'left' else [20.] * 7
        return values, time.monotonic_ns(), []
    monkeypatch.setattr(module, 'leader_stream_sample', sample)
    source = module.LeaderArms()
    source.open()
    try:
        time.sleep(.02)
        assert samples == [] and source.fresh_snapshot() is None
        source.resume()
        frame = None
        deadline = time.monotonic() + .5
        while frame is None and time.monotonic() < deadline:
            frame = source.fresh_snapshot()
            time.sleep(.005)
        assert frame.positions == {'left': [10.] * 7, 'right': [20.] * 7}
        assert not frame.reused_positions
        source.pause()
        assert source.fresh_snapshot() is None
        time.sleep(.03)
        count = len(samples)
        time.sleep(.03)
        assert len(samples) == count
    finally:
        source.close()
    assert all(reader.closed for reader in readers)


def test_bad_left_frame_cannot_overlap_two_reads_on_right_bus(monkeypatch):
    class Reader:
        def __init__(self, spec, **kwargs): self.spec = spec
        def open(self): pass
        def close(self): pass
        def settings(self): return {}
    monkeypatch.setattr(module, 'LeaderCalibrationReader', Reader)
    monkeypatch.setattr(module, 'open_spec', lambda side: {'side': side, 'port': 'fake-' + side})
    monkeypatch.setattr(module, 'load_calibration', lambda *args: {})
    lock = threading.Lock()
    active, overlaps = set(), []
    def sample(reader, *args):
        side = reader.spec['side']
        if side == 'left':
            raise ValueError('synthetic bad frame')
        with lock:
            if side in active: overlaps.append(side)
            active.add(side)
        time.sleep(.04)
        with lock: active.discard(side)
        return [0.] * 7, time.monotonic_ns(), []
    monkeypatch.setattr(module, 'leader_stream_sample', sample)
    source = module.LeaderArms()
    source.open()
    try:
        source.resume()
        time.sleep(.16)
    finally:
        source.close()
    assert overlaps == []
