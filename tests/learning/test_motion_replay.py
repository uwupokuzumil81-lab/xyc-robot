"""Motion replay checks real command semantics with fake drivers and real TCP."""

import ast
import json
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xyc.learning import cli
from xyc.contracts import ACTION, ACTION_NAMES, ACTION_SEMANTICS, CAMERAS, OBS_STATE, STATE_NAMES, mobile_action_metadata
from xyc.learning.replay import ReplayPlan, execute
from xyc.transport import motion
from xyc.transport.motion import MotionClient


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'data'
    for directory in ('meta/episodes/chunk-000', 'data/chunk-000'):
        (root / directory).mkdir(parents=True)
    features = {OBS_STATE: {'dtype': 'float32', 'shape': [18], 'names': list(STATE_NAMES)},
                ACTION: {'dtype': 'float32', 'shape': [18], 'names': list(ACTION_NAMES)}}
    for camera in CAMERAS:
        features['observation.images.' + camera] = {'dtype': 'video', 'shape': [64, 64, 3]}
    info = {'codebase_version': 'v3.0', 'total_episodes': 1, 'fps': 30, 'features': features,
            'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4'}
    (root / 'meta/info.json').write_text(json.dumps(info))
    protocol = motion
    assets = {'state_units': protocol.STATE_UNITS, 'action_units': protocol.ACTION_UNITS,
              'action_semantics': ACTION_SEMANTICS, 'mobile_action_scale': mobile_action_metadata(),
              'calibrations': {'calibration/' + side: {'arm': side, 'test_calibration': side}
                               for side in ('left', 'right')}}
    (root / 'meta/robot.json').write_text(json.dumps(assets))
    rows = [{OBS_STATE: [0.] * 18, ACTION: [0.] * 14 + [1., 1., 0., 0.],
             'episode_index': 0, 'frame_index': index, 'observation.source_time': [.04 * index]}
            for index in range(4)]
    pq.write_table(pa.Table.from_pylist(rows), root / 'data/chunk-000/file-000.parquet')
    episode = {'episode_index': 0, 'length': 4, 'data/chunk_index': 0, 'data/file_index': 0}
    for camera in CAMERAS:
        key = 'observation.images.' + camera
        prefix = 'videos/' + key
        episode.update({prefix + '/chunk_index': 0, prefix + '/file_index': 0, prefix + '/from_timestamp': 0.})
        video = root / f'videos/{key}/chunk-000/file-000.mp4'
        video.parent.mkdir(parents=True)
        video.write_bytes(b'fingerprint-only-video-not-decoded')
    pq.write_table(pa.Table.from_pylist([episode]), root / 'meta/episodes/chunk-000/file-000.parquet')
    return root


class FakeRobot:
    def __init__(self, ids, *, height=0.):
        self.ids, self.height = ids, height
        self.arms = [0.] * 14
        self.enabled = False
        self.commands, self.stops, self.faults = [], [], []
        self.starts = self.homes = self.heartbeats = 0

    def snapshot(self):
        return {'ready': True, 'calibration_ids': self.ids, 'state': [*self.arms, self.height, 0., 0., 0.],
                'chassis': {'enabled': self.enabled, 'homing': ''}, 'reason': ''}

    def start(self, ids):
        self.starts += 1
        self.enabled = True

    def target(self, action, ids, *, valid_for_s=.25):
        assert ids == self.ids
        self.commands.append(list(action))
        self.arms = list(action[:14])

    def renew(self):
        self.heartbeats += 1

    def home(self):
        self.homes += 1
        self.height = 0.

    def stop(self, *, freeze_arms=True):
        self.stops.append(freeze_arms)

    def fault_stop(self, reason):
        self.faults.append(reason)


class FakeClient:
    def __init__(self, robot):
        self.session = motion.ReplaySession(robot)
        self.sequence = 0
        self.snapshot = None
        self.closed = False

    def call(self, operation, **values):
        if operation == 'target':
            values.setdefault('valid_for_s', .25)
        self.sequence += 1
        self.snapshot = self.session.process({'sequence': self.sequence, 'operation': operation, **values})
        return self.snapshot

    def close(self):
        self.session.disconnect()
        self.closed = True


class Clock:
    def __init__(self): self.now = 10.
    def __call__(self): return self.now
    def sleep(self, delay): self.now += delay


def prepare(client, plan):
    schema = plan.metadata.schema
    return client.call('prepare', protocol=motion.PROTOCOL, calibration_ids=plan.ids,
                       action_names=list(schema.action_names), action_units=list(schema.action_units),
                       state_names=list(schema.state_names), state_units=list(schema.state_units))


def test_plan_uses_recorded_ternary_actions_and_rejects_fractional_mobile_replay(dataset):
    normal = ReplayPlan(dataset)
    np.testing.assert_allclose(normal.times, [0., .04, .08, .12])
    with pytest.raises(ValueError, match='speed 1'):
        ReplayPlan(dataset, speed=.5)
    assert normal.timing == 'original_observation_source_time'
    assert normal.actions[0, 15] != normal.initial_state[15]  # action is not next state.
    with pytest.raises(ValueError): ReplayPlan(dataset, speed=2.)
    with pytest.raises(ValueError): ReplayPlan(dataset, frame=3, frames=2)
    canonical = motion.checked_action([0.] * 14 + [1., -1., 0., 1.])
    assert canonical[14:] == [1., -1., 0., 1.]
    with pytest.raises(ValueError, match='exactly'):
        motion.checked_action([0.] * 14 + [.5, 0., 0., 0.])


def test_plan_rejects_mixed_new_names_and_old_physical_units(dataset):
    assets_path = dataset / 'meta/robot.json'
    assets = json.loads(assets_path.read_text())
    assets['action_units'] = ['normalized'] * 14 + ['m/s', 'm/s', 'm/s', 'rad/s']
    assets_path.write_text(json.dumps(assets))
    path = dataset / 'data/chunk-000/file-000.parquet'
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row[ACTION][14:] = [.015, .15, 0., 0.]
    pq.write_table(pa.Table.from_pylist(rows), path)

    with pytest.raises(ValueError, match='旧动作格式'):
        ReplayPlan(dataset)


def test_inspection_never_connects_and_needs_no_torch(dataset, monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError('Inspection attempted a robot connection')
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    assert cli.main(['replay', '--dataset', str(dataset)]) == 0
    code = 'import xyc.learning.replay,sys; assert "torch" not in sys.modules and "rclpy" not in sys.modules'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_source_and_calibration_checks_precede_any_writes(dataset):
    plan = ReplayPlan(dataset)
    robot = FakeRobot(['wrong-left', 'wrong-right'])
    client = FakeClient(robot)
    with pytest.raises(ValueError, match='calibration'):
        prepare(client, plan)
    assert not robot.commands and robot.starts == 0
    with pytest.raises(ValueError, match='Prepare'):
        client.call('target', action=[0.] * 18)
    good = FakeClient(FakeRobot(plan.ids))
    prepare(good, plan)
    good.call('start')
    with pytest.raises(ValueError, match='18'):
        good.call('target', action=[0.] * 14)
    with pytest.raises(ValueError, match=r'exactly -1, 0 or \+1'):
        good.call('target', action=[0.] * 14 + [.5, 0., 0., 0.])


def test_recorded_start_beyond_calibration_is_rejected_before_any_enable_or_homing(dataset, tmp_path):
    plan = ReplayPlan(dataset)
    plan.initial_state[2] = 100.01
    robot, clock = FakeRobot(plan.ids, height=None), Clock()
    robot.arms[2] = 97.80018
    client = FakeClient(robot)
    with pytest.raises(ValueError, match='arm_left_elbow_flex.*97.800.*100.010'):
        execute(plan, output_dir=tmp_path / 'margin', home_lift=True,
                client_factory=lambda *_: client, clock=clock, sleep=clock.sleep, progress=lambda _: None)
    assert robot.starts == robot.homes == 0 and not robot.commands
    meta = json.loads((tmp_path / 'margin/meta.json').read_text(encoding='utf-8'))
    assert meta['frames_sent'] == 0 and meta['failure_snapshot']['state'][2] == robot.arms[2]


def test_replay_start_uses_full_calibrated_range_without_extra_margin(dataset):
    plan = ReplayPlan(dataset)
    positions = list(plan.initial_state[:14])
    positions[2] = 99.35838
    plan.initial_state[2] = float(np.float32(positions[2]))
    plan.check_start_pose(positions)
    plan.initial_state[2] = 99.
    plan.check_start_pose(positions)
    positions[2] = 98.5
    plan.check_start_pose(positions)
    positions[2] = 97.80018
    plan.initial_state[2] = 99.35838
    plan.check_start_pose(positions)
    for endpoint in (-100., 100.):
        plan.initial_state[2] = endpoint
        plan.check_start_pose(positions)
    for invalid in (-100.01, 100.01, float('nan')):
        plan.initial_state[2] = invalid
        with pytest.raises(ValueError, match='arm_left_elbow_flex'):
            plan.check_start_pose(positions)


def test_feedback_fault_and_calibration_change_have_distinct_errors(dataset):
    plan = ReplayPlan(dataset)
    robot = FakeRobot(plan.ids)
    client = FakeClient(robot)
    prepare(client, plan)
    client.call('start')
    original = robot.snapshot
    def faulted():
        return {**original(), 'ready': False, 'reason': 'ID 11 持续过流：5.23 A'}
    robot.snapshot = faulted
    with pytest.raises(ValueError, match='ID 11.*5.23 A'):
        client.call('heartbeat')
    robot.ids = ['changed-left', 'changed-right']
    with pytest.raises(ValueError, match='calibration changed'):
        client.call('heartbeat')


@pytest.mark.parametrize('confirmed', (True, False))
def test_rejected_tcp_reply_preserves_fault_snapshot_and_stop_detail(confirmed):
    protocol = motion
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    messages = []
    snapshot = {'ready': False, 'reason': 'ID 11 持续过流：5.23 A',
                'chassis': {'fault': 'ID 11 持续过流：5.23 A'}}
    stop_error = '' if confirmed else 'right arm_right_shoulder_pan: torque-off unconfirmed'
    def serve():
        connection, _ = listener.accept()
        with connection, connection.makefile('rwb') as channel:
            message = protocol.receive(channel)
            messages.append(message)
            protocol.send(channel, {'sequence': message['sequence'], 'protocol': protocol.PROTOCOL,
                                    'accepted': False, 'message': snapshot['reason'], 'snapshot': snapshot,
                                    'stop_confirmed': confirmed, 'stop_error': stop_error})
    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    client = MotionClient('127.0.0.1', listener.getsockname()[1], keepalive_period=None)
    try:
        with pytest.raises(RuntimeError, match='ID 11') as caught:
            client.call('heartbeat')
        assert caught.value.snapshot == snapshot and client.snapshot == snapshot
        assert client.last_stop_confirmed is confirmed
        assert caught.value.stop_error == stop_error
        if not confirmed:
            assert stop_error in str(caught.value)
        with pytest.raises(motion.MotionRejected) as repeated:
            client.call('fault')
        assert repeated.value is caught.value
    finally:
        client.close()
        worker.join(2.)
        listener.close()
    assert len(messages) == 1 and messages[0]['operation'] == 'heartbeat'


def test_execute_aligns_then_replays_once_and_stops_velocity(dataset, tmp_path):
    plan = ReplayPlan(dataset)
    robot, clock = FakeRobot(plan.ids), Clock()
    client = FakeClient(robot)
    result = execute(plan, output_dir=tmp_path / 'run', client_factory=lambda *_: client,
                     clock=clock, sleep=clock.sleep, progress=lambda _: None)
    records = [json.loads(line) for line in (tmp_path / 'run/steps.jsonl').read_text().splitlines()]
    actual = [r for r in records if r['kind'] == 'replay']
    assert len(actual) == 4 and robot.starts == 1 and client.closed
    np.testing.assert_allclose([r['action'] for r in actual], plan.actions)
    assert all(r['driver_accepted'] for r in actual)
    assert [r['frame'] for r in actual] == [0, 1, 2, 3]
    assert robot.stops == [False]  # Normal end keeps the last arm goal, with mobile zeroed.
    meta = json.loads((tmp_path / 'run/meta.json').read_text())
    assert meta['status'] == 'completed' and meta['execution'] == 'real_robot' and meta['task_success'] is None
    assert result['frames_sent'] == 4


def test_missing_lift_reference_needs_explicit_home_permission(dataset, tmp_path):
    plan = ReplayPlan(dataset)
    robot = FakeRobot(plan.ids, height=None)
    client, clock = FakeClient(robot), Clock()
    with pytest.raises(ValueError, match='home-lift'):
        execute(plan, output_dir=tmp_path / 'nohome', client_factory=lambda *_: client,
                clock=clock, sleep=clock.sleep, progress=lambda _: None)
    assert robot.starts == robot.homes == 0
    robot = FakeRobot(plan.ids, height=None)
    client, clock = FakeClient(robot), Clock()
    execute(plan, output_dir=tmp_path / 'home', home_lift=True, client_factory=lambda *_: client,
            clock=clock, sleep=clock.sleep, progress=lambda _: None)
    assert robot.homes == 1


def test_unknown_target_reply_is_not_retried_and_stops(dataset, tmp_path):
    plan = ReplayPlan(dataset)
    robot, clock = FakeRobot(plan.ids), Clock()
    client = FakeClient(robot)
    original = client.call

    def lost(operation, **values):
        result = original(operation, **values)
        if operation == 'target' and len(robot.commands) == 4:
            raise TimeoutError('reply lost after actual driver acceptance')
        return result

    client.call = lost
    with pytest.raises(TimeoutError, match='reply lost'):
        execute(plan, output_dir=tmp_path / 'lost', client_factory=lambda *_: client,
                clock=clock, sleep=clock.sleep, progress=lambda _: None)
    assert len(robot.commands) == 4 and len(robot.faults) == 1 and not robot.stops
    meta = json.loads((tmp_path / 'lost/meta.json').read_text())
    assert meta['status'] == 'failed' and meta['stop_confirmed'] is True


def test_partial_start_failure_still_stops(dataset):
    plan = ReplayPlan(dataset)
    robot = FakeRobot(plan.ids)
    client = FakeClient(robot)
    prepare(client, plan)
    def broken(ids):
        robot.enabled = True
        raise RuntimeError('lost start reply after enabling')
    robot.start = broken
    with pytest.raises(RuntimeError): client.call('start')
    client.close()
    assert len(robot.faults) == 1 and not robot.stops


def test_fault_stop_failure_is_retained_and_session_cannot_restart(dataset):
    plan = ReplayPlan(dataset)
    robot = FakeRobot(plan.ids)
    client = FakeClient(robot)
    prepare(client, plan)
    client.call('start')

    def stop_failure(reason):
        robot.faults.append(reason)
        raise RuntimeError('right stop reply lost')

    robot.fault_stop = stop_failure
    with pytest.raises(RuntimeError, match='stop reply lost'):
        client.call('fault')
    assert client.session.stop_error == 'right stop reply lost'
    with pytest.raises(ValueError, match='faulted'):
        client.call('start')
    client.close()
    assert len(robot.faults) == 1 and robot.starts == 1


def test_real_tcp_framing_and_repeated_sequence_rejected(dataset):
    plan = ReplayPlan(dataset)
    robot = FakeRobot(plan.ids)
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    errors = []
    def server():
        protocol = motion
        session = protocol.ReplaySession(robot)
        try:
            connection, _ = listener.accept()
            with connection, connection.makefile('rwb') as channel:
                while True:
                    try: message = protocol.receive(channel)
                    except EOFError: break
                    result = session.process(message)
                    protocol.send(channel, {'sequence': message['sequence'], 'protocol': protocol.PROTOCOL,
                                            'accepted': True, 'snapshot': result})
        except Exception as exc:
            errors.append(exc)
        finally:
            session.disconnect()
    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    client = MotionClient('127.0.0.1', listener.getsockname()[1])
    try:
        assert client.call('status')['ready']
        prepare(client, plan)
        client.call('start')
        client.call('target', action=plan.actions[0].tolist())
        client.call('finish')
    finally:
        client.close()
        thread.join(3.)
        listener.close()
    assert not errors and not thread.is_alive() and len(robot.commands) == 1
    session = motion.ReplaySession(robot)
    session.process({'sequence': 1, 'operation': 'status'})
    with pytest.raises(ValueError, match='sequence'):
        session.process({'sequence': 1, 'operation': 'status'})


@contextmanager
def deadline_server(robot, *, timeout=.35, target_delay=0.):
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)
    done = threading.Event()
    messages, errors = [], []
    def serve():
        protocol = motion
        session = protocol.ReplaySession(robot)
        try:
            connection, _ = listener.accept()
            connection.settimeout(timeout)
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with connection, connection.makefile('rwb') as channel:
                while True:
                    try:
                        message = protocol.receive(channel)
                    except (EOFError, socket.timeout):
                        break
                    messages.append(message)
                    if message['operation'] == 'target':
                        time.sleep(target_delay)
                    result = session.process(message)
                    protocol.send(channel, {'sequence': message['sequence'], 'protocol': protocol.PROTOCOL,
                                            'accepted': True, 'snapshot': result})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            errors.append(exc)
        finally:
            session.disconnect()
            done.set()
    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        yield listener.getsockname()[1], messages, errors, done
    finally:
        worker.join(3.)
        listener.close()
        assert not worker.is_alive()


def test_idle_timeout_reproduced_then_fixed_without_motion(dataset):
    plan = ReplayPlan(dataset)
    old_robot = FakeRobot(plan.ids)
    with deadline_server(old_robot) as (port, messages, errors, done):
        old = MotionClient('127.0.0.1', port, keepalive_period=None)
        try:
            old.call('status')
            assert done.wait(1.)
            with pytest.raises(ConnectionError, match='通信中断'):
                old.call('status')
        finally:
            old.close()
        assert not old_robot.commands and not errors
    robot = FakeRobot(plan.ids)
    with deadline_server(robot) as (port, messages, errors, done):
        client = MotionClient('127.0.0.1', port)
        try:
            client.call('status')
            time.sleep(.75)  # Longer than two backend idle deadlines; main thread deliberately pauses.
            assert client.call('status')['ready'] and not done.is_set()
        finally:
            client.close()
        assert len(messages) >= 5 and all(m['operation'] == 'status' for m in messages)
        assert not robot.commands and robot.starts == robot.heartbeats == 0 and not errors


def test_readonly_keepalive_does_not_renew_velocity_watchdog(dataset):
    plan = ReplayPlan(dataset)
    class VelocityLeaseRobot(FakeRobot):
        active_until = 0.
        def target(self, action, ids, *, valid_for_s=.25):
            super().target(action, ids, valid_for_s=valid_for_s)
            self.active_until = time.monotonic() + .12
        def renew(self):
            super().renew()
            self.active_until = time.monotonic() + .12
        def snapshot(self):
            result = super().snapshot()
            result['velocity_active'] = time.monotonic() < self.active_until
            return result
    robot = VelocityLeaseRobot(plan.ids)
    with deadline_server(robot) as (port, messages, errors, done):
        client = MotionClient('127.0.0.1', port)
        try:
            prepare(client, plan)
            client.call('start')
            client.call('target', action=plan.actions[0].tolist())
            time.sleep(.4)
            assert not client.call('status')['velocity_active']
            assert robot.heartbeats == 0 and len(robot.commands) == 1
            client.call('finish')
        finally:
            client.close()
        assert not errors


def test_keepalive_never_steals_or_retries_a_target_reply(dataset):
    plan = ReplayPlan(dataset)
    robot = FakeRobot(plan.ids)
    with deadline_server(robot, target_delay=.12) as (port, messages, errors, done):
        client = MotionClient('127.0.0.1', port)
        try:
            prepare(client, plan)
            client.call('start')
            assert client.call('target', action=plan.actions[0].tolist())['ready']
            time.sleep(.12)
            client.call('finish')
        finally:
            client.close()
        assert len(robot.commands) == 1 and not errors
        assert [m['sequence'] for m in messages] == list(range(1, len(messages) + 1))


def source_classes(path, globals_):
    """Exercise ROS adapter methods with fake messages on a PC without rclpy."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    classes = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)], type_ignores=[])
    scope = {'Node': object, **globals_}
    exec(compile(classes, str(path), 'exec'), scope)
    return scope


@pytest.mark.parametrize('owner', ['teleop', 'replay', 'policy'])
def test_existing_coordinator_uses_stream_writes_for_both_owners(owner):
    path = Path(__file__).parents[2] / 'ros_ws/src/xyc_hardware/xyc_hardware/arm_group_coordinator.py'
    scope = source_classes(path, {'threading': threading, 'time': time})
    node = scope['ArmGroupCoordinator'].__new__(scope['ArmGroupCoordinator'])
    node.source = owner
    node.mutex = threading.Lock()
    node.validate_command = node.check_command_time = lambda _: None
    node.token = lambda: 1
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000))
    node.publish_recording_command = lambda *_: None
    calls = []
    node.call_bus = lambda side, operation, token, **kwargs: calls.append((side, operation, kwargs['targets']))
    request = SimpleNamespace(position_normalized=list(range(14)), sequence=7,
                              header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)),
                              valid_for=SimpleNamespace(sec=0, nanosec=250_000_000))
    response = SimpleNamespace(accepted=False, message='')
    with ThreadPoolExecutor(max_workers=2) as workers:
        node.bus_workers = workers
        node.set_targets(request, response)
    assert response.accepted
    assert sorted(calls) == [('left', 'stream_targets', tuple(range(7))),
                             ('right', 'stream_targets', tuple(range(7, 14)))]


def test_robot_rejects_expired_model_observation_before_any_target_write(dataset):
    plan = ReplayPlan(dataset)
    robot = FakeRobot(plan.ids)
    snapshot = robot.snapshot
    robot.snapshot = lambda: {**snapshot(), 'source_time_ns': 2_000_000_000}
    client = FakeClient(robot)
    prepare(client, plan)
    client.call('start')
    with pytest.raises(ValueError, match='no target was written'):
        client.call('target', action=[0.] * 18, observation_time_ns=1_600_000_000)
    with pytest.raises(ValueError, match='no target was written'):
        client.call('target', action=[0.] * 18, observation_time_ns=1_750_000_000)
    assert robot.commands == []
    client.call('target', action=[0.] * 18, observation_time_ns=1_950_000_000)
    assert len(robot.commands) == 1


@pytest.mark.parametrize('observed, validity, expected', [
    (1_780_000_000, .2, .03), (1_950_000_000, .1, .1)])
def test_robot_deducts_observation_age_from_the_remaining_target_lifetime(dataset, observed, validity, expected):
    plan = ReplayPlan(dataset)
    robot = FakeRobot(plan.ids)
    snapshot = robot.snapshot
    robot.snapshot = lambda: {**snapshot(), 'source_time_ns': 2_000_000_000}
    calls = []
    robot.target = lambda *args, **kwargs: calls.append(kwargs['valid_for_s'])
    client = FakeClient(robot)
    prepare(client, plan)
    client.call('start')
    client.call('target', action=[0.] * 18, observation_time_ns=observed, valid_for_s=validity)
    assert calls == [pytest.approx(expected)]


def test_replay_uses_the_configured_lift_travel(dataset, monkeypatch):
    from xyc.learning import replay
    path = dataset / 'data/chunk-000/file-000.parquet'
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row[OBS_STATE][14] = .7
    pq.write_table(pa.Table.from_pylist(rows), path)
    monkeypatch.setattr(replay, 'robot_profile', lambda: SimpleNamespace(lift_max_height_m=.8))
    assert ReplayPlan(dataset).initial_state[14] == pytest.approx(.7)
    monkeypatch.setattr(replay, 'robot_profile', lambda: SimpleNamespace(lift_max_height_m=.5))
    with pytest.raises(ValueError, match=r'0..0.5 m'):
        ReplayPlan(dataset)


@pytest.mark.parametrize('recorded, measured', [(0., .0045), (.6, .5955)])
def test_replay_completes_alignment_at_a_recorded_lift_endpoint(dataset, tmp_path, monkeypatch, recorded, measured):
    from xyc.learning import replay
    monkeypatch.setattr(replay, 'robot_profile', lambda: SimpleNamespace(lift_max_height_m=.6))
    path = dataset / 'data/chunk-000/file-000.parquet'
    rows = pq.read_table(path).to_pylist()
    rows[0][OBS_STATE][14] = float(np.float32(recorded))
    pq.write_table(pa.Table.from_pylist(rows), path)
    plan = ReplayPlan(dataset)
    assert plan.initial_state[14] == recorded
    robot, clock = FakeRobot(plan.ids, height=measured), Clock()
    result = execute(plan, output_dir=tmp_path / 'endpoint', client_factory=lambda *_: FakeClient(robot),
                     clock=clock, sleep=clock.sleep, progress=lambda _: None)
    assert result['frames_sent'] == 4
    assert [row[14] for row in robot.commands[:3]] == [0., 0., 0.]


@pytest.mark.parametrize('missing', [False, True])
def test_replay_rejects_an_unknown_or_mismatched_recorded_speed(dataset, missing):
    path = dataset / 'meta/robot.json'
    assets = json.loads(path.read_text())
    if missing:
        assets.pop('mobile_action_scale')
    else:
        assets['mobile_action_scale']['base_x_m_s'] = .30
    path.write_text(json.dumps(assets))
    with pytest.raises(ValueError, match='speeds are missing or differ'):
        ReplayPlan(dataset)
