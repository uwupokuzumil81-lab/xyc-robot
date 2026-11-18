"""Exclusive-owner motion control transport; no ROS imports or direct motor access."""

from xyc.robot.profile import robot_profile

import json
import math
import socket
import threading
import time

from xyc.contracts import (
    ACTION_NAMES, ACTION_UNITS, COMMAND_VALIDITY_S, STATE_NAMES, STATE_UNITS, checked_action,
)

PROTOCOL = 'xyc.motion_replay.v3'
PORT = robot_profile().ports['motion']
MAX_FRAME = 65536


def send(channel, message):
    encoded = json.dumps(message, ensure_ascii=True, allow_nan=False, separators=(',', ':')).encode() + b'\n'
    if len(encoded) > MAX_FRAME:
        raise ValueError('Replay message is too large')
    channel.write(encoded)
    channel.flush()


def receive(channel):
    raw = channel.readline(MAX_FRAME + 1)
    if not raw:
        raise EOFError('Replay connection closed')
    if len(raw) > MAX_FRAME or not raw.endswith(b'\n'):
        raise ValueError('Replay message is incomplete or too large')
    result = json.loads(raw)
    if not isinstance(result, dict) or type(result.get('sequence')) is not int:
        raise ValueError('Replay message needs an integer sequence')
    return result


class MotionRejected(RuntimeError):
    def __init__(self, message, *, operation, snapshot=None):
        super().__init__(message)
        self.operation, self.snapshot = operation, snapshot


class MotionClient:
    """Serialized TCP requests shared by dataset replay and live policy control."""

    def __init__(self, host, port, *, keepalive_period=.08):
        self.connection = socket.create_connection((host, port), timeout=3.)
        self.connection.settimeout(4.)
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.channel = self.connection.makefile('rwb')
        self.sequence = 0
        self.snapshot = None
        self.last_stop_confirmed = False
        self._io_lock = threading.RLock()
        self._stop_keepalive = threading.Event()
        self._closed = False
        self._transport_error = None
        self._keepalive_worker = None
        if keepalive_period is not None:
            if not 0 < keepalive_period <= .15:
                self.close()
                raise ValueError('Motion keepalive period must be in (0, 0.15] seconds')
            self._keepalive_worker = threading.Thread(
                target=self._keepalive, args=(keepalive_period,), name='xyc-motion-read-only-keepalive', daemon=True)
            self._keepalive_worker.start()

    def call(self, operation, **payload):
        started = time.monotonic()
        if operation == 'target':
            payload.setdefault('valid_for_s', COMMAND_VALIDITY_S)
        # One request owns send+receive together; keepalive cannot steal an action's reply.
        with self._io_lock:
            if self._closed or self._stop_keepalive.is_set() or self._transport_error:
                raise ConnectionError(self._transport_error or '运动控制连接已关闭；未自动重连或重发动作。')
            if operation == 'target':
                payload['valid_for_s'] -= time.monotonic() - started
                if payload['valid_for_s'] <= 0:
                    raise TimeoutError('Target expired while waiting for the control link; no target was sent')
            self.sequence += 1
            try:
                send(self.channel, {'sequence': self.sequence, 'operation': operation, **payload})
                result = receive(self.channel)
                if result.get('sequence') != self.sequence or result.get('protocol') != PROTOCOL:
                    raise ValueError('Wrong motion backend or mismatched reply sequence')
            except (EOFError, OSError, ValueError) as exc:
                self._transport_error = f'运动控制通信中断（{operation}）：{exc}；动作结果未确认，不自动重发。'
                self._stop_keepalive.set()
                raise ConnectionError(self._transport_error) from exc
            if result.get('accepted') is not True:
                self.snapshot = result.get('snapshot', self.snapshot)
                self.last_stop_confirmed = result.get('stop_confirmed') is True
                raise MotionRejected(result.get('message', 'Motion command rejected'),
                                     operation=operation, snapshot=self.snapshot)
            self.snapshot = result['snapshot']
            return self.snapshot

    def _keepalive(self, period):
        while not self._stop_keepalive.wait(period):
            try:
                # Read-only status keeps TCP alive but NEVER renews a moving base/lift goal.
                # If the control loop stalls, the driver's original motion timeout still stops it.
                self.call('status')
            except (OSError, RuntimeError, ValueError):
                self._stop_keepalive.set()
                return

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop_keepalive.set()
        if self._keepalive_worker is not None:
            self._keepalive_worker.join(timeout=.25)
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        if self._keepalive_worker is not None:
            self._keepalive_worker.join(timeout=1.)
        try:
            with self._io_lock:
                self.channel.close()
        finally:
            self.connection.close()


class ReplaySession:
    """Exactly one client owns one replay; unknown outcomes are never retried.

    The injected robot implements snapshot, start, home, target, renew and stop.
    status is read-only. All writes require a successful prepare handshake.
    """
    def __init__(self, robot):
        self.robot = robot
        self.sequence = 0
        self.prepared = False
        self.started = False
        self.calibration_ids = None

    def process(self, message):
        sequence = message.get('sequence')
        if type(sequence) is not int or not self.sequence < sequence < 2**63:
            raise ValueError('Replay sequence repeated or went backwards')
        self.sequence = sequence
        operation = message.get('operation')
        if operation == 'status':
            return self.robot.snapshot()
        if operation == 'prepare':
            if self.prepared or message.get('protocol') != PROTOCOL:
                raise ValueError('Replay protocol mismatch or repeated preparation')
            if (tuple(message.get('action_names', [])) != ACTION_NAMES
                    or tuple(message.get('action_units', [])) != ACTION_UNITS
                    or tuple(message.get('state_names', [])) != STATE_NAMES
                    or tuple(message.get('state_units', [])) != STATE_UNITS):
                raise ValueError('Dataset state/action names, order or units differ from the robot')
            state = self.robot.snapshot()
            ids = message.get('calibration_ids')
            if not state.get('ready') or ids != state.get('calibration_ids'):
                raise ValueError('Robot feedback unavailable or follower calibration differs from the dataset')
            self.prepared, self.calibration_ids = True, ids
            return state
        if not self.prepared:
            raise ValueError('Prepare the replay before issuing commands')
        if operation in ('stop', 'finish'):
            self.robot.stop(freeze_arms=operation != 'finish')
            self.started = False
            return self.robot.snapshot()
        if operation == 'start':
            if self.started:
                raise ValueError('Replay already started')
            self.started = True
            self.robot.start(self.calibration_ids)
            return self.robot.snapshot()
        if not self.started:
            raise ValueError('Replay is not started')
        state = self.robot.snapshot()
        if state.get('calibration_ids') != self.calibration_ids:
            raise ValueError('Follower calibration changed during replay')
        if not state.get('ready'):
            raise ValueError('Replay feedback not ready: ' + str(state.get('reason') or 'missing feedback'))
        if operation == 'home':
            self.robot.home()
        elif operation == 'heartbeat':
            self.robot.renew()
        elif operation == 'target':
            validity = message.get('valid_for_s')
            if (type(validity) not in (int, float) or not math.isfinite(validity)
                    or not 0 < validity <= COMMAND_VALIDITY_S):
                raise ValueError('Target validity must be positive and at most 0.25 seconds')
            observed = message.get('observation_time_ns')
            if observed is not None:
                now = state.get('source_time_ns')
                if (type(observed) is not int or type(now) is not int
                        or not 0 <= now - observed < int(COMMAND_VALIDITY_S * 1e9)):
                    raise ValueError('Model observation expired before execution; no target was written')
                validity = min(validity, COMMAND_VALIDITY_S - (now - observed) / 1e9)
            self.robot.target(checked_action(message.get('action')), self.calibration_ids,
                              valid_for_s=validity)
        else:
            raise ValueError('Unknown replay operation')
        return self.robot.snapshot()

    def disconnect(self):
        try:
            if self.started:
                self.robot.stop()
        finally:
            self.started = False
