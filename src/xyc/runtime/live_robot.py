"""Live robot observation and command transport for model-independent runtimes."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import time
from urllib.request import Request, urlopen

from xyc.learning.contracts import ACTION_NAMES, ACTION_UNITS, STATE_NAMES, STATE_UNITS
from .policy_executor import CommandResult


def follower_calibration_ids(robot_assets):
    documents = robot_assets.get('calibrations', {})
    result = []
    for side in ('left', 'right'):
        document = documents.get('calibration/' + side)
        if not isinstance(document, dict) or document.get('arm') != side:
            raise ValueError('Checkpoint is missing the follower calibration: ' + side)
        encoded = json.dumps(document, sort_keys=True, separators=(',', ':'),
                             ensure_ascii=False, allow_nan=False).encode()
        result.append(hashlib.sha256(encoded).hexdigest())
    return result


class LiveCameraClient:
    """Fetch one new JPEG from every camera through the existing SSH preview port."""

    def __init__(self, base_url, cameras, *, max_skew_s=.05, timeout_s=1.5):
        self.base_url = base_url.rstrip('/')
        self.cameras = tuple(cameras)
        self.max_skew_ns = round(max_skew_s * 1e9)
        self.timeout_s = float(timeout_s)
        if not self.cameras or not 0 < max_skew_s <= .2 or not 0 < self.timeout_s <= 5.:
            raise ValueError('Invalid live camera configuration')

    def _one(self, name):
        request = Request(self.base_url + '/snapshot/' + name,
                          headers={'Cache-Control': 'no-cache', 'Connection': 'close'})
        with urlopen(request, timeout=self.timeout_s) as response:
            if response.status != 200 or response.headers.get_content_type() != 'image/jpeg':
                raise ConnectionError(f'{name}: camera snapshot is unavailable')
            stamp = int(response.headers['X-XYC-Source-Time-Ns'])
            data = response.read(2 * 1024 * 1024 + 1)
        if not 4 <= len(data) <= 2 * 1024 * 1024 or not data.startswith(b'\xff\xd8') or stamp <= 0:
            raise ValueError(f'{name}: invalid live JPEG or timestamp')
        return name, stamp, data

    def capture(self):
        for _ in range(4):
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=len(self.cameras)) as pool:
                rows = list(pool.map(self._one, self.cameras))
            stamps = [row[1] for row in rows]
            if max(stamps) - min(stamps) <= self.max_skew_ns:
                return ({name: data for name, _, data in rows}, stamps,
                        time.monotonic() - started)
        raise TimeoutError('Four live cameras did not produce a frame group within 50 ms')


class RobotControlSink:
    """Route normalized policy output through the same backend used by replay."""

    def __init__(self, host='localhost', port=17886):
        from xyc.learning.replay import ReplayClient, transport
        self.protocol = transport()
        self.client = ReplayClient(host, port, keepalive_period=.08)
        self.started = False
        self.closed = False

    def status(self):
        return self.client.call('status')

    def prepare(self, schema, calibration_ids):
        if (tuple(schema.action_names) != ACTION_NAMES or tuple(schema.action_units) != ACTION_UNITS
                or tuple(schema.state_names) != STATE_NAMES or tuple(schema.state_units) != STATE_UNITS):
            raise ValueError('Checkpoint does not use the canonical 18-D robot contract')
        return self.client.call(
            'prepare', protocol=self.protocol.PROTOCOL, calibration_ids=list(calibration_ids),
            action_names=list(schema.action_names), action_units=list(schema.action_units),
            state_names=list(schema.state_names), state_units=list(schema.state_units))

    def start(self):
        self.started = True  # A lost reply may conceal torque enable; stop on every later error.
        return self.client.call('start')

    def home_lift(self):
        return self.client.call('home')

    def heartbeat(self):
        return self.client.call('heartbeat')

    def dispatch(self, command, *, sequence, valid_for_s):
        del valid_for_s  # The server adds a fresh ROS deadline to every accepted request.
        mobile = command['mobile_command']
        values = [*command['arm_position_normalized'],
                  mobile['lift'], mobile['x'], mobile['y'], mobile['yaw']]
        sent = time.time_ns()
        snapshot = self.client.call('target', action=values)
        return CommandResult(
            accepted=True, command_id=str(sequence), source_time_ns=sent, source_clock='pc_wall',
            message='Unified robot controller accepted the model target',
            feedback_state=snapshot.get('state'),
            feedback_time_ns=snapshot.get('feedback_monotonic_ns'),
            feedback_clock='robot_monotonic')

    def stop(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.started:
                self.client.call('stop')
        finally:
            self.started = False
            self.client.close()


def validate_live_state(snapshot, expected_calibration_ids):
    state = snapshot.get('state')
    if (snapshot.get('ready') is not True or snapshot.get('calibration_ids') != list(expected_calibration_ids)
            or not isinstance(state, list) or len(state) != 18
            or any(value is None or not math.isfinite(float(value)) for value in state)):
        raise ValueError('Live robot feedback is incomplete, stale or uses different calibration assets')
    return [float(value) for value in state]
