"""Live robot observation and command transport for model-independent runtimes."""

from xyc.robot.profile import robot_profile

from concurrent.futures import ThreadPoolExecutor
import math
import time
from urllib.request import Request, urlopen

from xyc.contracts import ACTION_NAMES, ACTION_UNITS, STATE_NAMES, STATE_UNITS
from xyc.observation import CAMERAS, ObservationBuffer
from xyc.transport.motion import MotionClient, PROTOCOL
from .policy_executor import CommandResult




class LiveCameraClient:
    """Fetch one new JPEG from every camera through the existing SSH preview port."""

    def __init__(self, base_url, cameras, *, timeout_s=1.5):
        self.base_url = base_url.rstrip('/')
        self.cameras = tuple(cameras)
        self.timeout_s = float(timeout_s)
        if not self.cameras or not 0 < self.timeout_s <= 5.:
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
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=len(self.cameras)) as pool:
            rows = list(pool.map(self._one, self.cameras))
        return ({name: data for name, _, data in rows}, [stamp for _, stamp, _ in rows],
                time.monotonic() - started)


class RobotControlSink:
    """Route normalized policy output through the same backend used by replay."""

    def __init__(self, host='localhost', port=robot_profile().ports['motion']):
        self.client = MotionClient(host, port)
        self.started = False
        self.closed = False

    def status(self):
        return self.client.call('status')

    def prepare(self, schema, calibration_ids):
        if (tuple(schema.action_names) != ACTION_NAMES or tuple(schema.action_units) != ACTION_UNITS
                or tuple(schema.state_names) != STATE_NAMES or tuple(schema.state_units) != STATE_UNITS):
            raise ValueError('Checkpoint does not use the canonical 18-D robot contract')
        return self.client.call(
            'prepare', protocol=PROTOCOL, calibration_ids=list(calibration_ids),
            action_names=list(schema.action_names), action_units=list(schema.action_units),
            state_names=list(schema.state_names), state_units=list(schema.state_units))

    def start(self):
        self.started = True  # A lost reply may conceal torque enable; stop on every later error.
        return self.client.call('start')

    def home_lift(self):
        return self.client.call('home')

    def heartbeat(self):
        return self.client.call('heartbeat')

    def halt(self):
        """Stop motion while retaining the connection for another evaluation."""
        result = self.client.call('stop')
        self.started = False
        return result

    def dispatch(self, command, *, sequence, valid_for_s):
        mobile = command['mobile_command']
        values = [*command['arm_position_normalized'],
                  mobile['lift'], mobile['x'], mobile['y'], mobile['yaw']]
        sent = time.time_ns()
        payload = {'action': values, 'valid_for_s': valid_for_s}
        if 'observation_time_ns' in command:
            payload['observation_time_ns'] = command['observation_time_ns']
        snapshot = self.client.call('target', **payload)
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


class LiveObservationClient:
    """Pair stamped JPEGs and measured states with the recording aligner."""

    def __init__(self, cameras, sink, calibration_ids):
        self.cameras, self.sink = cameras, sink
        self.calibration_ids = list(calibration_ids)
        self.buffer = ObservationBuffer()

    def capture(self):
        started = time.monotonic()
        for _ in range(4):
            images, stamps, _ = self.cameras.capture()
            snapshot = self.sink.status()
            validate_live_state(snapshot, self.calibration_ids)
            now_ns = snapshot.get('source_time_ns')
            if type(now_ns) is not int or now_ns <= 0:
                raise ValueError('Robot backend must provide its ROS observation clock')
            for name, stamp in zip(self.cameras.cameras, stamps):
                self.buffer.add_camera(name, stamp, images[name])
            for side, sample in snapshot.get('observation_samples', {}).items():
                self.buffer.add_arm(side, sample)
            aligned = self.buffer.pick(now_ns)
            if aligned is not None:
                if list(aligned.calibration_ids) != self.calibration_ids:
                    raise ValueError('Aligned observation calibration differs from the checkpoint')
                return {
                    'state': list(aligned.state),
                    'images': {name: frame.jpeg for name, frame in zip(CAMERAS, aligned.images)},
                    'source_time_ns': aligned.stamp_ns, 'source_clock': 'robot_ros',
                    'camera_stamps_ns': [frame.stamp_ns for frame in aligned.images],
                    'state_stamps_ns': list(aligned.state_stamps_ns),
                    'calibration_ids': list(aligned.calibration_ids), 'sessions': list(aligned.sessions),
                    'camera_skew_s': aligned.camera_skew_s,
                    'observation_fetch_s': time.monotonic() - started,
                }
        raise TimeoutError('Live observation alignment failed: ' + self.buffer.reason)
