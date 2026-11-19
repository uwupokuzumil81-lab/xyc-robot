# SPDX-License-Identifier: Apache-2.0
"""Execute a PC dataset stream through existing ROS arm and mobile drivers."""

import json
import math
import select
import socket
import threading
import time

import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
from std_srvs.srv import Trigger
from xyc_interfaces.msg import ArmState, MobileCommand
from xyc_interfaces.srv import SetArmTargets

from xyc_hardware.chassis_control import CHASSIS_NOT_READY, COMMAND_QOS, COMMAND_RESULT_QOS, mobile_message
from xyc_hardware.motion_stop import MotionFaultStop
from xyc.contracts import ARM_NAMES, COMMAND_VALIDITY_S, checked_mobile_action
from xyc.transport.motion import (
    PORT, PROTOCOL, ReplaySession, receive, send,
)


class MotionReplay(Node):
    def __init__(self, **kwargs):
        super().__init__('motion_replay', **kwargs)
        self.port = self.declare_parameter('port', PORT).value
        self.command_source = self.declare_parameter('command_source', 'replay').value
        if self.command_source not in ('replay', 'policy'):
            raise ValueError('command_source must be replay or policy')
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.arms, self.mobile = {}, None
        self.arm_control = {}
        self.mobile_results = {}
        self.last_mobile = None
        self.command_sequence = 0
        self.armed = False
        self.motion_fault = ''
        self.stop_error = None
        self.fault_request = MotionFaultStop(self)
        self.closing = threading.Event()
        self.connection = None
        self.session = self.peer_events = None
        self.activity_timer = self.create_timer(.01, self.watch_session)
        self.targets = self.create_client(SetArmTargets, '/robot/arms/set_targets')
        self.start_arms = self.create_client(Trigger, '/robot/arms/start')
        self.owner = self.create_client(GetParameters, '/arm_group_coordinator/get_parameters')
        self.mobile_services = {name: self.create_client(Trigger, '/robot/chassis/' + name)
                                for name in ('enable', 'stop', 'home_lift')}
        self.mobile_pub = self.create_publisher(MobileCommand, '/robot/mobile/command', COMMAND_QOS)
        self.state_subscriptions = [self.create_subscription(
            ArmState, f'/motors/{side}/state', lambda msg, side=side: self._arm(side, msg),
            qos_profile_sensor_data) for side in ('left', 'right')]
        self.state_subscriptions += [
            self.create_subscription(String, '/robot/chassis/status', self._mobile, 1),
            self.create_subscription(
                String, '/robot/chassis/recording_command', self._mobile_result, COMMAND_RESULT_QOS)]
        self.state_subscriptions += [self.create_subscription(
            String, f'/motors/{side}/recording_sample', lambda msg, side=side: self._arm_control(side, msg),
            qos_profile_sensor_data) for side in ('left', 'right')]
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.listener.bind(('127.0.0.1', self.port))
            self.port = self.listener.getsockname()[1]
            self.listener.listen(1)
            self.listener.settimeout(.2)
        except BaseException:
            self.listener.close()
            raise
        self.worker = threading.Thread(target=self._serve, daemon=True)
        self.worker.start()
        self.get_logger().info(
            f'Unified {self.command_source} control ready on loopback {self.port}; startup sends no commands')

    def _arm(self, side, message):
        with self.lock:
            self.arms[side] = time.monotonic(), message

    def _mobile(self, message):
        try:
            payload = json.loads(message.data)
        except ValueError:
            return
        with self.lock:
            self.mobile = time.monotonic(), payload

    def _arm_control(self, side, message):
        try:
            payload = json.loads(message.data)
        except ValueError:
            return
        with self.lock:
            self.arm_control[side] = time.monotonic(), payload

    def _mobile_result(self, message):
        try:
            event = json.loads(message.data)
            identity = event['sequence']
        except (ValueError, KeyError):
            return
        with self.condition:
            self.mobile_results[identity] = event
            while len(self.mobile_results) > 64:
                self.mobile_results.pop(next(iter(self.mobile_results)))
            self.condition.notify_all()

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            positions, ids, errors = [], [], []
            if self.motion_fault:
                errors.append(self.motion_fault)
            if self.stop_error:
                errors.append(self.stop_error)
            driver_targets, driver_target_ticks, driver_target_stamps = [], [], {}
            for index, side in enumerate(('left', 'right')):
                entry = self.arms.get(side)
                valid = (entry is not None and now - entry[0] <= .3 and entry[1].complete
                         and list(entry[1].names) == list(ARM_NAMES[index * 7:(index + 1) * 7])
                         and len(entry[1].position_normalized) == 7 and all(entry[1].valid)
                         and all(math.isfinite(v) for v in entry[1].position_normalized))
                if not valid:
                    errors.append(side + ' arm feedback unavailable')
                positions.extend(list(entry[1].position_normalized) if valid else [None] * 7)
                ids.append(entry[1].calibration_id if entry else '')
                control = self.arm_control.get(side)
                payload = control[1] if control else {}
                fresh_targets = control is not None and now - control[0] <= .3
                for output, key in ((driver_targets, 'driver_targets'),
                                    (driver_target_ticks, 'driver_target_ticks')):
                    values = payload.get(key)
                    output.extend(list(values) if fresh_targets and isinstance(values, list)
                                  and len(values) == 7 else [None] * 7)
                driver_target_stamps[side] = payload.get('stamp_ns') if fresh_targets else None
                if self.armed and (control is None or now - control[0] > .3 or not control[1].get('valid')
                                   or control[1].get('calibration_id') != ids[-1]):
                    errors.append(side + ' arm ownership/fault: ' + str(control[1].get('reason') if control else 'waiting'))
            arms_ready = not errors
            mobile = self.mobile[1] if self.mobile else {}
            if (not self.mobile or now - self.mobile[0] > .4 or mobile.get('fault')
                    or mobile.get('feedback_paused') or len(mobile.get('base_measured') or []) != 3):
                errors.append('base/lift feedback unavailable: ' + str(
                    mobile.get('fault') or mobile.get('reason', 'waiting')))
            height = mobile.get('height_m')
            velocity = mobile.get('base_measured') or [None] * 3
            return {'ready': not errors, 'arms_ready': arms_ready, 'reason': '; '.join(errors), 'calibration_ids': ids,
                    'stop_confirmed': self.stop_error == '' if self.stop_error is not None else None,
                    'stop_error': self.stop_error,
                    'state': [*positions, height, *velocity], 'chassis': mobile,
                    'driver_targets': driver_targets, 'driver_target_ticks': driver_target_ticks,
                    'driver_target_stamps_ns': driver_target_stamps,
                    'observation_samples': {side: self.arm_control[side][1]
                                            for side in ('left', 'right') if side in self.arm_control},
                    'source_time_ns': self.get_clock().now().nanoseconds,
                    'sample_received_monotonic_s': {
                        side: self.arms[side][0] if side in self.arms else None for side in ('left', 'right')},
                    'chassis_received_monotonic_s': self.mobile[0] if self.mobile else None,
                    'feedback_monotonic_ns': time.monotonic_ns()}

    def _call(self, client, request, timeout=3., *, deadline=None):
        local_deadline = time.monotonic() + timeout
        deadline = min(local_deadline, deadline) if deadline is not None else local_deadline
        if deadline <= time.monotonic():
            raise TimeoutError('ROS command deadline expired before dispatch')
        available = client.wait_for_service(timeout_sec=min(.5, deadline - time.monotonic()))
        if time.monotonic() >= deadline:
            raise TimeoutError('ROS command deadline expired before dispatch')
        if not available:
            raise RuntimeError('Required replay ROS service is unavailable')
        with self.lock:
            self.require_active_link()
            if time.monotonic() >= deadline:
                raise TimeoutError('ROS command deadline expired before dispatch')
            future = client.call_async(request)
        try:
            while True:
                self.require_active_link()
                if self.closing.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError('ROS reply timed out; command outcome unknown, not retried')
                if future.done():
                    break
                time.sleep(.001)
        except (OSError, RuntimeError):
            client.remove_pending_request(future)
            raise
        result = future.result()
        if result is None:
            raise RuntimeError('Empty ROS reply')
        return result

    def _trigger(self, client):
        result = self._call(client, Trigger.Request())
        if not result.success:
            raise RuntimeError(result.message)

    def start(self, ids):
        parameters = GetParameters.Request()
        parameters.names = ['command_source']
        result = self._call(self.owner, parameters)
        if len(result.values) != 1 or result.values[0].string_value != self.command_source:
            raise ValueError(
                f'Start the {self.command_source} backend; the arm controller has a different owner')
        current = self.snapshot()
        if not current['ready'] or ids != current['calibration_ids']:
            raise ValueError('Robot feedback/calibration changed before starting')
        if self.armed:
            self._set_arms(current['state'][:14], ids)
        else:
            self._trigger(self.start_arms)
        self.armed = True
        self._enable_mobile(ids)
        self.last_mobile = None
        deadline = time.monotonic() + .8
        while not self.snapshot()['ready']:
            if time.monotonic() >= deadline:
                raise RuntimeError('Arm ownership/feedback was not confirmed after startup')
            time.sleep(.005)

    def _enable_mobile(self, ids):
        # Enabling arm hold may briefly interrupt polling on the shared left
        # bus. Rebuild the stationary window AFTER hold, rather than trusting
        # the older status published before that transaction.
        deadline = time.monotonic() + 3.
        reason = 'Waiting for fresh stationary feedback after arm hold'
        while time.monotonic() < deadline:
            current = self.snapshot()
            if ids != current['calibration_ids'] or current['chassis'].get('fault'):
                raise ValueError(current.get('reason') or 'Calibration changed or chassis faulted')
            if (not current['ready'] or current['chassis'].get('enable_ready') is False):
                reason = current.get('reason') or current['chassis'].get('enable_wait_reason') or reason
            else:
                try:
                    self._trigger(self.mobile_services['enable'])
                    return
                except RuntimeError as exc:
                    # This tagged rejection is issued before any motor writes.
                    # Register/write failures have unknown outcomes and are not retried.
                    if not str(exc).startswith(CHASSIS_NOT_READY):
                        raise
                    reason = str(exc)
            time.sleep(.01)
        raise TimeoutError('底盘/升降启动等待超时：' + reason)

    def home(self):
        self._trigger(self.mobile_services['home_lift'])
        self.last_mobile = None
        self.renew()

    def _set_arms(self, positions, ids, *, valid_for_s=COMMAND_VALIDITY_S, stamp=None, deadline=None):
        self.command_sequence += 1
        request = SetArmTargets.Request()
        request.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        request.source_id = self.command_source
        request.sequence = self.command_sequence
        request.valid_for.nanosec = int(valid_for_s * 1e9)
        request.arm_calibration_ids = list(ids)
        request.position_normalized = list(positions)
        reply = self._call(self.targets, request, timeout=.6, deadline=deadline)
        if not reply.accepted:
            raise RuntimeError('Arm replay target rejected: ' + reply.message)

    def _publish_mobile(self, values, *, stamp=None, expires_ns=None):
        message = mobile_message(self, values, stamp=stamp, expires_ns=expires_ns)
        with self.lock:
            self.require_active_link()
            if self.get_clock().now().nanoseconds >= message.expires_at.sec * 10**9 + message.expires_at.nanosec:
                raise TimeoutError('Mobile command expired before ROS publication')
            self.mobile_pub.publish(message)
        return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec

    def target(self, action, ids, *, valid_for_s):
        deadline = time.monotonic() + valid_for_s
        generated = self.get_clock().now().to_msg()
        lifetime = max(0., deadline - time.monotonic())
        mobile = checked_mobile_action(action[14:])
        status = self.snapshot()
        if (not status['ready'] or not status['chassis'].get('enabled')
                or status['chassis'].get('homing') or status['state'][14] is None):
            raise ValueError('Replay needs enabled base/lift, completed homing and fresh feedback')
        if time.monotonic() >= deadline:
            raise TimeoutError('Target deadline expired before arm dispatch')
        self._set_arms(action[:14], ids, valid_for_s=lifetime, stamp=generated, deadline=deadline)
        if time.monotonic() >= deadline:
            raise TimeoutError('Target deadline expired before base/lift dispatch')
        if mobile == self.last_mobile and not any(mobile):
            current = self.snapshot()
            if not current['ready']:
                raise RuntimeError('Base/lift feedback unavailable: ' + current['reason'])
            if current['chassis'].get('zero_targets_verified') is True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Target deadline expired before base/lift confirmation')
                return
        expires_ns = generated.sec * 10**9 + generated.nanosec + int(lifetime * 1e9)
        self.last_mobile = None  # An in-flight command invalidates the previous result.
        stamp = self._publish_mobile(mobile, stamp=generated, expires_ns=expires_ns)
        deadline = min(deadline, time.monotonic() + .2)
        with self.condition:
            while stamp not in self.mobile_results:
                self.require_active_link()
                if self.closing.is_set() or time.monotonic() >= deadline:
                    raise TimeoutError('No base/lift command result before the deadline')
                self.condition.wait(timeout=max(0., deadline - time.monotonic()))
            event = self.mobile_results.pop(stamp)
            if event.get('accepted') is not True:
                raise RuntimeError('Base/lift command rejected: ' + event.get('error', 'missing confirmation status'))
            if time.monotonic() >= deadline:
                raise TimeoutError('Target deadline expired before base/lift confirmation')
            if event['targets'] != mobile:
                raise RuntimeError('Base/lift accepted a different target')
        self.last_mobile = list(mobile)

    def renew(self):
        self._publish_mobile(self.last_mobile if self.last_mobile is not None else [0.] * 4)

    def stop(self, *, freeze_arms=True):
        self.last_mobile = None
        self._trigger(self.mobile_services['stop'])
        self.last_mobile = [0.] * 4
        if freeze_arms and self.armed:
            state = self.snapshot()
            if not state['arms_ready']:
                raise RuntimeError('Arm hold could not be confirmed: ' + state['reason'])
            self._set_arms(state['state'][:14], state['calibration_ids'])

    def _begin_fault(self, reason):
        if not self.motion_fault:
            self.motion_fault = reason or 'Motion control fault'
            self.last_mobile = None
            self.fault_request.begin(self.motion_fault)
            self.condition.notify_all()

    def _link_failure(self):
        if self.peer_events is not None and self.peer_events.poll(0):
            return 'Motion control connection closed during an active request'
        if (self.session is not None and self.session.started and self.session.activity_deadline is not None
                and time.monotonic() >= self.session.activity_deadline):
            return 'Motion control loop timed out waiting for a target or control heartbeat'
        return ''

    def watch_session(self):
        with self.lock:
            if self.session is not None and self.session.started and not self.motion_fault:
                if reason := self._link_failure():
                    self._begin_fault(reason)

    def require_active_link(self):
        with self.lock:
            if reason := self.motion_fault or self._link_failure():
                if self.session is not None and self.session.started:
                    self._begin_fault(reason)
                raise ConnectionError(reason)

    def fault_stop(self, reason):
        with self.lock:
            self._begin_fault(reason)
        errors = self.fault_request.wait()
        error = 'Protective stop not confirmed: ' + '; '.join(errors) if errors else ''
        with self.lock:
            self.stop_error = error
        if error:
            raise RuntimeError(error)

    def _serve(self):
        while not self.closing.is_set():
            try:
                connection, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            session = ReplaySession(self)
            events = select.poll()
            # Linux reports peer shutdown independently of the request reader,
            # including while a synchronous ROS service is in progress.
            events.register(connection, select.POLLRDHUP | select.POLLHUP | select.POLLERR)
            with self.lock:
                self.connection, self.session, self.peer_events = connection, session, events
            try:
                connection.settimeout(.35)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                with connection, connection.makefile('rwb') as channel:
                    while not self.closing.is_set():
                        message = receive(channel)
                        try:
                            if session.started:
                                self.require_active_link()
                            result = session.process(message)
                        except Exception as exc:
                            try:
                                session.disconnect(str(exc))
                            except Exception:
                                pass  # The session retains the unconfirmed stop error.
                            stop_error = self.stop_error if self.motion_fault else session.stop_error
                            if stop_error:
                                self.get_logger().error(stop_error)
                            send(channel, {'sequence': message['sequence'], 'protocol': PROTOCOL,
                                           'accepted': False, 'message': str(exc), 'snapshot': self.snapshot(),
                                           'stop_confirmed': stop_error == '', 'stop_error': stop_error or ''})
                            raise
                        send(channel, {'sequence': message['sequence'], 'protocol': PROTOCOL,
                                       'accepted': True, 'snapshot': result})
            except Exception as exc:
                self.get_logger().warning('Replay connection ended: ' + str(exc))
            finally:
                try:
                    session.disconnect()
                except Exception as exc:
                    self.get_logger().error(str(exc))
                with self.lock:
                    self.connection = self.session = self.peer_events = None
                    connection.close()

    def close(self):
        self.closing.set()
        self.listener.close()
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        self.worker.join(timeout=5.)


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = SingleThreadedExecutor()
    try:
        node = MotionReplay()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        executor.shutdown()
        rclpy.try_shutdown()
