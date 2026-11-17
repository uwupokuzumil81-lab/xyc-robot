# SPDX-License-Identifier: Apache-2.0
"""Coordinate two separately owned arm buses; never claim atomic writes."""

from concurrent.futures import ThreadPoolExecutor
import json
import math
import threading
import time
import sys

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.task import Future
from std_srvs.srv import Trigger
from std_msgs.msg import String
from xyc_interfaces.msg import ArmState
from xyc_interfaces.srv import ArmBusTransaction, SetArmTargets

from xyc_hardware.arm_calibration import arm_names
from xyc_hardware.teleop_transport import TARGET_COMMAND_EXPIRED, TELEOP_TARGET_DROPPED


class ArmTargetExpired(ValueError):
    pass


class BusResponseTimeout(RuntimeError):
    pass


class ArmGroupCoordinator(Node):
    def __init__(self, **kwargs):
        super().__init__('arm_group_coordinator', **kwargs)
        self.declare_parameter('command_source', 'keyboard')
        self.declare_parameter('auto_hold', False)
        self.streaming_targets = self.declare_parameter('streaming_targets', False).value
        self.source = self.get_parameter('command_source').value
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError('Set a nonempty command_source')
        if self.streaming_targets and self.source not in ('teleop', 'replay', 'policy'):
            raise ValueError('Streaming targets require a teleop or replay command owner')
        self.callbacks = ReentrantCallbackGroup()
        self.mutex = threading.Lock()
        self.bus_workers = ThreadPoolExecutor(max_workers=2)
        self.transaction_workers = ThreadPoolExecutor(max_workers=2)
        self.state_lock = threading.Lock()
        self.states = {}
        self.recording_publisher = self.create_publisher(
            String, '/robot/arms/recording_command', qos_profile_sensor_data)
        self.bus_clients = {
            side: self.create_client(ArmBusTransaction, f'/motors/{side}/group/transaction',
                                     callback_group=self.callbacks)
            for side in ('left', 'right')
        }
        self.state_subscriptions = [
            self.create_subscription(ArmState, f'/motors/{side}/state',
                                     lambda msg, side=side: self.on_state(side, msg),
                                     qos_profile_sensor_data, callback_group=self.callbacks)
            for side in ('left', 'right')
        ]
        self.start_srv = self.create_service(Trigger, '/robot/arms/start', self.handle_start,
                                             callback_group=self.callbacks)
        self.target_srv = self.create_service(SetArmTargets, '/robot/arms/set_targets',
                                              self.handle_targets, callback_group=self.callbacks)
        self.phase = 'idle'
        self.sessions = {}
        self.next_token = 1
        self.last_sequence = -1
        self.get_logger().info('Arm coordinator ready; automatic hold is opt-in')
        self.auto_timer = None
        if self.get_parameter('auto_hold').value:
            self.auto_timer = self.create_timer(.5, self.auto_start, callback_group=self.callbacks)

    async def auto_start(self):
        if not all(client.service_is_ready() for client in self.bus_clients.values()):
            return
        self.auto_timer.cancel()  # one attempt; never retry automatically
        result = await self.handle_start(Trigger.Request(), Trigger.Response())
        if result.success:
            self.get_logger().info(result.message)
        else:
            self.get_logger().error('Automatic arm hold failed: ' + result.message)

    async def run_transaction(self, operation, request, response):
        # Keep RPC waits out of the ROS executor so feedback and bus replies
        # continue to be taken while a transaction is in progress.
        completion = Future(executor=self.executor)

        def finish(task):
            try:
                result = task.result()
            except Exception as exc:
                completion.set_exception(exc)
            else:
                completion.set_result(result)

        self.transaction_workers.submit(operation, request, response).add_done_callback(finish)
        return await completion

    async def handle_start(self, request, response):
        return await self.run_transaction(self.start, request, response)

    async def handle_targets(self, request, response):
        return await self.run_transaction(self.set_targets, request, response)

    def on_state(self, side, msg):
        with self.state_lock:
            self.states[side] = (time.monotonic(), msg)

    def call_bus(self, side, operation, token=0, targets=None, session=None, expires_ns=None):
        client = self.bus_clients[side]
        if not client.service_is_ready():
            raise RuntimeError(f'{side}: bus service unavailable')
        request = ArmBusTransaction.Request()
        request.operation = operation
        request.session_id = session if session is not None else self.sessions.get(side, '')
        request.token = token
        if expires_ns is not None:
            request.expires_at.sec, request.expires_at.nanosec = divmod(expires_ns, 1_000_000_000)
        if targets is not None:
            request.targets = list(targets)
        wait = 3.0
        if expires_ns is not None:
            remaining = (expires_ns - self.get_clock().now().nanoseconds) * 1e-9
            if remaining <= 0:
                raise ArmTargetExpired(f'{side}: {operation} expired before dispatch')
            wait = min(wait, remaining + .120)
        future = client.call_async(request)
        deadline = time.monotonic() + wait
        # The client response already completes the future. Waiting for a
        # second executor callback adds scheduling latency on every bus call.
        while not future.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                client.remove_pending_request(future)
                raise BusResponseTimeout(f'{side}: {operation} response timed out; outcome unknown')
            time.sleep(min(remaining, .001))
        result = future.result()
        if result is None or not result.success:
            if result is not None and result.message == TARGET_COMMAND_EXPIRED:
                raise ArmTargetExpired(f'{side}: {operation} rejected: {result.message}')
            raise RuntimeError(f'{side}: {operation} rejected: {result.message if result else "no response"}')
        expected_session = session if session is not None else self.sessions.get(side, '')
        if operation != 'inspect' and result.session_id != expected_session:
            raise RuntimeError(f'{side}: session changed during {operation}')
        return result

    def token(self):
        if self.next_token >= 2**64:
            raise RuntimeError('Transaction IDs exhausted')
        result = self.next_token
        self.next_token += 1
        return result

    def abort_prepared(self, prepared, token):
        errors = []
        for side in prepared:
            try:
                self.call_bus(side, 'abort', token)
            except Exception as exc:
                errors.append(str(exc))
        return errors

    def protective_release(self):
        errors = []
        for side in ('left', 'right'):
            try:
                self.call_bus(side, 'protective_release')
            except Exception as exc:
                errors.append(str(exc))
        return errors

    def start(self, request, response):
        if not self.mutex.acquire(blocking=False):
            response.message = 'Another arm transaction is in progress'
            return response
        prepared = []
        committing = False
        try:
            if self.phase != 'idle':
                raise ValueError(f'Cannot start while coordinator is {self.phase}')
            for side in ('left', 'right'):
                result = self.call_bus(side, 'inspect', session='')
                if result.state != 'idle' or not result.session_id:
                    raise ValueError(f'{side}: bus not idle')
                self.sessions[side] = result.session_id
            token = self.token()
            for side in ('left', 'right'):
                self.call_bus(side, 'prepare_hold', token)
                prepared.append(side)
            committing = True  # A lost response can conceal an enable.
            for side in ('left', 'right'):
                self.call_bus(side, 'commit', token)
            self.phase = 'holding'
            response.success = True
            response.message = 'Both buses reported holding; hardware writes were sequential'
        except Exception as exc:
            self.phase = 'fault'
            errors = self.protective_release() if committing else self.abort_prepared(prepared, token if prepared else 0)
            response.message = str(exc) + ('; cleanup unconfirmed: ' + '; '.join(errors) if errors else '')
        finally:
            self.mutex.release()
        return response

    def validate_command(self, request):
        if self.phase != 'holding':
            raise ValueError('Both arms are not in holding mode')
        if request.source_id != self.source:
            raise ValueError('Command source does not match the configured owner')
        if request.sequence <= self.last_sequence:
            raise ValueError('Command sequence repeated or went backwards')
        self.check_command_time(request)
        if any(not math.isfinite(value) for value in request.position_normalized):
            raise ValueError('All 14 targets must be finite')
        if any(not (0. if i % 7 == 6 else -100.) <= value <= 100.
               for i, value in enumerate(request.position_normalized)):
            raise ValueError('Arm target is outside its normalized range')
        with self.state_lock:
            snapshots = dict(self.states)
        for i, side in enumerate(('left', 'right')):
            if side not in snapshots:
                raise ValueError(f'{side}: state unavailable')
            received, state = snapshots[side]
            if (not 0 <= time.monotonic() - received <= .5 or state.arm != side
                    or not state.complete or state.calibration_id != request.arm_calibration_ids[i]
                    or list(state.names) != list(arm_names(side))
                    or len(state.valid) != 7 or not all(state.valid)):
                raise ValueError(f'{side}: state stale, incomplete or calibration mismatch')
        self.last_sequence = request.sequence  # Never replay, even after a timeout.

    def check_command_time(self, request):
        lifetime = request.valid_for.sec * 1_000_000_000 + request.valid_for.nanosec
        stamp = request.header.stamp.sec * 1_000_000_000 + request.header.stamp.nanosec
        now = self.get_clock().now().nanoseconds
        if not 0 < lifetime <= 2_000_000_000 or stamp <= 0 or now < stamp:
            raise ValueError('Command timestamp is invalid')
        if now - stamp >= lifetime:
            raise ArmTargetExpired(TARGET_COMMAND_EXPIRED)

    def set_targets(self, request, response):
        if not self.mutex.acquire(blocking=False):
            response.message = 'Another arm transaction is in progress'
            return response
        prepared = []
        committing = False
        started = time.monotonic()
        sent_ns = None
        try:
            self.validate_command(request)
            token = self.token()
            targets = tuple(request.position_normalized)
            expires_ns = (request.header.stamp.sec * 1_000_000_000 + request.header.stamp.nanosec
                          + request.valid_for.sec * 1_000_000_000 + request.valid_for.nanosec)
            # Each preparation only reads its own serial bus. Join both before
            # committing either so a peer rejection can still abort cleanly.
            failures = []
            pending = {
                side: self.bus_workers.submit(self.call_bus, side, 'prepare_targets', token,
                                              targets[i * 7:(i + 1) * 7], expires_ns=expires_ns)
                for i, side in enumerate(('left', 'right')) if not self.streaming_targets
            }
            for side, task in pending.items():
                try:
                    task.result()
                except Exception as exc:
                    failures.append(exc)
                else:
                    prepared.append(side)
            if failures:
                if (all(isinstance(exc, (ArmTargetExpired, BusResponseTimeout)) for exc in failures)
                        and self.get_clock().now().nanoseconds >= expires_ns):
                    raise ArmTargetExpired('; '.join(map(str, failures)))
                raise RuntimeError('; '.join(map(str, failures)))
            if self.source in ('teleop', 'replay'):
                self.check_command_time(request)
                sent_ns = self.get_clock().now().nanoseconds
                committing = True  # Either independent bus may have started writing.
                operation = 'stream_targets' if self.streaming_targets else 'commit'
                commits = [self.bus_workers.submit(
                    self.call_bus, side, operation, token,
                    targets=targets[i * 7:(i + 1) * 7] if self.streaming_targets else None,
                    expires_ns=expires_ns) for i, side in enumerate(('left', 'right'))]
                for task in commits:
                    try:
                        task.result()
                    except Exception as exc:
                        failures.append(exc)
                if failures:
                    if ((self.streaming_targets or len(failures) == len(commits))
                            and all(isinstance(exc, ArmTargetExpired) for exc in failures)):
                        # Every bus replied: each either accepted the stream
                        # target or confirmed expiry BEFORE writing. Keep the
                        # existing goals and replan, without publishing a full
                        # 14-D command event for this incomplete group.
                        committing = False
                        raise ArmTargetExpired('; '.join(map(str, failures)))
                    raise RuntimeError('; '.join(map(str, failures)))
            else:
                for side in ('left', 'right'):
                    self.check_command_time(request)
                    if sent_ns is None:
                        sent_ns = self.get_clock().now().nanoseconds
                    committing = True
                    self.call_bus(side, 'commit', token, expires_ns=expires_ns)
            response.accepted = True
            response.message = 'Both buses accepted targets; check feedback for arrival'
            try:
                self.publish_recording_command(request, sent_ns)
            except Exception as exc:
                self.get_logger().warning(f'Recording command event unavailable: {exc}',
                                          throttle_duration_sec=1.)
        except Exception as exc:
            if not committing and self.source == 'teleop' and isinstance(exc, ArmTargetExpired):
                # Expiry has a known outcome. Never replay the old request;
                # the next fresh stream frame replans from measured feedback.
                response.message = TELEOP_TARGET_DROPPED
                return response
            if committing:
                self.phase = 'fault'
                errors = self.protective_release()
            else:
                errors = self.abort_prepared(prepared, token if prepared else 0)
                if errors:
                    self.phase = 'fault'
            response.message = str(exc) + ('; cleanup unconfirmed: ' + '; '.join(errors) if errors else '')
            if self.source == 'teleop':
                self.get_logger().warning(f'Arm target rejected: {response.message}')
        finally:
            self.mutex.release()
            elapsed = time.monotonic() - started
            if self.source == 'teleop' and elapsed >= .300:
                self.get_logger().warning(
                    f'Arm target sequence {request.sequence} took {elapsed:.3f} s '
                    f'(commit dispatched={committing}, accepted={response.accepted})',
                    throttle_duration_sec=1.0,
                )
        return response

    def publish_recording_command(self, request, sent_ns):
        """A new accepted request, not a periodically re-stamped held goal."""
        if not self.recording_publisher.get_subscription_count():
            return
        event = {'kind': 'arms', 'sequence': request.sequence,
                 'generated_ns': request.header.stamp.sec * 1_000_000_000 + request.header.stamp.nanosec,
                 'sent_ns': sent_ns, 'accepted_ns': self.get_clock().now().nanoseconds,
                 'targets': list(request.position_normalized),
                 'calibration_ids': list(request.arm_calibration_ids),
                 'sessions': [self.sessions[side] for side in ('left', 'right')]}
        self.recording_publisher.publish(String(data=json.dumps(event, separators=(',', ':'), allow_nan=False)))

    def destroy_node(self):
        self.transaction_workers.shutdown(wait=True)
        self.bus_workers.shutdown(wait=True)
        return super().destroy_node()


def main(args=None):
    sys.setswitchinterval(.001)  # Keep ROS replies and the two bus workers responsive.
    rclpy.init(args=args)
    node = None
    executor = SingleThreadedExecutor()
    try:
        node = ArmGroupCoordinator()
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
