# SPDX-License-Identifier: Apache-2.0
"""PC operator logic runs without ROS or serial hardware on Windows."""
import importlib.util
import json
import os
import socket
import sys
import threading
import time
import types
from pathlib import Path
import unittest
from unittest.mock import patch
from unittest.mock import MagicMock

from xyc_hardware.teleop_transport import checked_chassis


tool_override = os.environ.get('XYC_PC_TOOL')
tool_path = Path(tool_override) if tool_override else Path(__file__).resolve().parents[4] / 'tools/teleop_stream.py'
spec = importlib.util.spec_from_file_location('pc_teleop_stream', tool_path)
pc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pc)


def response(ack=None, enabled=True):
    return {'chassis_supported': True,
            'chassis_status': {'enabled': enabled, 'height_m': 0., 'fault': ''},
            'chassis_ack': ack}


class TestPCKeyboard(unittest.TestCase):
    def test_slow_local_writer_initialization_finishes_before_control_connection(self):
        fake = types.ModuleType('leader_arm')
        fake.ROOT = Path(pc.__file__).resolve().parents[1]
        fake.USB_SERIALS = {'left': 'fake-left', 'right': 'fake-right'}
        fake.arm_names = lambda side: tuple(f'{side}_{i}' for i in range(7))
        fake.identity = lambda settings: {}
        fake.load_calibration = lambda *args: {}
        fake.open_spec = lambda side: {'port': 'fake-' + side}
        fake.require_torque_off = lambda settings: None
        fake.LeaderCalibrationReader = MagicMock()
        fake.LeaderCalibrationReader.return_value.settings.return_value = {}
        events, replies, errors = [], [], []
        local = MagicMock()
        def prepare(_directory):
            time.sleep(.3)  # Deliberately longer than the real first-frame deadline.
            events.append('prepared')
            return local
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0))
            listener.listen(1)
            listener.settimeout(3.)
            def bridge():
                try:
                    connection, _ = listener.accept()
                    events.append('accepted')
                    with connection, connection.makefile('rwb') as channel:
                        connection.settimeout(.18)
                        pc.send_frame(channel, {'sequence': 1})
                        replies.append(pc.read_frame(channel))
                except Exception as exc:
                    errors.append(str(exc))
            worker = threading.Thread(target=bridge, daemon=True)
            worker.start()
            args = types.SimpleNamespace(record=True, record_config={'dataset': 'xyc/test', 'task': 'test'},
                                         side='both', keyboard=False, monitor=False, preview=False)
            with patch.dict(sys.modules, {'leader_arm': fake}), patch.object(pc, 'PORT', listener.getsockname()[1]), \
                    patch.object(pc, 'SSHDataTunnel') as tunnel, \
                    patch.object(pc.RecordingRemote, 'start_local', side_effect=prepare), \
                    patch.object(pc, 'leader_stream_sample', side_effect=lambda *a, **kw: ([0.] * 7, time.monotonic_ns(), [])):
                with self.assertRaises(EOFError):
                    pc.run_leader(args)
            worker.join(3.)
        self.assertEqual(errors, [])
        self.assertEqual(events, ['prepared', 'accepted'])
        self.assertEqual(replies[0]['sequence'], 1)
        self.assertEqual(set(replies[0]['positions']), {'left', 'right'})
        local.close.assert_called_once()
        tunnel.assert_called_once_with('raspberrypi.local', 'pi5')

    def test_preview_and_recording_use_a_separate_owned_ssh_connection(self):
        process = MagicMock()
        process.poll.return_value = None
        tunnel = pc.SSHDataTunnel('robot.local', 'operator', wait=lambda _: None)
        with patch.object(tunnel, 'port_open', side_effect=[False, False, True]), \
                patch.object(pc.shutil, 'which', return_value='/usr/bin/ssh'), \
                patch.object(pc.subprocess, 'Popen', return_value=process) as launch:
            self.assertIs(tunnel.start(), tunnel)
        command = launch.call_args.args[0]
        self.assertEqual(command[0], '/usr/bin/ssh')
        self.assertIn('17883:127.0.0.1:17883', command)
        self.assertIn('operator@robot.local', command)
        tunnel.close()
        process.terminate.assert_called_once()

    def test_data_tunnel_refuses_a_shared_or_stale_17883_listener(self):
        tunnel = pc.SSHDataTunnel('robot.local', 'operator')
        with patch.object(tunnel, 'port_open', return_value=True), \
                patch.object(pc.subprocess, 'Popen') as launch, \
                self.assertRaisesRegex(ValueError, '必须由桌面客户端单独建立'):
            tunnel.start()
        launch.assert_not_called()

    def test_recording_is_opt_in_and_does_not_change_plain_teleoperation(self):
        for platform in ('win32', 'linux'):
            with self.subTest(platform=platform), patch.object(pc.sys, 'platform', platform), \
                    patch.object(pc.sys, 'argv', ['teleop_stream.py', 'leader', '--side', 'both', '--keyboard']), \
                    patch.object(pc, 'run_leader') as run:
                self.assertEqual(pc.main(), 0)
                self.assertFalse(run.call_args.args[0].record)
        with patch.object(pc.sys, 'argv', ['teleop_stream.py', 'leader', '--side', 'both',
                                         '--record', '--dataset', 'xyc/test', '--task', '拿起玩具',
                                         '--episode-seconds', '10', '--fps', '30', '--episodes', '5']), \
                patch.object(pc, 'run_leader') as run:
            self.assertEqual(pc.main(), 0)
            args = run.call_args.args[0]
            self.assertTrue(args.keyboard)
            self.assertEqual(args.record_config['episode_frames'], 300)
            self.assertEqual(args.record_config['scope'], 'mobile')
            self.assertEqual(args.record_config['num_episodes'], 5)
            self.assertEqual(args.data_dir, Path(pc.__file__).resolve().parents[1] / 'data' / 'datasets')
            self.assertEqual(args.robot_host, 'raspberrypi.local')
            self.assertEqual(args.robot_user, 'pi5')

    def test_local_recording_waits_for_matching_backend_and_data_channel(self):
        from types import SimpleNamespace
        ready = threading.Event()
        recorder = pc.RecordingRemote({'task': 'pick', 'dataset': 'xyc/test'})
        recorder.local = SimpleNamespace(ready=ready)
        recorder.receive({'recording_supported': True, 'recording_status': {'phase': 'idle'}})
        self.assertIsNone(recorder.command())  # Never ask an old backend to save on the Pi.
        recorder.receive({'recording_supported': True, 'recording_status': {'phase': 'idle', 'storage': 'pc'}})
        self.assertIsNone(recorder.command())
        ready.set()
        recorder.receive({'recording_supported': True, 'recording_status': {'phase': 'idle', 'storage': 'pc'}})
        self.assertIsNone(recorder.command())
        recorder.key('r')
        self.assertEqual(recorder.command()['operation'], 'start')
        recovered = pc.RecordingRemote({'task': 'pick', 'dataset': 'xyc/test', 'resume': True})
        recovered.local = SimpleNamespace(ready=ready)
        recovered.receive({'recording_supported': True, 'recording_status': {
            'phase': 'error', 'storage': 'pc', 'buffering': 'robot_episode_cache'}})
        self.assertIsNone(recovered.command())
        recovered.key('r')
        self.assertEqual(recovered.command()['operation'], 'start')

    def test_recording_hotkeys_are_separate_from_motion_and_survive_recording_failure(self):
        recorder = pc.RecordingRemote({'task': 'pick', 'dataset': 'xyc/test'})
        keyboard = pc.LocalChassisKeyboard(recorder)
        recorder.receive({'recording_supported': True, 'recording_status': {'phase': 'idle'}})
        self.assertIsNone(recorder.command())
        keyboard.handle_key('R')
        self.assertEqual(recorder.command()['operation'], 'start')
        recorder.receive({'recording_supported': True, 'recording_ack': {
            'id': 1, 'accepted': False, 'message': 'disk full'}})
        self.assertIn('disk full', recorder.summary())
        keyboard.handle_key('B')
        self.assertEqual(recorder.command()['operation'], 'save')
        recorder.receive({'recording_supported': True, 'recording_ack': {'id': 2, 'accepted': True}})
        keyboard.handle_key('N')
        self.assertEqual(recorder.command()['operation'], 'discard')
        self.assertEqual(keyboard.base, (0., 0., 0.))
        self.assertEqual(keyboard.lift, 0.)

    def test_home_cannot_interrupt_an_open_or_starting_recording(self):
        for phase in ('waiting', 'recording', 'saving'):
            with self.subTest(phase=phase):
                recorder = pc.RecordingRemote({'dataset': 'xyc/test', 'task': 'pick'})
                recorder.receive({'recording_supported': True, 'recording_status': {'phase': phase}})
                keyboard = self.ready_keyboard()
                keyboard.recording = recorder
                keyboard.handle_key('H', 1.1)
                self.assertIsNone(keyboard.operation)
                self.assertIn('先 B 保存或 N 丢弃', keyboard.notice)
        recorder.status = {'phase': 'idle'}
        recorder.key('r')
        keyboard.handle_key('h', 1.2)
        self.assertIsNone(keyboard.operation)
        recorder.operation = None
        keyboard.handle_key('h', 1.3)
        self.assertEqual(keyboard.operation['name'], 'home_lift')

    def test_preview_opening_is_separate_from_robot_control_and_failure_is_nonfatal(self):
        with patch('urllib.request.build_opener') as build, patch('webbrowser.open', return_value=True) as browser:
            request = build.return_value.open
            request.return_value.__enter__.return_value.status = 200
            self.assertTrue(pc.open_camera_preview())
            browser.assert_called_once_with('http://localhost:17883', new=2)
        with patch('urllib.request.build_opener') as build, patch('webbrowser.open') as browser:
            build.return_value.open.side_effect = OSError('not forwarded')
            self.assertFalse(pc.open_camera_preview(wait=lambda _: None))
            browser.assert_not_called()

    def test_browser_open_retries_when_camera_backend_is_still_starting(self):
        from unittest.mock import MagicMock
        ready = MagicMock()
        ready.__enter__.return_value.status = 200
        with patch('urllib.request.build_opener') as build, \
                patch('webbrowser.open', return_value=True) as browser:
            request = build.return_value.open
            request.side_effect = [OSError('starting'), ready]
            self.assertTrue(pc.open_camera_preview(wait=lambda _: None))
            self.assertEqual(request.call_count, 2)
            browser.assert_called_once()

    @unittest.skipUnless(socket.has_ipv6, 'IPv6 unavailable')
    def test_preview_opens_with_an_ipv6_only_ssh_listener_even_when_proxy_is_configured(self):
        from http.server import BaseHTTPRequestHandler, HTTPServer
        class IPv6Server(HTTPServer):
            address_family = socket.AF_INET6
        class Ready(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-Length', '2')
                self.end_headers()
                self.wfile.write(b'{}')
        try:
            server = IPv6Server(('::1', 0), Ready)
        except OSError as exc:
            self.skipTest(str(exc))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = f'http://localhost:{server.server_address[1]}'
        try:
            with patch.object(pc, 'PREVIEW_URL', url), patch('webbrowser.open', return_value=True) as browser, \
                    patch.dict(os.environ, {'http_proxy': 'http://127.0.0.1:1', 'no_proxy': ''}):
                self.assertTrue(pc.open_camera_preview(wait=lambda _: None))
                browser.assert_called_once_with(url, new=2)
        finally:
            server.shutdown()
            server.server_close()
            worker.join(2.)

    def test_preview_defaults_to_automatic_on_both_supported_desktops(self):
        for platform in ('win32', 'linux'):
            with self.subTest(platform=platform), patch.object(pc.sys, 'platform', platform), \
                    patch.object(pc.sys, 'argv', ['teleop_stream.py', 'leader', '--side', 'both', '--keyboard']), \
                    patch.object(pc, 'run_leader') as run:
                self.assertEqual(pc.main(), 0)
                self.assertTrue(run.call_args.args[0].preview)

    def ready_keyboard(self):
        keyboard = pc.LocalChassisKeyboard()
        keyboard.receive(response({'id': 1, 'accepted': True, 'message': 'ready'}), 1.)
        return keyboard

    def test_enable_is_automatic_and_directions_wait_for_fresh_enabled_status(self):
        keyboard = pc.LocalChassisKeyboard()
        keyboard.receive(response(enabled=False), 1.)
        keyboard.handle_key('w', 1.)
        self.assertEqual(keyboard.command(1.01)['base'], [0., 0., 0.])
        self.assertEqual(keyboard.command(1.03)['operation'], {'id': 1, 'name': 'enable'})
        keyboard.receive(response({'id': 1, 'accepted': True}), 1.04)
        keyboard.handle_key('w', 1.05)
        self.assertEqual(keyboard.command(1.06)['base'], [.15, 0., 0.])
        keyboard.handle_key('e', 1.07)
        self.assertIsNone(keyboard.command(1.08)['operation'])

    def test_enabled_robot_status_is_sufficient_even_before_operation_ack(self):
        keyboard = pc.LocalChassisKeyboard()
        keyboard.receive(response(), 1.)
        keyboard.handle_key('W', 1.01)
        self.assertEqual(keyboard.command(1.02)['base'], [.15, 0., 0.])

    def test_pending_ack_keeps_automatic_enable_and_failure_survives_voltage_notice(self):
        keyboard = pc.LocalChassisKeyboard()
        keyboard.receive(response({'id': 1, 'pending': True, 'message': '正在等待底盘控制服务'},
                                  enabled=False), 1.)
        self.assertEqual(keyboard.command(1.01)['operation'], {'id': 1, 'name': 'enable'})
        keyboard.receive(response({'id': 1, 'accepted': False, 'message': 'ID 8 寄存器核对失败'},
                                  enabled=False), 1.02)
        keyboard.set_notice('right 主臂 ID 6 电压状态异常')
        self.assertIn('ID 8 寄存器核对失败', keyboard.control_error)
        self.assertIsNone(keyboard.command(1.03)['operation'])

    def test_upper_and_lower_case_operator_keys_have_identical_effects(self):
        for key in 'wasdjlikhq':
            with self.subTest(key=key):
                lower, upper = self.ready_keyboard(), self.ready_keyboard()
                lower.handle_key(key, 1.01)
                upper.handle_key(key.upper(), 1.01)
                self.assertEqual(upper.command(1.02), lower.command(1.02))
                self.assertEqual(upper.quit_at, lower.quit_at)

    def test_key_release_and_stale_robot_status_produce_zero_velocity(self):
        keyboard = self.ready_keyboard()
        keyboard.handle_key('w', 1.01)
        keyboard.handle_key('i', 1.01)
        self.assertEqual(keyboard.command(1.02)['lift_m_s'], .015)
        self.assertEqual(keyboard.command(1.27)['base'], [0., 0., 0.])
        self.assertEqual(keyboard.command(1.27)['lift_m_s'], 0.)
        keyboard.handle_key('w', 1.6)
        self.assertEqual(keyboard.command(1.61)['base'], [0., 0., 0.])

    def test_homing_keeps_packets_live_but_ignores_direction_keys(self):
        keyboard = self.ready_keyboard()
        keyboard.handle_key('w', 1.01)
        keyboard.handle_key('i', 1.01)
        for phase in ('seeking', 'settling'):
            with self.subTest(phase=phase):
                state = response()
                state['chassis_status']['homing'] = phase
                keyboard.receive(state, 1.02)
                keyboard.handle_key('K', 1.03)
                self.assertEqual(keyboard.command(1.04)['base'], [0., 0., 0.])
                self.assertEqual(keyboard.command(1.04)['lift_m_s'], 0.)
                self.assertIn('等待完成', keyboard.notice)
        keyboard.receive(response(), 1.1)
        keyboard.handle_key('I', 1.11)
        self.assertEqual(keyboard.command(1.12)['lift_m_s'], .015)

    def test_space_overrides_pending_home_and_old_ack_cannot_clear_new_stop(self):
        keyboard = self.ready_keyboard()
        keyboard.handle_key('h', 1.1)
        keyboard.handle_key(' ', 1.11)
        self.assertEqual(keyboard.command(1.12)['operation'], {'id': 3, 'name': 'stop'})
        keyboard.receive(response({'id': 2, 'accepted': True}), 1.12)
        self.assertIsNotNone(keyboard.command(1.13)['operation'])
        keyboard.receive(response({'id': 3, 'accepted': True}), 1.14)
        self.assertIsNone(keyboard.command(1.15)['operation'])

    def test_quit_waits_for_stop_confirmation_or_a_bounded_disconnect(self):
        keyboard = self.ready_keyboard()
        keyboard.handle_key('q', 1.1)
        with patch.object(pc.time, 'monotonic', return_value=1.2):
            self.assertFalse(keyboard.should_quit())
        keyboard.receive(response({'id': 2, 'accepted': True}), 1.21)
        with patch.object(pc.time, 'monotonic', return_value=1.22):
            self.assertTrue(keyboard.should_quit())

    def test_server_capability_and_velocity_validation_reject_invalid_inputs(self):
        keyboard = pc.LocalChassisKeyboard()
        with self.assertRaisesRegex(ValueError, 'chassis_mode'):
            keyboard.receive({}, 1.)
        for payload in ({'base': [1, 0, 0], 'lift_m_s': 0},
                        {'base': [.1, 0, 0], 'lift_m_s': 0},
                        {'base': [0, 0, 0], 'lift_m_s': float('nan')},
                        {'base': [0, 0, 0], 'lift_m_s': 0, 'operation': {'id': True, 'name': 'enable'}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                checked_chassis(payload)
        self.assertEqual(checked_chassis(keyboard.command(1.))[0], (0., 0., 0.))

    @unittest.skipUnless(pc.sys.platform == 'linux', 'Linux terminal test')
    def test_linux_terminal_reads_keys_and_restores_attributes(self):
        import pty
        import select
        import termios
        master, slave = pty.openpty()
        try:
            with os.fdopen(slave, 'r') as stream:
                original = termios.tcgetattr(stream.fileno())
                with pc.ConsoleKeys(stream) as reader:
                    os.write(master, b'w\x1b[Ai')
                    select.select([stream], [], [], .2)
                    self.assertEqual(reader.poll(), ['w', 'i'])
                    self.assertIsNone(reader.is_held('w'))
                self.assertEqual(termios.tcgetattr(stream.fileno()), original)
        finally:
            os.close(master)

    def test_windows_held_key_checks_release_and_console_focus(self):
        reader = pc.ConsoleKeys()
        reader.window = 100
        reader.foreground = lambda: 100
        reader.key_state = lambda code: 0x8000
        with patch.object(pc.sys, 'platform', 'win32'):
            self.assertTrue(reader.is_held('w'))
            reader.foreground = lambda: 101
            self.assertFalse(reader.is_held('w'))
            reader.foreground = lambda: 100
            reader.key_state = lambda code: 0
            self.assertFalse(reader.is_held('w'))


if __name__ == '__main__':
    unittest.main()
