# SPDX-License-Identifier: Apache-2.0
"""Exercise production launch exit handlers and motor shutdown with fake devices."""

import json
import os
from pathlib import Path
import runpy
import signal
import sys
import time
from unittest.mock import Mock, patch

import pytest
from launch import LaunchDescription, LaunchService
from launch.actions import ExecuteProcess, RegisterEventHandler, Shutdown, TimerAction
from launch.event_handlers import OnProcessExit, OnShutdown
from rclpy.node import Node

from test_arm_calibration import fixture
from test_chassis_control import make_control
from test_joint_control import FakeJoint, inspect_target
from xyc_hardware import motor_state_node
from xyc_hardware.arm_joint_control import ArmBusHold, ArmBusTransaction, JointControl
from xyc_hardware.chassis_control import AXIS_IDS, ChassisRos
from xyc_hardware.lift_tracking import LiftTracker


def run_fake_motor_owner(side, directory):
    class FakeMotorOwner(motor_state_node.MotorStateNode):
        def __init__(self):
            # Use the real main loop and shutdown without opening or discovering serial devices.
            Node.__init__(self, 'shutdown_probe_' + side, start_parameter_services=False)
            self.bus = side
            self.reader = Mock()
            self.chassis = None
            self.joint_controls = {}
            for name, entry in fixture(side)[2]['motors'].items():
                driver = FakeJoint(entry)
                driver.bus = side
                control = JointControl(driver, entry)
                control.hold_current_position()
                goal = inspect_target(control, 70.)
                assert goal.target_ticks != driver.position
                driver.write_goal(goal.target_ticks)
                control._record_goal(goal.target_ticks, driver.position)
                self.joint_controls[name] = control
            group = ArmBusHold(side, self.joint_controls, max_prepare_age_s=5.,
                               write_group_goals=Mock(), read_group_snapshot=Mock())
            group.state = 'holding'
            self.arm_transaction = ArmBusTransaction(group, timeout_s=5.)

            if side == 'left':
                control, driver, clock, samples = make_control()
                control.enable()
                control.tracker = LiftTracker(control.lift_geometry, reference_id='shutdown-probe',
                                              zero_tick=2000, reference_time_s=clock[0])
                control.height = 0.
                control.set_mobile([1, 1, 1, 1], deadline=clock[0] + .25)
                clock[0] += .01
                control.observe(samples)
                assert all(control.sent[motor_id] for motor_id in AXIS_IDS)
                self.chassis = object.__new__(ChassisRos)
                self.chassis.control = control
            (directory / f'{side}.ready').touch()

        def destroy_node(self):
            result = super().destroy_node()
            state = {
                'arm_torque': [joint.driver.torque for joint in self.joint_controls.values()],
                'arm_busy': [joint.busy for joint in self.joint_controls.values()],
                'serial_closed': self.reader.close.called,
            }
            if self.chassis is not None:
                control = self.chassis.control
                state['last_velocity_write'] = control.driver.writes[-1]
                state['velocity_owned'] = sorted(control.driver.velocity_owned)
            (directory / f'{side}.stopped').write_text(json.dumps(state), encoding='utf-8')
            return result

    with patch.object(motor_state_node, 'MotorStateNode', FakeMotorOwner):
        motor_state_node.main(args=[])


def kill_control_process(directory):
    deadline = time.monotonic() + 8.
    while not all((directory / f'{side}.ready').exists() for side in ('left', 'right')):
        if time.monotonic() >= deadline:
            raise TimeoutError('Fake motor owners did not reach their initial motion state')
        time.sleep(.01)
    os.kill(os.getpid(), signal.SIGKILL)


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process and ROS signal behavior')
@pytest.mark.parametrize('mode', ['replay', 'policy'])
def test_control_process_killed_stops_all_motor_owners(mode, tmp_path):
    launch_path = Path(__file__).resolve().parents[1] / 'launch' / f'{mode}.launch.py'
    generate = runpy.run_path(str(launch_path))['generate_launch_description']
    with patch.dict(generate.__globals__, Node=Mock()) as scope:
        generate()
        assert scope['Node'].call_args.kwargs['executable'] == 'motion_replay'
        on_exit = scope['Node'].call_args.kwargs.get('on_exit')

    def process(name, *, exit_actions=None):
        return ExecuteProcess(
            cmd=[sys.executable, str(Path(__file__).resolve()), name, str(tmp_path)],
            name=name, on_exit=exit_actions,
            additional_env={'ROS_AUTOMATIC_DISCOVERY_RANGE': 'LOCALHOST'},
        )

    control = process('crash', exit_actions=on_exit)
    exits, shutdowns = [], []

    def record_exit(event, _context):
        exits.append(event.returncode)

    def record_shutdown(event, _context):
        shutdowns.append(event.reason)

    launch = LaunchService()
    launch.include_launch_description(LaunchDescription([
        RegisterEventHandler(OnProcessExit(target_action=control, on_exit=record_exit)),
        RegisterEventHandler(OnShutdown(on_shutdown=record_shutdown)),
        process('left'), process('right'), control,
        TimerAction(period=12., actions=[Shutdown(reason='test timeout')]),
    ]))
    assert launch.run() == 0
    assert exits == [-signal.SIGKILL]
    assert shutdowns == ['motion control process exited']
    for side in ('left', 'right'):
        stopped = json.loads((tmp_path / f'{side}.stopped').read_text(encoding='utf-8'))
        assert stopped['serial_closed']
        assert stopped['arm_torque'] == [0] * 7
        assert stopped['arm_busy'] == [False] * 7
        if side == 'left':
            assert stopped['last_velocity_write'] == {str(motor_id): 0 for motor_id in AXIS_IDS}
            assert stopped['velocity_owned'] == []


if __name__ == '__main__':
    role, output = sys.argv[1:]
    if role == 'crash':
        kill_control_process(Path(output))
    else:
        run_fake_motor_owner(role, Path(output))
