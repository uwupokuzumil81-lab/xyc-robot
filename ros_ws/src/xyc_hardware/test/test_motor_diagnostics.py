# SPDX-License-Identifier: Apache-2.0
"""Grouped ownership diagnostics and shutdown, without opening a serial port."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import rclpy
from std_srvs.srv import Trigger
from xyc_interfaces.srv import ProtectiveStop

from xyc.robot.calibration import arm_names
from xyc_hardware.arm_joint_control import BoundJointDriver
from xyc_hardware.motor_state_node import MotorStateNode
from test_arm_calibration import fixture


class TestMotorDiagnostics(unittest.TestCase):
    def test_recording_state_does_not_relabel_held_ticks_as_a_new_action(self):
        document = fixture('right')[2]
        names = list(arm_names('right'))
        publisher = MagicMock()
        owner = SimpleNamespace(
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=123_000_000)),
            group_session_id='test-session', chassis=None, recording_publisher=publisher,
            joint_controls={name: SimpleNamespace(owned=True, fault='', target_ticks=2000,
                                                   gripper_contact_latched=False,
                                                   entry=document['motors'][name]) for name in names})
        state = SimpleNamespace(names=names, calibration_id='a' * 64, complete=True,
                                position_normalized=[1.] * 7,
                                sample_stamps=[SimpleNamespace(sec=0, nanosec=120_000_000)] * 7)
        MotorStateNode.publish_recording_sample(owner, state, None)
        sample = json.loads(publisher.publish.call_args.args[0].data)
        self.assertTrue(sample['valid'])
        self.assertNotIn('target', sample)
        self.assertEqual(sample['state'], [1.] * 7)
        owner.joint_controls[names[2]].fault = 'overcurrent'
        MotorStateNode.publish_recording_sample(owner, state, None)
        self.assertFalse(json.loads(publisher.publish.call_args.args[0].data)['valid'])

    def setUp(self):
        document = fixture('right')[2]
        class Driver:
            bus = 'right'
            joint_id = 6
            torques = {2: 1, 7: 1}
            def read_torque(self):
                return self.torques.get(self.joint_id, 0)
            def read_coordinate_settings(self, ids):
                entry = document['motors'][arm_names('right')[ids[0] - 1]]
                return {ids[0]: {k: entry[k] for k in ('hardware_homing_offset', 'operating_mode')}}
        self.driver = Driver()
        self.owner = SimpleNamespace(bus='right', reader=self.driver)

    def test_torque_status_reads_actual_seven_motor_registers(self):
        result = MotorStateNode.read_torque_status(self.owner, Trigger.Request(), Trigger.Response())
        self.assertTrue(result.success)
        values = json.loads(result.message)['torques']
        self.assertEqual(len(values), 7)
        self.assertEqual(values['arm_right_shoulder_lift'], 1)
        self.assertEqual(values['arm_right_elbow_flex'], 0)
        self.assertEqual(values['arm_right_gripper'], 1)
        self.assertEqual(self.driver.joint_id, 6)

    def test_bound_driver_restores_bus_id_after_a_failed_read(self):
        handle = BoundJointDriver(self.driver, 7)
        self.assertEqual(handle.read_torque(), 1)
        self.assertEqual(self.driver.joint_id, 6)
        with self.assertRaises(ValueError):
            handle.read_coordinate_settings([6])
        self.driver.read_torque = MagicMock(side_effect=OSError('injected read failure'))
        with self.assertRaises(OSError):
            handle.read_torque()
        self.assertEqual(self.driver.joint_id, 6)

    def test_latched_control_fault_is_visible_independently_of_live_feedback(self):
        self.owner.joint_controls = {'arm_right_elbow_flex': SimpleNamespace(
            phase='fault', owned=False, fault='overcurrent', last_message='stopped',
            gripper_contact_latched=False, gripper_torque_limit_raw=None)}
        result = MotorStateNode.joint_status(self.owner, Trigger.Request(), Trigger.Response())
        status = json.loads(result.message)['arm_right_elbow_flex']
        self.assertEqual(status['fault'], 'overcurrent')
        self.assertFalse(status['owned'])

    def test_motor_owner_reports_unconfirmed_protective_stop(self):
        logger = MagicMock()
        self.owner.get_logger = lambda: logger
        release = MagicMock(side_effect=OSError('unconfirmed release'))
        self.owner.arm_transaction = SimpleNamespace(protective_release=release)
        self.owner.motion_fault, self.owner.stop_confirmed = 'fault', False
        self.owner.chassis = None
        self.owner.stop_arm = lambda reason: MotorStateNode.stop_arm(self.owner, reason)
        response = MotorStateNode.protective_stop(self.owner, ProtectiveStop.Response())
        self.assertFalse(response.success)
        release.assert_called_once_with('fault')
        logger.error.assert_called_once()
        self.assertIn('unconfirmed release', logger.error.call_args.args[0])

    def test_protective_stop_stops_arm_even_when_chassis_stop_raises(self):
        error = OSError('lost zero-speed write')
        self.owner.chassis = MagicMock()
        self.owner.chassis.protective_stop.side_effect = error
        self.owner.stop_arm = MagicMock(return_value=None)
        self.owner.motion_fault, self.owner.stop_confirmed = 'test fault', False
        self.owner.get_logger = MagicMock()
        response = MotorStateNode.protective_stop(self.owner, ProtectiveStop.Response())
        self.assertFalse(response.success)
        self.assertIn('lost zero-speed write', response.message)
        self.owner.stop_arm.assert_called_once_with('test fault')

    def test_alternating_log_levels_do_not_crash_the_node(self):
        owns_context = not rclpy.ok()
        if owns_context:
            rclpy.init(args=[])
        node = rclpy.create_node('joint_event_logging_test')
        owner = SimpleNamespace(get_logger=node.get_logger)
        try:
            for level in ('info', 'error', 'info', 'error'):
                MotorStateNode.log_joint_event(owner, level, 'test event', 'test_joint')
        finally:
            node.destroy_node()
            if owns_context:
                rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()
