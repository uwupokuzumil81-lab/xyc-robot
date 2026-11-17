# SPDX-License-Identifier: Apache-2.0
"""Calibration tests: coordinate round trips, wrap, file safety and write limits."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from xyc_hardware.arm_calibration import (
    CalibrationReader, RangeRecorder, arm_names, check_hardware, denormalize_tick,
    import_legacy, main, normalize_tick, save_document, validate_document,
)


def fixture(arm='left'):
    models = (2825, 2569, 2569, 2825, 2825, 2825, 2825)
    spec = {
        'arm': arm, 'port': '/dev/test-' + arm, 'baudrate': 1_000_000,
        'motors': dict(zip(arm_names(arm), zip(range(1, 8), models))),
    }
    legacy = {
        name: {'id': motor_id, 'drive_mode': 0, 'homing_offset': -100 + motor_id,
               'range_min': 1000, 'range_max': 3000}
        for name, (motor_id, _) in spec['motors'].items()
    }
    document = import_legacy(spec, legacy)
    return spec, legacy, document


class TestCalibrationCoordinates(unittest.TestCase):
    def test_legacy_import_retains_model_unit_conventions(self):
        spec, _, document = fixture()
        for name in spec['motors']:
            entry = document['motors'][name]
            lower = 0 if name.endswith('_gripper') else -100
            self.assertEqual(normalize_tick(entry, 1000), lower)
            self.assertEqual(normalize_tick(entry, 3000), 100)
            for value in (lower, (lower + 100) / 2, 100):
                self.assertAlmostEqual(normalize_tick(entry, denormalize_tick(entry, value)), value)

    def test_legacy_direction_inversion_and_both_arm_identities(self):
        for arm in ('left', 'right'):
            spec, legacy, _ = fixture(arm)
            for entry in legacy.values():
                entry['drive_mode'] = 1
            document = import_legacy(spec, legacy)
            for entry in document['motors'].values():
                self.assertEqual(normalize_tick(entry, 1000), 100)
                self.assertEqual(denormalize_tick(entry, 100), 1000)

    def test_wrapping_does_not_create_a_false_full_turn_range(self):
        recorder = RangeRecorder({'joint': 4090})
        for tick in (4095, 2, 9, 80, 10, 4090, 4010):
            recorder.update({'joint': tick})
        self.assertEqual(recorder.minimum['joint'], -80)
        self.assertEqual(recorder.maximum['joint'], 86)
        _, _, document = fixture()
        entry = document['motors'][arm_names('left')[0]]
        entry.update(reference_tick=4090, min_delta_ticks=-80, max_delta_ticks=86)
        for tick in (4010, 4090, 0, 80):
            self.assertEqual(denormalize_tick(entry, normalize_tick(entry, tick)), tick)

    def test_out_of_range_and_nonfinite_actions_are_rejected(self):
        _, _, document = fixture()
        entry = document['motors'][arm_names('left')[0]]
        for value in (float('nan'), float('inf'), True, 100.1):
            with self.assertRaises(ValueError):
                denormalize_tick(entry, value)
        for tick in (999, 3001, -1, True):
            with self.assertRaises(ValueError):
                normalize_tick(entry, tick)

    def test_corrupt_sample_is_rejected_before_any_joint_updates(self):
        recorder = RangeRecorder({'a': 1000, 'b': 1000})
        with self.assertRaises(ValueError):
            recorder.update({'a': 1010, 'b': 2000})
        self.assertEqual(recorder.delta, {'a': 0, 'b': 0})
        with self.assertRaises(ValueError):
            recorder.update({'a': 1010})

    def test_repeated_turn_is_rejected_as_ambiguous(self):
        recorder = RangeRecorder({'joint': 0})
        for delta in range(500, 4001, 500):
            recorder.update({'joint': delta % 4096})
        with self.assertRaises(ValueError):
            recorder.update({'joint': 4500 % 4096})

    def test_hardware_offset_or_wrong_arm_cannot_silently_reuse_file(self):
        spec, _, document = fixture()
        settings = deepcopy(document['motors'])
        check_hardware(document, settings, spec)
        settings[arm_names('left')[0]]['hardware_homing_offset'] += 1
        with self.assertRaises(ValueError):
            check_hardware(document, settings, spec)
        right, _, _ = fixture('right')
        with self.assertRaises(ValueError):
            check_hardware(document, document['motors'], right)

    def test_unmoved_joint_and_extra_wheel_are_not_valid_calibration(self):
        _, _, document = fixture()
        bad = deepcopy(document)
        bad['motors'][arm_names('left')[0]]['min_delta_ticks'] = 0
        bad['motors'][arm_names('left')[0]]['max_delta_ticks'] = 10
        with self.assertRaises(ValueError):
            validate_document(bad)
        bad = deepcopy(document)
        bad['motors']['base_left_wheel'] = deepcopy(next(iter(bad['motors'].values())))
        with self.assertRaises(ValueError):
            validate_document(bad)


class TestCalibrationTransport(unittest.TestCase):
    def reader(self, allow_release):
        spec, _, _ = fixture()
        serial = MagicMock()
        serial.write.side_effect = len
        with patch('xyc_hardware.feetech_reader.serial.Serial', return_value=serial):
            reader = CalibrationReader(spec, allow_release=allow_release)
        return reader, serial

    def test_inspection_can_read_settings_but_cannot_release_torque(self):
        reader, serial = self.reader(False)
        reader._request(1, 2, b'\x1f\x03')
        self.assertEqual(serial.write.call_args.args[0][4], 2)
        serial.write.reset_mock()
        with self.assertRaises(ValueError):
            reader._request(1, 3, b'\x28\x00')
        serial.write.assert_not_called()

    def test_only_selected_arm_torque_off_write_is_allowed(self):
        reader, serial = self.reader(True)
        for motor_id, instruction, parameters in (
            (1, 3, b'\x28\x01'), (1, 3, b'\x1f\x00\x00'),
            (1, 3, b'\x2a\x00\x08'), (1, 3, b'\x37\x00'),
            (8, 3, b'\x28\x00'), (11, 3, b'\x28\x00'),
            (254, 3, b'\x28\x00'), (8, 2, b'\x1f\x03'),
        ):
            with self.assertRaises(ValueError):
                reader._request(motor_id, instruction, parameters)
        serial.write.assert_not_called()
        reader._request(7, 3, b'\x28\x00')
        packet = serial.write.call_args.args[0]
        self.assertEqual(packet[2:7], bytes([7, 4, 3, 40, 0]))
        self.assertEqual(packet[-1], (~sum(packet[2:-1])) & 255)

    def test_release_checks_readback_and_attempts_all_seven_on_failure(self):
        reader, _ = self.reader(True)
        with patch.object(reader, '_request') as request, \
             patch.object(reader, '_receive', return_value={}), \
             patch.object(reader, '_read', return_value=b'\x01'):
            with self.assertRaises(RuntimeError):
                reader.release_arm()
        self.assertEqual([call.args[0] for call in request.call_args_list], list(range(1, 8)))


class TestCalibrationPersistence(unittest.TestCase):
    def test_replacement_backs_up_previous_calibration(self):
        _, _, document = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'left.json'
            save_document(path, document)
            original = path.read_bytes()
            updated = deepcopy(document)
            updated['source'] = 'new_passive_recording'
            save_document(path, updated)
            backups = list(path.parent.glob('left.backup-*.json'))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual(json.loads(path.read_text())['source'], 'new_passive_recording')

    def test_failed_atomic_replace_preserves_previous_file(self):
        _, _, document = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'left.json'
            save_document(path, document)
            original = path.read_bytes()
            with patch('xyc_hardware.arm_calibration.os.replace', side_effect=OSError('disk fault')):
                with self.assertRaises(OSError):
                    save_document(path, document)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob('*.tmp')), [])

    def test_recording_requires_an_interactive_terminal_before_opening_port(self):
        with patch('sys.stdin.isatty', return_value=False), \
             patch('xyc_hardware.arm_calibration.CalibrationReader') as reader:
            with self.assertRaises(SystemExit) as result:
                main(['--arm', 'left', '--record'])
        self.assertEqual(result.exception.code, 1)
        reader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
