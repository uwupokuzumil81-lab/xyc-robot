# SPDX-License-Identifier: Apache-2.0
"""Protocol tests using a captured model reply and malformed/partial traffic."""
import unittest
from unittest.mock import Mock, patch

import serial

from xyc.robot.feetech import FeetechReader, Reply, decode_feedback
from xyc_hardware.arm_joint_control import JointControl


class FakeSerial:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.writes = []

    def reset_input_buffer(self):
        pass

    def write(self, data):
        self.writes.append(data)
        return len(data)

    def read(self, size):
        return self.chunks.pop(0) if self.chunks else b''


class TestReadOnlyProtocol(unittest.TestCase):
    def reader(self, chunks):
        fake = FakeSerial(chunks)
        with patch('xyc.robot.feetech.serial.Serial', return_value=fake):
            reader = FeetechReader('/unused', [1], timeout_s=0.002, clock_ns=lambda: 123)
        return reader, fake

    def test_captured_reply_survives_noise_and_fragmentation(self):
        reader, fake = self.reader([
            b'noise\xff', b'\xff\x01\x04', b'\x00\x09', b'\x0b\xe6',
        ])
        self.assertEqual(reader.read_models(), {1: 2825})
        self.assertEqual(fake.writes, [bytes.fromhex('ffff0104020302f3')])

    def test_corrupt_length_and_checksum_do_not_hide_good_reply(self):
        reader, _ = self.reader([
            bytes.fromhex('ffff01f000ffff010400090be5ffff010400090be6'),
        ])
        self.assertEqual(reader.read_models(), {1: 2825})

    def test_missing_feedback_is_missing_not_cached_or_zero_success(self):
        reader, fake = self.reader([])
        self.assertEqual(reader.read_feedback(), {})
        self.assertEqual(fake.writes[0][4], 0x82)

    def test_serial_write_budget_is_independent_of_feedback_poll_period(self):
        with patch('xyc.robot.feetech.serial.Serial') as serial_class:
            FeetechReader('/unused', [1], timeout_s=.014)
        self.assertEqual(serial_class.call_args.kwargs['write_timeout'], .05)
        self.assertEqual(serial_class.call_args.kwargs['timeout'], 0)

    def test_incomplete_poll_retries_a_fresh_collection_and_never_merges_rounds(self):
        for recovered in (False, True):
            reader, fake = self.reader([])
            reader.motor_ids = (1, 2)
            first = {1: Reply(b'old', 0, 1)}
            second = {2: Reply(b'fresh', 0, 2)}
            if recovered:
                second[1] = Reply(b'fresh', 0, 3)
            with patch.object(reader, '_receive', side_effect=[first, second]):
                self.assertEqual(reader.read_feedback(), second)
            self.assertEqual(len(fake.writes), 2)
            self.assertEqual(fake.writes[0], fake.writes[1])
            self.assertTrue(all(packet[4] == 0x82 for packet in fake.writes))

    def test_single_feedback_write_timeout_retries_only_a_fresh_read(self):
        raw = bytes.fromhex('010802800304792100000100000200')
        body = bytes([1, len(raw) + 2, 0]) + raw
        packet = b'\xff\xff' + body + bytes([(~sum(body)) & 255])
        reader, fake = self.reader([packet])
        fake.write = Mock(side_effect=[serial.SerialTimeoutException('Write timeout'),
                                       len(bytes.fromhex('fffffe0582380f0132'))])
        result = reader.read_feedback()
        self.assertEqual(result[1].data, raw)
        self.assertEqual(fake.write.call_count, 2)
        self.assertEqual(fake.write.call_args_list[0], fake.write.call_args_list[1])
        self.assertEqual(fake.write.call_args.args[0][4], 0x82)

    def test_persistent_write_timeout_or_disconnect_is_not_hidden(self):
        for error, calls in ((serial.SerialTimeoutException('Write timeout'), 2),
                             (serial.SerialException('disconnected'), 1)):
            reader, fake = self.reader([])
            fake.write = Mock(side_effect=error)
            with self.assertRaises(type(error)):
                reader.read_feedback()
            self.assertEqual(fake.write.call_count, calls)

    def test_actuation_and_arbitrary_register_requests_are_rejected(self):
        reader, fake = self.reader([])
        for instruction, parameters in [(3, b'\x28\x01'), (2, b'\x28\x01')]:
            with self.assertRaises(ValueError):
                reader._request(1, instruction, parameters)
        self.assertEqual(fake.writes, [])

    def test_signed_feedback_and_engineering_units(self):
        raw = bytes.fromhex('010802800304792100000100000200')
        data = decode_feedback(raw)
        self.assertEqual(data['position_ticks'], 2049)
        self.assertEqual(data['velocity_raw'], -2)
        self.assertEqual(data['load_raw'], -3)
        self.assertAlmostEqual(data['voltage_v'], 12.1)
        self.assertEqual(data['temperature_c'], 33.0)
        self.assertAlmostEqual(data['current_a'], 0.013)
        self.assertTrue(data['moving'])

    def test_coordinate_snapshot_decodes_actual_signed_offset_without_writes(self):
        reader, fake = self.reader([bytes.fromhex('ffff010500dd0e000e')])
        self.assertEqual(reader.read_coordinate_settings([1]), {
            1: {'hardware_homing_offset': -1757, 'operating_mode': 0},
        })
        self.assertEqual(fake.writes[0][4:7], bytes([2, 31, 3]))
        before = len(fake.writes)
        for ids in ([8], [1, 1], [], [True]):
            with self.assertRaises(ValueError):
                reader.read_coordinate_settings(ids)
        self.assertEqual(len(fake.writes), before)

    def test_signed_current_decodes_direction_and_validates_finite_magnitude(self):
        raw = bytearray.fromhex('010802800304792100000100000200')
        for magnitude in (2, 47, 100):
            for sign in (0, 0x8000):
                raw[13:15] = (magnitude | sign).to_bytes(2, 'little')
                sample = decode_feedback(bytes(raw))
                expected = magnitude * .0065 * (-1 if sign else 1)
                self.assertAlmostEqual(sample['current_a'], expected)
                JointControl.validate_sample(sample)


if __name__ == '__main__':
    unittest.main()
