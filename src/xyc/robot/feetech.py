# SPDX-License-Identifier: Apache-2.0
"""Small read-only STS transport, independent of ROS and the old robot project.

Only READ (0x02) and SYNC_READ (0x82) requests are emitted. Register addresses
and wire framing follow Feetech's public STS protocol and the verified robot
configuration. No torque, goal, calibration, mode, or EEPROM writes are exposed.
"""

from dataclasses import dataclass
import math
import time
from typing import Callable

import serial


MODEL_NAMES = {777: 'STS3215', 2569: 'STS3095', 2825: 'STS3250'}
SERIAL_WRITE_TIMEOUT_S = .05


@dataclass(frozen=True)
class Reply:
    data: bytes
    servo_error: int
    stamp_ns: int


def signed_magnitude(value: int, sign_bit: int) -> int:
    magnitude = value & ((1 << sign_bit) - 1)
    return -magnitude if value & (1 << sign_bit) else magnitude


def decode_feedback(data: bytes) -> dict:
    """Decode STS registers 56..70; load_raw is not physical joint torque."""
    if len(data) != 15:
        raise ValueError('STS feedback must contain 15 bytes')
    return {
        'position_ticks': signed_magnitude(int.from_bytes(data[0:2], 'little'), 15),
        'velocity_raw': signed_magnitude(int.from_bytes(data[2:4], 'little'), 15),
        'load_raw': signed_magnitude(int.from_bytes(data[4:6], 'little'), 10),
        'voltage_v': data[6] * 0.1,
        'temperature_c': float(data[7]),
        'current_a': signed_magnitude(int.from_bytes(data[13:15], 'little'), 15) * 0.0065,
        'moving': bool(data[10]),
    }


class FeetechReader:
    """One exclusive serial connection, owned by one node's executor thread."""

    def __init__(
        self,
        device: str,
        motor_ids: list[int],
        *,
        baudrate: int = 1_000_000,
        timeout_s: float = 0.02,
        clock_ns: Callable[[], int] = time.time_ns,
    ):
        if not motor_ids or len(set(motor_ids)) != len(motor_ids):
            raise ValueError('Provide distinct motor IDs')
        if any(type(i) is not int or not 1 <= i <= 253 for i in motor_ids):
            raise ValueError('Motor IDs must be integers in 1..253')
        if len(motor_ids) > 242:
            raise ValueError('Too many motor IDs in a sync-read request')
        if type(baudrate) is not int or baudrate <= 0:
            raise ValueError('baudrate must be a positive integer')
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('timeout_s must be finite and positive')
        self.motor_ids = tuple(motor_ids)
        self.timeout_s = timeout_s
        self.clock_ns = clock_ns
        # A closed object allows cleanup even when opening partially fails.
        self.port = serial.Serial(
            port=None, baudrate=baudrate, bytesize=8, parity='N', stopbits=1,
            timeout=0, write_timeout=SERIAL_WRITE_TIMEOUT_S, exclusive=True,
        )
        self.port.port = device

    def open(self):
        self.port.open()
        # Match the verified SDK's default control lines and allow USB settling.
        time.sleep(0.5)

    def close(self):
        self.port.close()

    def _request(self, motor_id: int, instruction: int, parameters: bytes):
        if instruction == 0x02:
            allowed = (
                motor_id in self.motor_ids
                and parameters in (b'\x03\x02', b'\x1f\x03', b'\x38\x0f')
            )
        elif instruction == 0x82:
            allowed = (
                motor_id == 254
                and parameters == b'\x38\x0f' + bytes(self.motor_ids)
            )
        else:
            allowed = False
        if not allowed:
            raise ValueError('Only configured read-only STS requests are allowed')
        body = bytes([motor_id, len(parameters) + 2, instruction]) + parameters
        packet = b'\xff\xff' + body + bytes([(~sum(body)) & 255])
        self.port.reset_input_buffer()
        if self.port.write(packet) != len(packet):
            raise OSError('Incomplete serial read-request transmission')

    def _receive(self, expected_ids: tuple[int, ...], size: int, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        received = bytearray()
        replies = {}
        while time.monotonic() < deadline and len(replies) < len(expected_ids):
            chunk = self.port.read(4096)
            if chunk:
                received.extend(chunk)
            else:
                time.sleep(0.0001)
            while len(received) >= 2:
                start = received.find(b'\xff\xff')
                if start < 0:
                    received[:] = received[-1:] if received[-1:] == b'\xff' else b''
                    break
                if start:
                    del received[:start]
                if len(received) < 4:
                    break
                # The reply length must match this transaction. Do not let a
                # malformed large length hide a later healthy packet.
                if received[2] > 253 or received[3] != size + 2:
                    del received[0]
                    continue
                packet_size = size + 6
                if len(received) < packet_size:
                    break
                packet = bytes(received[:packet_size])
                if packet[-1] != ((~sum(packet[2:-1])) & 255):
                    del received[0]
                    continue
                del received[:packet_size]
                motor_id = packet[2]
                if motor_id in expected_ids and motor_id not in replies:
                    replies[motor_id] = Reply(
                        data=packet[5:-1], servo_error=packet[4],
                        stamp_ns=self.clock_ns(),
                    )
        return replies

    def read_models(self) -> dict[int, int]:
        result = {}
        for motor_id in self.motor_ids:
            self._request(motor_id, 0x02, b'\x03\x02')
            reply = self._receive((motor_id,), 2, max(self.timeout_s, 0.1)).get(motor_id)
            if reply is None:
                raise ConnectionError(f'No model reply from motor ID {motor_id}')
            if reply.servo_error:
                raise ConnectionError(
                    f'Motor ID {motor_id} reports error 0x{reply.servo_error:02x}'
                )
            result[motor_id] = int.from_bytes(reply.data, 'little')
        return result

    def read_coordinate_settings(self, motor_ids: list[int]) -> dict[int, dict]:
        """Read homing offset/mode for a configured subset; never write EEPROM."""
        if (not motor_ids or len(set(motor_ids)) != len(motor_ids)
                or any(type(i) is not int or i not in self.motor_ids for i in motor_ids)):
            raise ValueError('Coordinate reads require distinct configured motor IDs')
        result = {}
        for motor_id in motor_ids:
            self._request(motor_id, 0x02, b'\x1f\x03')
            reply = self._receive((motor_id,), 3, max(self.timeout_s, 0.1)).get(motor_id)
            if reply is None or reply.servo_error:
                raise ConnectionError(f'ID {motor_id}: failed to read homing offset/mode')
            result[motor_id] = {
                'hardware_homing_offset': signed_magnitude(
                    int.from_bytes(reply.data[:2], 'little'), 11,
                ),
                'operating_mode': reply.data[2],
            }
        return result

    def read_feedback(self) -> dict[int, Reply]:
        for attempt in range(2):
            try:
                self._request(254, 0x82, b'\x38\x0f' + bytes(self.motor_ids))
                replies = self._receive(self.motor_ids, 15, self.timeout_s)
                if len(replies) == len(self.motor_ids) or attempt:
                    return replies
            except serial.SerialTimeoutException:
                if attempt:
                    raise
            # Retry only the READ after incomplete replies or a transmit timeout.
            # The second collection replaces the first; never combine old and
            # new replies to manufacture a complete feedback frame.
