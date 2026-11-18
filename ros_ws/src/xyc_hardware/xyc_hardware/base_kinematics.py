# SPDX-License-Identifier: Apache-2.0
"""Three-wheel body/wheel conversion only; it never accesses hardware.

Wheel geometry, mounting angles and signs are REQUIRED inputs. Nothing in this
module selects robot-specific dimensions or sends velocity commands.
"""

from dataclasses import dataclass
import math


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return float(value)


def _solve(matrix, values):
    """Solve an invertible 3x3 system with partial pivoting."""
    rows = [list(row) + [value] for row, value in zip(matrix, values)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(rows[row][col]))
        if abs(rows[pivot][col]) < 1e-10:
            raise ValueError('Wheel geometry is singular')
        rows[col], rows[pivot] = rows[pivot], rows[col]
        divisor = rows[col][col]
        rows[col] = [x / divisor for x in rows[col]]
        for row in range(3):
            if row == col:
                continue
            factor = rows[row][col]
            rows[row] = [x - factor * y for x, y in zip(rows[row], rows[col])]
    return tuple(row[3] for row in rows)


@dataclass(frozen=True)
class OmniGeometry:
    wheel_radius_m: float
    base_radius_m: float
    wheel_angles_rad: tuple
    wheel_signs: tuple
    max_wheel_rad_s: float

    def validate(self):
        if _number(self.wheel_radius_m, 'wheel_radius_m') <= 0:
            raise ValueError('wheel_radius_m must be positive')
        if _number(self.base_radius_m, 'base_radius_m') <= 0:
            raise ValueError('base_radius_m must be positive')
        if _number(self.max_wheel_rad_s, 'max_wheel_rad_s') <= 0:
            raise ValueError('max_wheel_rad_s must be positive')
        if len(self.wheel_angles_rad) != 3 or len(self.wheel_signs) != 3:
            raise ValueError('Provide three ordered wheel angles and signs')
        for angle in self.wheel_angles_rad:
            _number(angle, 'wheel angle')
        if any(type(sign) is not int or sign not in (-1, 1) for sign in self.wheel_signs):
            raise ValueError('Wheel signs must each be +1 or -1')
        _solve(self.matrix(), (0., 0., 0.))
        return self

    def matrix(self):
        return tuple(tuple(sign * item / self.wheel_radius_m for item in
                           (-math.cos(angle), -math.sin(angle), self.base_radius_m))
                     for angle, sign in zip(self.wheel_angles_rad, self.wheel_signs))

    def body_to_wheels(self, x_m_s, y_m_s, yaw_rad_s):
        self.validate()
        body = tuple(_number(value, name) for value, name in
                     zip((x_m_s, y_m_s, yaw_rad_s), ('x', 'y', 'yaw')))
        result = tuple(sum(a * b for a, b in zip(row, body)) for row in self.matrix())
        if any(abs(value) > self.max_wheel_rad_s for value in result):
            raise ValueError('Wheel target exceeds verified limit; no clipping')
        return result

    def wheels_to_body(self, wheel_rad_s):
        self.validate()
        if len(wheel_rad_s) != 3:
            raise ValueError('Provide three wheel velocities in configured order')
        measured = tuple(_number(value, 'wheel velocity') for value in wheel_rad_s)
        return _solve(self.matrix(), measured)

    def constrain_body_velocity(self, x_m_s, y_m_s, yaw_rad_s):
        """Preserve body direction/rotation ratios when a combined goal saturates."""
        self.validate()
        body = tuple(_number(value, 'body velocity') for value in (x_m_s, y_m_s, yaw_rad_s))
        peak = max(abs(sum(a * b for a, b in zip(row, body))) for row in self.matrix())
        scale = min(1., self.max_wheel_rad_s / peak) if peak else 1.
        # Keep floating-point reconstruction just inside the verified wheel limit.
        if scale < 1.:
            scale *= 1. - 1e-12
        return tuple(value * scale for value in body)
