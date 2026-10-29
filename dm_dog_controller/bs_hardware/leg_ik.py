from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Tuple

H = 0.075
HU = 0.22
HL = 0.22


@dataclass
class TandemLeg:
    x: float = 0.0
    y: float = 0.075
    z: float = -0.07
    theta_p: Tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))

    def initialize_offsets(self) -> None:
        theta1, theta2, theta3 = solve_tandem_leg(self.x, self.y, self.z, side="L")
        self.theta_p = (theta1, theta2, theta3)

    def ik_delta(self, side: str = "L") -> Tuple[float, float, float]:
        theta1, theta2, theta3 = solve_tandem_leg(self.x, self.y, self.z, side=side)
        return (theta1, theta2 - self.theta_p[1], theta3 - self.theta_p[2])


def solve_tandem_leg(x: float, y: float, z: float, side: str = "L") -> Tuple[float, float, float]:
    dyz2 = y * y + z * z
    root = math.sqrt(max(dyz2 - H * H, 1e-9))
    lyz = root
    lxz = math.sqrt(lyz * lyz + x * x)
    lxz = max(lxz, 1e-9)

    if side == "R":
        theta1 = math.atan(y / z) - math.atan(H / root)
    else:
        theta1 = math.atan(y / z) + math.atan(H / root)

    cos_knee = _clamp((lxz * lxz - HU * HU - HL * HL) / (2.0 * HU * HL), -1.0, 1.0)
    theta3 = math.acos(cos_knee)

    cos_hip = _clamp((HU + ((lxz * lxz - HU * HU - HL * HL) / (2.0 * HU))) / lxz, -1.0, 1.0)
    theta2 = math.atan(x / root) - math.acos(cos_hip)
    return theta1, theta2, theta3


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))
