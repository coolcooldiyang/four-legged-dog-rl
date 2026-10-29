from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional


P_MIN = -4.0 * math.pi
P_MAX = 4.0 * math.pi
V_MIN = -30.0
V_MAX = 30.0
KP_MIN = 0.0
KP_MAX = 500.0
KD_MIN = 0.0
KD_MAX = 100.0
T_MIN = -18.0
T_MAX = 18.0

FOC_PAYLOAD = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC])
REST_PAYLOAD = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD])
SET_ZERO_PAYLOAD = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xF1, 0x00])


@dataclass
class BmCommand:
    position: float = 0.0
    velocity: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    torque: float = 0.0


@dataclass
class BmFeedback:
    motor_id: int = 0
    state: int = 0
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0


def float_to_uint(value: float, min_value: float, max_value: float, bits: int) -> int:
    value = max(min_value, min(max_value, float(value)))
    span = max_value - min_value
    return int((value - min_value) * ((1 << bits) - 1) / span)


def uint_to_float(value: int, min_value: float, max_value: float, bits: int) -> float:
    span = max_value - min_value
    return float(value) * span / float((1 << bits) - 1) + min_value


def pack_bm_command(position: float, velocity: float, kp: float, kd: float, torque: float) -> bytes:
    command = BmCommand(position=position, velocity=velocity, kp=kp, kd=kd, torque=torque)
    p = float_to_uint(command.position, P_MIN, P_MAX, 16)
    v = float_to_uint(command.velocity, V_MIN, V_MAX, 12)
    kp_i = float_to_uint(command.kp, KP_MIN, KP_MAX, 12)
    kd_i = float_to_uint(command.kd, KD_MIN, KD_MAX, 12)
    t = float_to_uint(command.torque, T_MIN, T_MAX, 12)
    return bytes(
        [
            (p >> 8) & 0xFF,
            p & 0xFF,
            (v >> 4) & 0xFF,
            ((v & 0xF) << 4) | ((kp_i >> 8) & 0xF),
            kp_i & 0xFF,
            (kd_i >> 4) & 0xFF,
            ((kd_i & 0xF) << 4) | ((t >> 8) & 0xF),
            t & 0xFF,
        ]
    )


def unpack_bm_feedback(data: Iterable[int], source_can_id: Optional[int] = None) -> Optional[BmFeedback]:
    payload = bytes(data)
    if len(payload) < 6:
        return None

    motor_id = payload[0] & 0x0F
    state = payload[0] >> 4
    if motor_id == 0 and source_can_id is not None:
        motor_id = int(source_can_id)

    p_int = (payload[1] << 8) | payload[2]
    v_int = (payload[3] << 4) | (payload[4] >> 4)
    t_int = ((payload[4] & 0x0F) << 8) | payload[5]
    return BmFeedback(
        motor_id=motor_id,
        state=state,
        position=uint_to_float(p_int, P_MIN, P_MAX, 16),
        velocity=uint_to_float(v_int, V_MIN, V_MAX, 12),
        torque=uint_to_float(t_int, T_MIN, T_MAX, 12),
    )
