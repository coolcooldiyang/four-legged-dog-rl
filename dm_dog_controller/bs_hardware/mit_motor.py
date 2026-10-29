from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

P_MIN = -12.5
P_MAX = 12.5
V_MIN = -30.0
V_MAX = 30.0
KP_MIN = 0.0
KP_MAX = 500.0
KD_MIN = 0.0
KD_MAX = 5.0
T_MIN = -10.0
T_MAX = 10.0

ENABLE_PAYLOAD = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC])
DISABLE_PAYLOAD = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD])
SAVE_ZERO_PAYLOAD = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE])
CLEAR_ERROR_PAYLOAD = bytes([0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFB])


@dataclass
class MotorFeedback:
    motor_id: int = 0
    state: int = 0
    pos: float = 0.0
    vel: float = 0.0
    tor: float = 0.0
    t_mos: float = 0.0
    t_coil: float = 0.0
    seen: bool = False


@dataclass
class MitCommand:
    pos: float = 0.0
    vel: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    tor: float = 0.0


def float_to_uint(value: float, min_value: float, max_value: float, bits: int) -> int:
    value = max(min_value, min(max_value, float(value)))
    span = max_value - min_value
    return int((value - min_value) * ((1 << bits) - 1) / span)


def uint_to_float(value: int, min_value: float, max_value: float, bits: int) -> float:
    span = max_value - min_value
    return float(value) * span / float((1 << bits) - 1) + min_value


def pack_mit_command(pos: float, vel: float, kp: float, kd: float, tor: float) -> bytes:
    p = float_to_uint(pos, P_MIN, P_MAX, 16)
    v = float_to_uint(vel, V_MIN, V_MAX, 12)
    kp_i = float_to_uint(kp, KP_MIN, KP_MAX, 12)
    kd_i = float_to_uint(kd, KD_MIN, KD_MAX, 12)
    t = float_to_uint(tor, T_MIN, T_MAX, 12)
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


def unpack_feedback(data: Iterable[int]) -> Optional[MotorFeedback]:
    payload = bytes(data)
    if len(payload) < 6:
        return None
    motor_id = payload[0] & 0x0F
    state = payload[0] >> 4
    p_int = (payload[1] << 8) | payload[2]
    v_int = (payload[3] << 4) | (payload[4] >> 4)
    t_int = ((payload[4] & 0x0F) << 8) | payload[5]
    t_mos = float(payload[6]) if len(payload) > 6 else 0.0
    t_coil = float(payload[7]) if len(payload) > 7 else 0.0
    return MotorFeedback(
        motor_id=motor_id,
        state=state,
        pos=uint_to_float(p_int, P_MIN, P_MAX, 16),
        vel=uint_to_float(v_int, V_MIN, V_MAX, 12),
        tor=uint_to_float(t_int, T_MIN, T_MAX, 12),
        t_mos=t_mos,
        t_coil=t_coil,
        seen=True,
    )


def command_as_tuple(cmd: MitCommand) -> Tuple[float, float, float, float, float]:
    return (cmd.pos, cmd.vel, cmd.kp, cmd.kd, cmd.tor)
