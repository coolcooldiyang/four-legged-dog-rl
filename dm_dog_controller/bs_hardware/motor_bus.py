from __future__ import annotations

from contextlib import contextmanager
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .config import BM_CALF_KD, BM_CALF_KP
from .bm_motor import (
    FOC_PAYLOAD as BM_FOC_PAYLOAD,
    REST_PAYLOAD as BM_REST_PAYLOAD,
    pack_bm_command,
    unpack_bm_feedback,
)
from .can_hub import Usb2CanHub
from .mit_motor import (
    CLEAR_ERROR_PAYLOAD,
    DISABLE_PAYLOAD,
    ENABLE_PAYLOAD,
    MitCommand,
    MotorFeedback,
    pack_mit_command,
    unpack_feedback,
)

MAX_FEEDBACK_POSITION_JUMP = 0.75
BM_CALF_JOINT = 2


@dataclass(frozen=True)
class JointAddress:
    leg: int
    joint: int
    bus: int
    motor_id: int
    serial: str
    channel: int


class MotorBus:
    def __init__(self, can: Usb2CanHub, bus_map: Sequence[Tuple[str, int]]) -> None:
        if len(bus_map) != 4:
            raise ValueError("bus_map must contain four CAN buses")
        self.can = can
        self.bus_map = list(bus_map)
        self.addresses = _make_joint_addresses(self.bus_map)
        self.feedback: Dict[Tuple[int, int], MotorFeedback] = {
            (addr.leg, addr.joint): MotorFeedback(motor_id=addr.motor_id) for addr in self.addresses
        }
        self.commands = np.zeros((4, 3), dtype=np.float32)
        self.vel = np.zeros((4, 3), dtype=np.float32)
        self.kp = np.full((4, 3), 70.0, dtype=np.float32)
        self.kd = np.full((4, 3), 1.4, dtype=np.float32)
        self.tor = np.zeros((4, 3), dtype=np.float32)
        self.rejected_feedback = 0
        self.rejected_feedback_id = 0
        self._bm_calf_kp = float(BM_CALF_KP)
        self._bm_calf_kd = float(BM_CALF_KD)

    def send_mit(self, leg: int, joint: int, command: MitCommand) -> None:
        addr = self._addr(leg, joint)
        kp = float(command.kp)
        kd = float(command.kd)
        if self._is_bm_joint(leg, joint):
            is_zero_torque_query = (
                abs(command.vel) < 1e-9 and
                abs(command.kp) < 1e-9 and
                abs(command.kd) < 1e-9 and
                abs(command.tor) < 1e-9
            )
            if not is_zero_torque_query:
                kp = self._bm_calf_kp
                kd = self._bm_calf_kd
        self.commands[leg, joint] = command.pos
        self.vel[leg, joint] = command.vel
        self.kp[leg, joint] = kp
        self.kd[leg, joint] = kd
        self.tor[leg, joint] = command.tor
        if self._is_bm_joint(leg, joint):
            payload = pack_bm_command(command.pos, command.vel, kp, kd, command.tor)
        else:
            payload = pack_mit_command(command.pos, command.vel, kp, kd, command.tor)
        self.can.send(addr.serial, addr.channel, addr.motor_id, payload)

    def send_joint_position(self, leg: int, joint: int, pos: float, kp: float = 70.0, kd: float = 1.4) -> None:
        self.send_mit(leg, joint, MitCommand(pos=pos, vel=0.0, kp=kp, kd=kd, tor=0.0))

    def send_all_for_joint_id(
        self,
        joint: int,
        pos_by_leg: Sequence[float],
        kp: float = 70.0,
        kd: float = 1.4,
        tx_interval: float = 0.0,
    ) -> None:
        for leg in range(4):
            self.send_joint_position(leg, joint, float(pos_by_leg[leg]), kp=kp, kd=kd)
            if tx_interval > 0.0:
                time.sleep(tx_interval)

    def send_zero_query_all(self) -> None:
        for joint in range(3):
            for leg in range(4):
                self.send_mit(leg, joint, MitCommand())
                time.sleep(0.004)
                self.poll()

    @contextmanager
    def bm_calf_gain_override(self, kp: float, kd: float) -> Iterator[None]:
        previous = (self._bm_calf_kp, self._bm_calf_kd)
        self._bm_calf_kp = float(kp)
        self._bm_calf_kd = float(kd)
        try:
            yield
        finally:
            self._bm_calf_kp, self._bm_calf_kd = previous

    def switch_mit_mode_all(self) -> None:
        for addr in self.addresses:
            if self._is_bm_joint(addr.leg, addr.joint):
                continue
            payload = [addr.motor_id & 0xFF, (addr.motor_id >> 8) & 0xFF, 0x55, 0x0A, 0x01, 0x00, 0x00, 0x00]
            for _ in range(5):
                self.can.send(addr.serial, addr.channel, 0x7FF, payload)
                time.sleep(0.004)
        self.poll(limit=4096)

    def enable_all(self) -> None:
        for joint in range(3):
            for leg in range(4):
                addr = self._addr(leg, joint)
                payload = BM_FOC_PAYLOAD if self._is_bm_joint(leg, joint) else ENABLE_PAYLOAD
                self.can.send(addr.serial, addr.channel, addr.motor_id, payload)
                time.sleep(0.004)

    def disable_all(self) -> None:
        for addr in self.addresses:
            if self._is_bm_joint(addr.leg, addr.joint):
                self.can.send(addr.serial, addr.channel, addr.motor_id, pack_bm_command(0.0, 0.0, 0.0, 0.0, 0.0))
                time.sleep(0.002)
                self.can.send(addr.serial, addr.channel, addr.motor_id, BM_REST_PAYLOAD)
            else:
                self.can.send(addr.serial, addr.channel, addr.motor_id, DISABLE_PAYLOAD)
            time.sleep(0.002)

    def clear_errors_all(self) -> None:
        for addr in self.addresses:
            if self._is_bm_joint(addr.leg, addr.joint):
                continue
            self.can.send(addr.serial, addr.channel, addr.motor_id, CLEAR_ERROR_PAYLOAD)
            time.sleep(0.002)

    def poll(self, limit: int = 512) -> int:
        count = 0
        for frame in self.can.drain(limit=limit):
            if frame.is_tx_echo:
                continue
            bus = self._bus_from_frame(frame.serial, frame.channel)
            if bus is None:
                continue
            motor_id = self._feedback_motor_id(frame.data, frame.can_id)
            if motor_id not in (1, 2, 3):
                continue
            leg = bus
            joint = motor_id - 1
            if self._is_bm_joint(leg, joint):
                bm_fb = unpack_bm_feedback(frame.data, source_can_id=frame.can_id)
                if bm_fb is None:
                    continue
                fb = MotorFeedback(
                    motor_id=motor_id,
                    state=bm_fb.state,
                    pos=bm_fb.position,
                    vel=bm_fb.velocity,
                    tor=bm_fb.torque,
                    seen=True,
                )
            else:
                fb = unpack_feedback(frame.data)
                if fb is None:
                    continue
            prev = self.feedback[(leg, joint)]
            if prev.seen and abs(fb.pos - prev.pos) > MAX_FEEDBACK_POSITION_JUMP:
                self.rejected_feedback += 1
                self.rejected_feedback_id = fb.motor_id
                continue
            self.feedback[(leg, joint)] = fb
            count += 1
        return count

    def wait_for_feedback(self, timeout: float = 1.0, require_all: bool = True) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.poll()
            if not require_all:
                return any(fb.seen for fb in self.feedback.values())
            if all(fb.seen for fb in self.feedback.values()):
                return True
            time.sleep(0.002)
        self.poll()
        return all(fb.seen for fb in self.feedback.values()) if require_all else any(fb.seen for fb in self.feedback.values())

    def clear_feedback_history(self) -> None:
        for key, fb in self.feedback.items():
            self.feedback[key] = MotorFeedback(motor_id=fb.motor_id)

    def current_positions(self) -> np.ndarray:
        pos = np.zeros((4, 3), dtype=np.float32)
        for leg in range(4):
            for joint in range(3):
                pos[leg, joint] = self.feedback[(leg, joint)].pos
        return pos

    def current_velocities(self) -> np.ndarray:
        vel = np.zeros((4, 3), dtype=np.float32)
        for leg in range(4):
            for joint in range(3):
                vel[leg, joint] = self.feedback[(leg, joint)].vel
        return vel

    def missing_feedback(self) -> List[Tuple[int, int]]:
        return [(leg, joint) for leg in range(4) for joint in range(3) if not self.feedback[(leg, joint)].seen]

    def _addr(self, leg: int, joint: int) -> JointAddress:
        return self.addresses[leg * 3 + joint]

    def _bus_from_frame(self, serial: str, channel: int) -> Optional[int]:
        for idx, item in enumerate(self.bus_map):
            if item[0] == serial and item[1] == channel:
                return idx
        return None

    def _is_bm_joint(self, leg: int, joint: int) -> bool:
        return 0 <= leg < 4 and joint == BM_CALF_JOINT

    def _feedback_motor_id(self, data: Iterable[int], source_can_id: int) -> int:
        payload = bytes(data)
        if not payload:
            return int(source_can_id)
        motor_id = payload[0] & 0x0F
        return motor_id if motor_id != 0 else int(source_can_id)


def _make_joint_addresses(bus_map: Sequence[Tuple[str, int]]) -> List[JointAddress]:
    addresses: List[JointAddress] = []
    for leg, (serial, channel) in enumerate(bus_map):
        for joint in range(3):
            addresses.append(
                JointAddress(
                    leg=leg,
                    joint=joint,
                    bus=leg,
                    motor_id=joint + 1,
                    serial=serial,
                    channel=channel,
                )
            )
    return addresses
