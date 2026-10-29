from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .config import (
    BM_CALF_STARTUP_KD,
    BM_CALF_STARTUP_KP,
    MOTOR_INIT_SETTLE_SECONDS,
    STARTUP_JOINT_DIRECTIONS,
)
from .leg_ik import TandemLeg
from .motor_bus import MotorBus
from .motor_guard import get_guard

STARTUP_JOINT_DIRECTIONS_ARRAY = np.asarray(STARTUP_JOINT_DIRECTIONS, dtype=np.float32)
HIP_SPREAD_STEPS = 139
HIP_SPREAD_STEP_RAD = 0.001
LOWER_BODY_STEPS = 2200

# Per-motor startup output offsets in degrees.
# Leg order: FL, FR, RL, RR. Joint order: hip, thigh, calf.
# Positive values are added directly to the raw MIT motor position command.
STARTUP_OUTPUT_OFFSETS_DEG = [
    [0.0, 0.0, 0.0],  # FL
    [0.0, -0.0, -0.0],  # FR
    [0.0, -0.0, 0.0],  # RL
    [0.0, 0.0, 0.0],  # RR
]

# STARTUP_OUTPUT_OFFSETS_DEG = [
#     [0.0, 5.0, 5.0],  # FL
#     [0.0, -5.0, -0.0],  # FR
#     [0.0, -3.0, 0.0],  # RL
#     [0.0, 3.0, 0.0],  # RR
# ]

STARTUP_OUTPUT_OFFSETS_RAD = np.deg2rad(np.asarray(STARTUP_OUTPUT_OFFSETS_DEG, dtype=np.float32))

RL_CALF_REDUCTION_RAD = float(np.deg2rad(0.0))
RR_CALF_EXTRA_RAD = float(np.deg2rad(0.0))


@dataclass
class StartupResult:
    motor_offsets: np.ndarray
    startup_pose: np.ndarray
    initial_pose: np.ndarray


class DogStartupController:
    def __init__(
        self,
        motors: MotorBus,
        *,
        kp: float = 70.0,
        kd: float = 1.4,
        speed_scale: float = 1.0,
        require_feedback: bool = True,
        feedback_timeout: float = 2.0,
        tx_interval: float = 0.0,
        emergency_checker: Optional[Callable[[], bool]] = None,
        motor_init_settle: Optional[float] = None,
    ) -> None:
        self.motors = motors
        self.kp = float(kp)
        self.kd = float(kd)
        self.speed_scale = max(0.05, float(speed_scale))
        self.require_feedback = bool(require_feedback)
        self.feedback_timeout = max(0.1, float(feedback_timeout))
        self.tx_interval = max(0.0, float(tx_interval))
        self.emergency_checker = emergency_checker
        self.motor_init_settle = max(
            0.0,
            float(MOTOR_INIT_SETTLE_SECONDS if motor_init_settle is None else motor_init_settle),
        )
        self.motor_py = np.zeros((4, 3), dtype=np.float32)
        self.mit_mo_set = np.zeros((4, 3), dtype=np.float32)
        self.initial_pose = np.zeros((4, 3), dtype=np.float32)
        self.legs = [TandemLeg() for _ in range(4)]
        for leg in self.legs:
            leg.initialize_offsets()
        self._send_joint = 0
        self._record_startup_frames = False
        self._startup_frames: List[Tuple[int, np.ndarray]] = []

    def run(self) -> StartupResult:
        with self.motors.bm_calf_gain_override(BM_CALF_STARTUP_KP, BM_CALF_STARTUP_KD):
            self._check_emergency()
            self._wait_for_motor_init()
            self._read_initial_positions()
            self._check_emergency()
            self.motors.enable_all()
            time.sleep(0.001 / self.speed_scale)
            self._check_emergency()

            self._startup_frames.clear()
            self._record_startup_frames = True
            try:
                self._hip_spread()
                self._thigh_adjust()
                self._lower_body_with_ik()
            finally:
                self._record_startup_frames = False

        pose = self.target_positions.copy()
        return StartupResult(motor_offsets=self.motor_py.copy(), startup_pose=pose, initial_pose=self.initial_pose.copy())

    @property
    def target_positions(self) -> np.ndarray:
        return self.motor_py + self.mit_mo_set + STARTUP_OUTPUT_OFFSETS_RAD

    def hold_current_pose(self, duration: float = 0.5) -> None:
        with self.motors.bm_calf_gain_override(BM_CALF_STARTUP_KP, BM_CALF_STARTUP_KD):
            deadline = time.monotonic() + duration
            while time.monotonic() < deadline:
                self._check_emergency()
                self.flash()
                time.sleep(0.001 / self.speed_scale)

    def lie_down_reverse(self, hold_duration: float = 0.25) -> None:
        with self.motors.bm_calf_gain_override(BM_CALF_STARTUP_KP, BM_CALF_STARTUP_KD):
            self._check_emergency()
            if not self._startup_frames:
                self._interpolate_to_pose(self.initial_pose, steps=400)
            else:
                self._interpolate_from_feedback_to_pose(self.target_positions, steps=250)
                for joint, targets in reversed(self._startup_frames):
                    self._check_emergency()
                    self.motors.send_all_for_joint_id(joint, targets, kp=self.kp, kd=self.kd, tx_interval=self.tx_interval)
                    self.motors.poll()
                    time.sleep(0.001 / self.speed_scale)

            self.motor_py = self.initial_pose.copy()
            self.mit_mo_set[:, :] = 0.0
            if hold_duration > 0.0:
                self._hold_pose(self.initial_pose, hold_duration)

    def flash(self) -> None:
        self._check_emergency()
        joint = self._send_joint
        targets = self.target_positions[:, joint]
        self.motors.send_all_for_joint_id(joint, targets, kp=self.kp, kd=self.kd, tx_interval=self.tx_interval)
        if self._record_startup_frames:
            self._startup_frames.append((joint, np.asarray(targets, dtype=np.float32).copy()))
        self._send_joint = (self._send_joint + 1) % 3
        self.motors.poll()
        # Stall/overload watch (warning only unless MOTOR_GUARD_ABORT is enabled).
        get_guard(self.motors).update("startup")

    def _wait_for_motor_init(self) -> None:
        """Wait for the motor encoders to initialise before sending anything.

        The DM motors need a few seconds after the power is switched on before their
        encoder reading is valid. During that window NOTHING is transmitted (not even
        a zero-torque query), because the startup trajectory is built from the initial
        position reading - reading a bogus position would make the dog jump as soon as
        torque is enabled.
        """
        delay = max(0.0, float(self.motor_init_settle))
        if delay <= 0.0:
            return
        print(
            "waiting %.1fs for the motor encoders to initialise before sending anything..."
            % delay,
            flush=True,
        )
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            self._check_emergency()
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        print("motor encoder init wait done (%.1fs)" % delay, flush=True)

    def _read_initial_positions(self) -> None:
        self._check_emergency()
        self.motors.send_zero_query_all()
        got_feedback = self.motors.wait_for_feedback(timeout=self.feedback_timeout, require_all=self.require_feedback)
        if self.require_feedback and not got_feedback:
            raise RuntimeError("missing motor feedback before startup: %s" % (self.motors.missing_feedback(),))
        self.motor_py = self.motors.current_positions()
        self.initial_pose = self.motor_py.copy()
        self.motors.commands[:, :] = self.motor_py
        self.motors.vel[:, :] = 0.0
        self.motors.kp[:, :] = self.kp
        self.motors.kd[:, :] = self.kd
        self.motors.tor[:, :] = 0.0

    def _hip_spread(self) -> None:
        for _ in range(HIP_SPREAD_STEPS):
            self._check_emergency()
            logical_delta = np.zeros((4, 3), dtype=np.float32)
            logical_delta[0, 0] = -HIP_SPREAD_STEP_RAD
            logical_delta[1, 0] = HIP_SPREAD_STEP_RAD
            logical_delta[2, 0] = HIP_SPREAD_STEP_RAD
            logical_delta[3, 0] = -HIP_SPREAD_STEP_RAD
            self.motor_py += logical_delta * STARTUP_JOINT_DIRECTIONS_ARRAY
            time.sleep(0.001 / self.speed_scale)
            self.flash()

    def _thigh_adjust(self) -> None:
        for _ in range(940):
            self._check_emergency()
            logical_delta = np.zeros((4, 3), dtype=np.float32)
            logical_delta[0, 1] = 0.001
            logical_delta[1, 1] = 0.001
            logical_delta[2, 1] = -0.001
            logical_delta[3, 1] = -0.001
            self.motor_py += logical_delta * STARTUP_JOINT_DIRECTIONS_ARRAY
            time.sleep(0.001 / self.speed_scale)
            self.flash()

    def _lower_body_with_ik(self) -> None:
        for step in range(LOWER_BODY_STEPS):
            self._check_emergency()
            for leg in self.legs:
                leg.z -= 0.0001

            for i, leg in enumerate(self.legs):
                theta1, theta2, theta3 = leg.ik_delta(side="L")
                self.mit_mo_set[i, 0] = theta1
                if i < 2:
                    self.mit_mo_set[i, 1] = theta2
                    self.mit_mo_set[i, 2] = theta3
                else:
                    self.mit_mo_set[i, 1] = -theta2
                    self.mit_mo_set[i, 2] = -theta3
            self.mit_mo_set *= STARTUP_JOINT_DIRECTIONS_ARRAY
            reduction_ratio = float(step + 1) / float(LOWER_BODY_STEPS)
            self.mit_mo_set[2, 2] -= RL_CALF_REDUCTION_RAD * reduction_ratio
            self.mit_mo_set[3, 2] += RR_CALF_EXTRA_RAD * reduction_ratio

            self.flash()
            time.sleep(0.001 / self.speed_scale)

    def _interpolate_to_pose(self, target_pose: np.ndarray, steps: int = 400) -> None:
        current = self.target_positions.copy()
        target = np.asarray(target_pose, dtype=np.float32).reshape(4, 3)
        steps = max(1, int(steps))
        for i in range(steps):
            self._check_emergency()
            alpha = float(i + 1) / float(steps)
            pose = current * (1.0 - alpha) + target * alpha
            self._send_pose(pose)
            time.sleep(0.001 / self.speed_scale)

    def _interpolate_from_feedback_to_pose(self, target_pose: np.ndarray, steps: int = 250) -> None:
        self.motors.poll(limit=4096)
        current = self.motors.current_positions()
        if self.require_feedback and self.motors.missing_feedback():
            current = self.target_positions.copy()
        target = np.asarray(target_pose, dtype=np.float32).reshape(4, 3)
        steps = max(1, int(steps))
        for i in range(steps):
            self._check_emergency()
            alpha = float(i + 1) / float(steps)
            pose = current * (1.0 - alpha) + target * alpha
            self._send_pose(pose)
            time.sleep(0.001 / self.speed_scale)

    def _hold_pose(self, pose: np.ndarray, duration: float) -> None:
        deadline = time.monotonic() + max(0.0, float(duration))
        while time.monotonic() < deadline:
            self._check_emergency()
            self._send_pose(pose)
            time.sleep(0.001 / self.speed_scale)

    def _send_pose(self, pose: np.ndarray) -> None:
        self._check_emergency()
        pose = np.asarray(pose, dtype=np.float32).reshape(4, 3)
        for joint in range(3):
            self._check_emergency()
            self.motors.send_all_for_joint_id(joint, pose[:, joint], kp=self.kp, kd=self.kd, tx_interval=self.tx_interval)
            self.motors.poll()
        get_guard(self.motors).update("startup-pose")

    def _check_emergency(self) -> None:
        if self.emergency_checker is not None and self.emergency_checker():
            self.motors.disable_all()
            raise RuntimeError("emergency stop requested")
