from __future__ import annotations

from typing import Callable, Optional

from .motor_bus import MotorBus
from .startup import DogStartupController, StartupResult


class DogPostureActionLibrary:
    def __init__(
        self,
        motors: MotorBus,
        *,
        kp: float,
        kd: float,
        speed_scale: float,
        require_feedback: bool,
        feedback_timeout: float,
        switch_mit_mode: bool,
        emergency_checker: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.motors = motors
        self.kp = float(kp)
        self.kd = float(kd)
        self.speed_scale = float(speed_scale)
        self.require_feedback = bool(require_feedback)
        self.feedback_timeout = float(feedback_timeout)
        self.switch_mit_mode = bool(switch_mit_mode)
        self.emergency_checker = emergency_checker
        self.startup_controller: Optional[DogStartupController] = None
        self.startup_result: Optional[StartupResult] = None
        self.is_standing = False
        self._mit_mode_switched = False

    def stand_up(self) -> StartupResult:
        if self.is_standing and self.startup_result is not None:
            return self.startup_result
        if self.switch_mit_mode and not self._mit_mode_switched:
            self.motors.switch_mit_mode_all()
            self._mit_mode_switched = True

        startup = DogStartupController(
            self.motors,
            kp=self.kp,
            kd=self.kd,
            speed_scale=self.speed_scale,
            require_feedback=self.require_feedback,
            feedback_timeout=self.feedback_timeout,
            emergency_checker=self.emergency_checker,
        )
        self.startup_controller = startup
        self.startup_result = startup.run()
        self.startup_result.startup_pose[:] = startup.target_positions
        self.is_standing = True
        return self.startup_result

    def lie_down(self, hold_duration: float = 0.25) -> None:
        if not self.is_standing:
            return
        if self.startup_controller is not None:
            self.startup_controller.lie_down_reverse(hold_duration=hold_duration)
        self.startup_controller = None
        self.startup_result = None
        self.is_standing = False
