"""Stall / overload guard.

Why this exists
---------------
A jammed or unresponsive joint keeps trying to reach its commanded position, so its
current (and the reported torque) climbs. On this robot that has already pulled
enough current to trip the whole power supply: every motor went silent, the dog lost
all torque mid-motion, leaned and rolled over, and the policy node died with a
"motor feedback timeout".

This guard watches two independent, cheap signals per joint:

  * position error   - |commanded position - measured position|
  * reported torque  - |tor| from the MIT feedback frame

and it never sends anything itself. It has two modes:

  MONITOR (``MOTOR_GUARD_ABORT = False``, the default)
      Prints a warning when a warning threshold is crossed and keeps the running
      maxima per joint. Run the dog normally in this mode and read the printed
      summary to learn what "normal" looks like.

  ABORT (``MOTOR_GUARD_ABORT = True``)
      Additionally raises ``StallDetected`` once a joint stays past an abort
      threshold continuously for ``MOTOR_GUARD_ABORT_SECONDS``, so the caller can
      disable the motors before the supply trips.

Only enable ABORT after checking that the maxima printed in MONITOR mode stay well
below the abort thresholds.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

LEG_NAMES = ("FL", "FR", "RL", "RR")
JOINT_NAMES = ("hip", "thigh", "calf")


class StallDetected(RuntimeError):
    """Raised when a joint is considered stalled and MOTOR_GUARD_ABORT is enabled."""


@dataclass
class JointExtreme:
    max_pos_error: float = 0.0
    max_pos_error_at: str = "-"
    max_torque: float = 0.0
    max_torque_at: str = "-"
    warn_pos_error_count: int = 0
    warn_torque_count: int = 0


@dataclass
class _Violation:
    since: Optional[float] = None


class MotorGuard:
    def __init__(
        self,
        motors,
        *,
        abort_enabled: bool = False,
        warn_pos_error: float = 0.6,
        warn_torque: float = 4.0,
        abort_pos_error: float = 1.2,
        abort_torque: float = 8.0,
        abort_seconds: float = 0.3,
    ) -> None:
        self.motors = motors
        self.abort_enabled = bool(abort_enabled)
        self.warn_pos_error = float(warn_pos_error)
        self.warn_torque = float(warn_torque)
        self.abort_pos_error = float(abort_pos_error)
        self.abort_torque = float(abort_torque)
        self.abort_seconds = float(abort_seconds)
        self.extremes: Dict[Tuple[int, int], JointExtreme] = {}
        self._violation: Dict[Tuple[int, int], _Violation] = {}
        self._last_warn = 0.0
        self._samples = 0
        self._started = time.monotonic()

    def update(self, label: str = "") -> None:
        """Sample the bus once. Cheap; safe to call every control cycle."""
        motors = self.motors
        if motors is None:
            return
        now = time.monotonic()
        try:
            cmd = np.asarray(motors.commands, dtype=np.float32).reshape(4, 3)
        except Exception:
            return
        self._samples += 1

        offenders = []
        for leg in range(4):
            for j in range(3):
                fb = motors.feedback.get((leg, j))
                if fb is None or not fb.seen:
                    continue
                err = abs(float(cmd[leg, j]) - float(fb.pos))
                tor = abs(float(fb.tor))
                ex = self.extremes.setdefault((leg, j), JointExtreme())
                if err > ex.max_pos_error:
                    ex.max_pos_error = err
                    ex.max_pos_error_at = label
                if tor > ex.max_torque:
                    ex.max_torque = tor
                    ex.max_torque_at = label

                bad = False
                if err > self.warn_pos_error:
                    bad = True
                    ex.warn_pos_error_count += 1
                if tor > self.warn_torque:
                    bad = True
                    ex.warn_torque_count += 1
                if bad:
                    offenders.append("%s/%s err=%.2f tor=%.2f"
                                     % (LEG_NAMES[leg], JOINT_NAMES[j], err, tor))

                # abort bookkeeping
                if err > self.abort_pos_error or tor > self.abort_torque:
                    v = self._violation.setdefault((leg, j), _Violation())
                    if v.since is None:
                        v.since = now
                        print("MOTOR GUARD: %s/%s past the abort threshold at %s "
                              "(err=%.2f tor=%.2f) - timing how long it lasts"
                              % (LEG_NAMES[leg], JOINT_NAMES[j], label or "-", err, tor),
                              flush=True)
                    elif self.abort_enabled and (now - v.since) >= self.abort_seconds:
                        raise StallDetected(
                            "%s/%s stalled for %.2fs at %s (pos_error=%.2f rad, torque=%.2f Nm)"
                            % (LEG_NAMES[leg], JOINT_NAMES[j], now - v.since,
                               label or "-", err, tor))
                else:
                    v = self._violation.get((leg, j))
                    if v is not None and v.since is not None:
                        print("MOTOR GUARD: %s/%s back to normal after %.2fs"
                              % (LEG_NAMES[leg], JOINT_NAMES[j], now - v.since), flush=True)
                        v.since = None

        if offenders and (now - self._last_warn) > 1.0:
            self._last_warn = now
            print("MOTOR GUARD warning at %-22s : %s" % (label or "-", "; ".join(offenders)),
                  flush=True)

    def report(self) -> None:
        print()
        print("=" * 78)
        print("MOTOR GUARD SUMMARY  (mode: %s)" % ("ABORT" if self.abort_enabled else "MONITOR ONLY"))
        print("  samples=%d  over %.1fs" % (self._samples, time.monotonic() - self._started))
        print("  warn thresholds: pos_error>%.2f rad, |torque|>%.2f Nm"
              % (self.warn_pos_error, self.warn_torque))
        print("  abort thresholds: pos_error>%.2f rad, |torque|>%.2f Nm for %.2fs"
              % (self.abort_pos_error, self.abort_torque, self.abort_seconds))
        print()
        print("  %-5s %-6s %12s %-14s %12s %-14s" % ("leg", "joint", "max_perr", "where", "max|tor|", "where"))
        for leg in range(4):
            for j in range(3):
                ex = self.extremes.get((leg, j))
                if ex is None:
                    continue
                print("  %-5s %-6s %12.3f %-14s %12.3f %-14s"
                      % (LEG_NAMES[leg], JOINT_NAMES[j], ex.max_pos_error,
                         ex.max_pos_error_at or "-", ex.max_torque, ex.max_torque_at or "-"))
        if self.extremes:
            worst_err = max(e.max_pos_error for e in self.extremes.values())
            worst_tor = max(e.max_torque for e in self.extremes.values())
            print()
            print("  worst position error over the whole run: %.3f rad" % worst_err)
            print("  worst |torque| over the whole run      : %.3f Nm" % worst_tor)
            if not self.abort_enabled:
                print()
                print("  MONITOR ONLY - nothing was aborted. Use these maxima to decide")
                print("  MOTOR_GUARD_ABORT_POS_ERROR_RAD / MOTOR_GUARD_ABORT_TORQUE_NM,")
                print("  then set MOTOR_GUARD_ABORT = True in bs_hardware/config.py.")
        print("=" * 78, flush=True)


_GUARD: Optional[MotorGuard] = None


def get_guard(motors) -> MotorGuard:
    """One guard per process, created lazily and shared by startup and the RL loop."""
    global _GUARD
    if _GUARD is None or _GUARD.motors is not motors:
        from .config import (MOTOR_GUARD_ABORT, MOTOR_GUARD_ABORT_POS_ERROR_RAD,
                             MOTOR_GUARD_ABORT_SECONDS, MOTOR_GUARD_ABORT_TORQUE_NM,
                             MOTOR_GUARD_WARN_POS_ERROR_RAD, MOTOR_GUARD_WARN_TORQUE_NM)
        _GUARD = MotorGuard(
            motors,
            abort_enabled=MOTOR_GUARD_ABORT,
            warn_pos_error=MOTOR_GUARD_WARN_POS_ERROR_RAD,
            warn_torque=MOTOR_GUARD_WARN_TORQUE_NM,
            abort_pos_error=MOTOR_GUARD_ABORT_POS_ERROR_RAD,
            abort_torque=MOTOR_GUARD_ABORT_TORQUE_NM,
            abort_seconds=MOTOR_GUARD_ABORT_SECONDS,
        )
    return _GUARD
