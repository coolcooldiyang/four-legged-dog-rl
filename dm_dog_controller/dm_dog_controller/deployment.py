from __future__ import annotations

import argparse
import errno
import glob
import math
import os
import re
import select
import struct
import sys
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path
from typing import Callable, Deque, Dict, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch

from bs_hardware.can_hub import Usb2CanHub
from bs_hardware.action_library import DogPostureActionLibrary
from bs_hardware.config import (
    ACTION_DIM,
    ANGULAR_VELOCITY_SCALE,
    CAN_BITRATE,
    CAN_BUS_MAP,
    CAN_CHANNELS,
    CAN_SERIALS,
    CANFD_ENABLED,
    COMMAND_SCALE,
    DEFAULT_COMMAND,
    DEFAULT_DOF_POS,
    DOF_VELOCITY_SCALE,
    FIXED_RL_ORIGIN_POSE_RAW,
    HISTORY_LENGTH,
    IMU_BAUDRATE,
    IMU_PORT,
    MODEL_JOINT_DIRECTIONS,
    MODEL_JOINT_PERMUTATION,
    OBS_DIM,
    OBS_DIM_DEPLOYABLE,
    OBS_HEAD_ORDER,
    OBS_PHASE_PERIOD_S,
    RL_ACTION_SCALE,
    STARTUP_KD,
    STARTUP_KP,
    TRANSLATION_POLICY_BM_CALF_KD,
    TRANSLATION_POLICY_BM_CALF_KP,
    TRANSLATION_POLICY_RL_KD,
    TRANSLATION_POLICY_RL_KP,
    YAW_POLICY_BM_CALF_KD,
    YAW_POLICY_BM_CALF_KP,
    YAW_POLICY_RL_KD,
    YAW_POLICY_RL_KP,
    resolve_observation_spec,
)
from bs_hardware.imu_l1 import ImuReading, ImuSensor, euler_deg_to_projected_gravity
from bs_hardware.mit_motor import MitCommand
from bs_hardware.motor_bus import MotorBus
from bs_hardware.motor_guard import StallDetected, get_guard
from bs_hardware.startup import DogStartupController, StartupResult


MODEL_JOINT_DIRECTIONS_ARRAY = np.asarray(MODEL_JOINT_DIRECTIONS, dtype=np.float32)
MODEL_JOINT_DIRECTIONS_FLAT = MODEL_JOINT_DIRECTIONS_ARRAY.reshape(ACTION_DIM)
#: 策略关节顺序 → dm 顺序的置换索引（见 config.py 的 MODEL_JOINT_PERMUTATION）
MODEL_JOINT_PERMUTATION_ARRAY = np.asarray(MODEL_JOINT_PERMUTATION, dtype=np.int64)
#: 逆置换（对合时与正置换相同，这里显式计算保证一般性）
MODEL_JOINT_INV_PERMUTATION_ARRAY = np.argsort(MODEL_JOINT_PERMUTATION_ARRAY)
DEFAULT_DOF_POS_ARRAY = np.asarray(DEFAULT_DOF_POS, dtype=np.float32)
FIXED_RL_ORIGIN_POSE_RAW_ARRAY = np.asarray(FIXED_RL_ORIGIN_POSE_RAW, dtype=np.float32)


def _parse_head_order(value: str) -> Optional[list]:
    """把 config 里的 "ang_vel,gravity,cmd" 解析成列表；空串或默认顺序返回 None。"""
    if not value:
        return None
    names = [item.strip() for item in str(value).split(",") if item.strip()]
    if not names:
        return None
    if names == ["ang_vel", "gravity", "cmd"]:
        return None            # 契约默认顺序，走老路径
    return names


class PolicyDeployer:
    def __init__(self, model_path: str, *, name: str = "policy", head_order: Optional[str] = None):
        self.name = str(name)
        self.model_path = str(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = torch.jit.load(model_path, map_location=self.device)
        self.policy.eval()
        self.num_obs_single = OBS_DIM
        self.action_method_name, self._action_fn = self._resolve_action_method()
        self.total_input_dim = self._infer_total_input_dim()
        spec = resolve_observation_spec(self.total_input_dim)
        if spec is None:
            raise ValueError(
                "policy input dim %d matches no known observation layout (legacy %d per frame, "
                "deployable %d per frame); see bs_hardware/config.py"
                % (self.total_input_dim, OBS_DIM, OBS_DIM_DEPLOYABLE)
            )
        self.per_frame, self.history_length, self.has_phase, self.observation_layout_name = spec
        self.num_obs_single = self.per_frame
        parsed_head_order = _parse_head_order(head_order if head_order else OBS_HEAD_ORDER)
        self.head_order = tuple(parsed_head_order) if parsed_head_order else None
        # The traced command slice exists only in the legacy exports. Treat it as a
        # cross-check, never as the primary source of truth: rsl_rl's plain actor
        # export has no slice, and the previous fallback then assumed "command
        # first" - the exact opposite of how both layouts are actually trained.
        command_slice = self._infer_command_slice()
        self.traced_layout_name = None
        if command_slice is not None:
            inferred_offset, inferred_newest = self._resolve_observation_format(command_slice)
            if inferred_offset != 6:
                print(
                    "%s: WARNING traced command slice %s implies command offset %d (command-first layout); "
                    "the deployable layout builder only supports offset 6"
                    % (self.name, command_slice, inferred_offset),
                    flush=True,
                )
            self.command_offset, self.newest_first = inferred_offset, inferred_newest
        else:
            self.command_offset, self.newest_first = 6, True
        if self.has_phase and self.command_offset != 6 and self.head_order is None:
            raise ValueError(
                "deployable layout (per_frame=%d) requires the command at offset 6, got %d"
                % (self.per_frame, self.command_offset)
            )
        self._validate_inference()
        print(
            "loaded %s: path=%s entry=%s input_dim=%d per_frame=%d history=%d has_phase=%s obs_layout=%s "
            "traced_layout=%s command_offset=%d head_order=%s history_order=%s"
            % (
                self.name,
                self.model_path,
                self.action_method_name,
                self.total_input_dim,
                self.per_frame,
                self.history_length,
                self.has_phase,
                self.observation_layout_name,
                self.traced_layout_name,
                self.command_offset,
                ",".join(self.head_order) if self.head_order else "ang_vel,gravity,cmd",
                "newest_first" if self.newest_first else "oldest_first",
            ),
            flush=True,
        )

    def _resolve_action_method(self) -> Tuple[str, Callable[[torch.Tensor], torch.Tensor]]:
        method_names = set()
        try:
            method_names = set(self.policy._c._method_names())
        except Exception:
            pass
        if "forward" in method_names or hasattr(self.policy, "forward"):
            return "forward", self.policy
        if "act_inference" in method_names or hasattr(self.policy, "act_inference"):
            return "act_inference", getattr(self.policy, "act_inference")
        raise AttributeError(
            "policy has no callable action entry; exported TorchScript methods=%s"
            % (sorted(method_names),)
        )

    def _candidate_input_dims(self) -> list:
        """Every plausible flat observation width, smallest first.

        We probe the network instead of trusting the weights: on the legacy policy
        the actor's first Linear is (256, 38) because it consumes
        cat([command, STMEnc(obs), state_estimator(obs)]) - the observation width is
        not visible in any single weight shape.
        """
        dims = []
        for per_frame in (OBS_DIM, OBS_DIM_DEPLOYABLE):
            for history in range(1, 9):
                dims.append(per_frame * history)
        return sorted(set(dims))

    def _infer_total_input_dim(self) -> int:
        problems = []
        for dim in self._candidate_input_dims():
            if resolve_observation_spec(dim) is None:
                continue
            try:
                with torch.no_grad():
                    self._action_fn(torch.zeros(1, dim, dtype=torch.float32, device=self.device))
            except Exception as exc:
                problems.append((dim, type(exc).__name__))
                continue
            return dim
        # Nothing probed successfully: fall back to the widest 2-D weight that looks
        # like an observation width, then to the legacy default.
        candidates = []
        for name, value in self.policy.state_dict().items():
            if value.ndim != 2:
                continue
            input_dim = int(value.shape[1])
            if input_dim >= OBS_DIM and (input_dim % OBS_DIM == 0 or input_dim % OBS_DIM_DEPLOYABLE == 0):
                candidates.append((name, input_dim))
        if candidates:
            preferred_names = ("STMEnc.0.weight", "actor.0.weight", "0.weight")
            for preferred in preferred_names:
                for name, input_dim in candidates:
                    if name.endswith(preferred):
                        return input_dim
            return max(input_dim for _, input_dim in candidates)
        raise ValueError(
            "could not determine the policy observation width; probed widths all failed: %s" % (problems[-6:],)
        )

    def _infer_command_slice(self) -> Optional[Tuple[int, int]]:
        try:
            code = getattr(getattr(self.policy, self.action_method_name), "code")
        except Exception:
            return None
        match = re.search(r"command = torch\.slice\([^,]+,\s*1,\s*(\d+),\s*(\d+)\)", code)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    def _resolve_observation_format(self, command_slice: Optional[Tuple[int, int]]) -> Tuple[int, bool]:
        if command_slice is None:
            self.traced_layout_name = "command/base_ang_vel/projected_gravity"
            return 0, True
        start, end = command_slice
        if end - start != 3:
            raise ValueError("policy command slice must be length 3, got %s" % (command_slice,))
        if not (0 <= start < self.total_input_dim and end <= self.total_input_dim):
            raise ValueError("policy command slice %s is outside input dim %d" % (command_slice, self.total_input_dim))
        frame_index = start // self.per_frame
        command_offset = start % self.per_frame
        if command_offset == 0:
            self.traced_layout_name = "command/base_ang_vel/projected_gravity"
        elif command_offset == 6:
            self.traced_layout_name = "base_ang_vel/projected_gravity/command"
        else:
            raise ValueError(
                "unsupported policy command offset %d inferred from slice %s"
                % (command_offset, command_slice)
            )
        if frame_index == 0:
            newest_first = True
        elif frame_index == self.history_length - 1:
            newest_first = False
        else:
            raise ValueError(
                "unsupported policy command frame index %d for history length %d"
                % (frame_index, self.history_length)
            )
        return command_offset, newest_first

    def _validate_inference(self) -> None:
        with torch.no_grad():
            obs_tensor = torch.zeros(1, self.total_input_dim, dtype=torch.float32, device=self.device)
            action_tensor = self._action_fn(obs_tensor)
        action = action_tensor.detach().cpu().numpy().squeeze()
        if action.shape != (ACTION_DIM,):
            raise ValueError("policy action shape must be (%d,), got %s" % (ACTION_DIM, action.shape))

    def inference(self, obs_history_buffer: np.ndarray) -> np.ndarray:
        if obs_history_buffer.shape != (self.total_input_dim,):
            raise ValueError("obs history shape must be (%d,), got %s" % (self.total_input_dim, obs_history_buffer.shape))
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_history_buffer).float().unsqueeze(0).to(self.device)
            action_tensor = self._action_fn(obs_tensor)
        action = action_tensor.cpu().numpy().squeeze()
        if action.shape != (ACTION_DIM,):
            raise ValueError("policy action shape must be (%d,), got %s" % (ACTION_DIM, action.shape))
        return action

    def estimate_base_velocity(self, obs_history_buffer: np.ndarray) -> Optional[np.ndarray]:
        if obs_history_buffer.shape != (self.total_input_dim,):
            raise ValueError("obs history shape must be (%d,), got %s" % (self.total_input_dim, obs_history_buffer.shape))
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_history_buffer).float().unsqueeze(0).to(self.device)
            if hasattr(self.policy, "STMEnc") and hasattr(self.policy, "state_estimator"):
                activation = getattr(self.policy.state_estimator, "1", None)
                if activation is not None:
                    encoded = self.policy.STMEnc.forward(activation, obs_tensor)
                    vel = self.policy.state_estimator(encoded)
                    return vel.cpu().numpy().squeeze()
            if not hasattr(self.policy, "estimator"):
                return None
            parts = self.policy.estimator(obs_tensor)[:, 0:19]
            vel = parts[..., :3]
        return vel.cpu().numpy().squeeze()

    def _reset_policy_state(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if reset is None:
            return
        try:
            reset()
        except Exception:
            pass

    def reset_state(self) -> None:
        self._reset_policy_state()


class ObservationBuilder:
    def __init__(
        self,
        history_length: int = HISTORY_LENGTH,
        command_offset: int = 0,
        newest_first: bool = True,
        per_frame: int = OBS_DIM,
        has_phase: bool = False,
        phase_period: float = OBS_PHASE_PERIOD_S,
        head_order: Optional[Sequence[str]] = None,
    ) -> None:
        if history_length < 1:
            raise ValueError("history_length must be >= 1")
        self.head_order = tuple(head_order) if head_order else None
        if self.head_order is not None:
            if sorted(self.head_order) != ["ang_vel", "cmd", "gravity"]:
                raise ValueError(
                    "head_order 必须是 ang_vel/gravity/cmd 的排列，收到 %s" % (self.head_order,)
                )
            # 命令落在哪个偏移由顺序决定（0 / 3 / 6）
            command_offset = self.head_order.index("cmd") * 3
        if command_offset not in (0, 3, 6):
            raise ValueError("unsupported command offset %d" % command_offset)
        if has_phase and self.head_order is None and command_offset != 6:
            raise ValueError("the gait-phase layout requires the command at offset 6")
        if has_phase and self.head_order is not None and self.head_order[-1] != "cmd" and command_offset != 6:
            # 允许（相位与命令位置无关），但打印出来方便排查
            print(
                "note: gait-phase layout with command at offset %d (head_order=%s)"
                % (command_offset, ",".join(self.head_order)),
                flush=True,
            )
        self.history_length = int(history_length)
        self.command_offset = int(command_offset)
        self.newest_first = bool(newest_first)
        self.per_frame = int(per_frame)
        self.has_phase = bool(has_phase)
        self.phase_period = max(1e-3, float(phase_period))
        # Gait clock for the deployable (unitree_rl_gym style) layout. It is started
        # when the RL policy actually takes over (see DeploymentRunner.run), because
        # the real robot has no episode boundary to align the phase with.
        self.phase = 0.0
        self._phase_started_at: Optional[float] = None
        self.command = np.asarray(DEFAULT_COMMAND, dtype=np.float32)
        self.command_scale = np.asarray(COMMAND_SCALE, dtype=np.float32)
        self.default_dof_pos = np.asarray(DEFAULT_DOF_POS, dtype=np.float32)
        self.dof_pos = np.zeros(ACTION_DIM, dtype=np.float32)
        self.dof_vel = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last_action = np.zeros(ACTION_DIM, dtype=np.float32)
        self.history: Deque[np.ndarray] = deque(maxlen=self.history_length)
        for _ in range(self.history_length):
            self.history.append(np.zeros(self.per_frame, dtype=np.float32))

    # ------------------------------------------------------------------ phase
    def start_phase(self, now: Optional[float] = None) -> None:
        """Restart the gait clock from phase 0 (call when RL takes over)."""
        self._phase_started_at = time.monotonic() if now is None else float(now)
        self.phase = 0.0

    def stop_phase(self) -> None:
        self._phase_started_at = None

    def tick_phase(self, now: Optional[float] = None) -> float:
        """Advance the gait clock to the current time and return the phase in [0, 1)."""
        if not self.has_phase or self._phase_started_at is None:
            return self.phase
        current = time.monotonic() if now is None else float(now)
        elapsed = max(0.0, current - self._phase_started_at)
        self.phase = (elapsed % self.phase_period) / self.phase_period
        return self.phase

    def phase_pair(self) -> np.ndarray:
        angle = 2.0 * math.pi * float(self.phase)
        return np.asarray((math.sin(angle), math.cos(angle)), dtype=np.float32)

    def update_action(self, action: np.ndarray) -> None:
        self.last_action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)

    def update_command(self, command: Sequence[float]) -> None:
        self.command = np.asarray(command, dtype=np.float32).reshape(3)

    def update_joints(self, dof_pos: np.ndarray, dof_vel: np.ndarray, default_dof_pos: Optional[np.ndarray] = None) -> None:
        self.dof_pos = np.asarray(dof_pos, dtype=np.float32).reshape(ACTION_DIM)
        self.dof_vel = np.asarray(dof_vel, dtype=np.float32).reshape(ACTION_DIM)
        if default_dof_pos is not None:
            self.default_dof_pos = np.asarray(default_dof_pos, dtype=np.float32).reshape(ACTION_DIM)

    def prime_history(self, imu: Optional[ImuReading]) -> None:
        self.history.clear()
        for _ in range(self.history_length):
            self.build(imu)

    def history_vector(self) -> np.ndarray:
        frames = list(reversed(self.history)) if self.newest_first else list(self.history)
        return np.concatenate(frames).astype(np.float32)

    def build(
        self,
        imu: Optional[ImuReading],
        *,
        gyro: Optional[Sequence[float]] = None,
        projected_gravity: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """构造一帧观测。

        `gyro` / `projected_gravity` 是给非硬件场景（MuJoCo sim2sim、离线回放）用的
        显式入口：传进来的是**原始**机体系角速度 [rad/s] 与重力投影单位向量，缩放由
        这里统一处理，保证与走 IMU 那条路完全一致（同一份代码，只是数据来源不同）。
        只传 imu 时行为与以前逐位相同。
        """
        base_ang_vel = np.zeros(3, dtype=np.float32)
        gravity = np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
        if gyro is not None:
            base_ang_vel = np.asarray(gyro, dtype=np.float32) * ANGULAR_VELOCITY_SCALE
        elif imu is not None and imu.gyro is not None:
            base_ang_vel = np.asarray(imu.gyro, dtype=np.float32) * ANGULAR_VELOCITY_SCALE
        if projected_gravity is not None:
            gravity = np.asarray(projected_gravity, dtype=np.float32)
        elif imu is not None:
            gravity = np.asarray(euler_deg_to_projected_gravity(imu.euler_deg), dtype=np.float32)

        command = self.command * self.command_scale
        if self.head_order is not None:
            # 按策略要求的顺序拼前 9 维（三个 3 向量），其余段固定
            blocks = {
                "ang_vel": base_ang_vel,
                "gravity": gravity,
                "cmd": command,
            }
            parts = [blocks[name] for name in self.head_order]
            parts += [
                self.dof_pos - self.default_dof_pos,
                self.dof_vel * DOF_VELOCITY_SCALE,
                self.last_action,
            ]
        elif self.command_offset == 0:
            parts = [
                command,
                base_ang_vel,
                gravity,
                self.dof_pos - self.default_dof_pos,
                self.dof_vel * DOF_VELOCITY_SCALE,
                self.last_action,
            ]
        else:
            parts = [
                base_ang_vel,
                gravity,
                command,
                self.dof_pos - self.default_dof_pos,
                self.dof_vel * DOF_VELOCITY_SCALE,
                self.last_action,
            ]
        if self.has_phase:
            # unitree_rl_gym keeps the gait clock as the last two entries of the frame
            parts.append(self.phase_pair())
        obs = np.concatenate(parts).astype(np.float32)
        if obs.shape != (self.per_frame,):
            raise RuntimeError(
                "single observation shape mismatch: got %s, expected (%d,) - check "
                "has_phase/per_frame against the model file" % (obs.shape, self.per_frame)
            )
        self.history.append(obs)
        return self.history_vector()


class RemoteSwitch:
    """Placeholder for the receiver/remote switch.

    Later this should read the actual remote channel and return True only when
    the operator requests RL control.
    """

    def rl_requested(self) -> bool:
        return False


class CommandSource(Protocol):
    quit_requested: bool

    def open(self) -> None:
        ...

    def close(self) -> None:
        ...

    def poll(self) -> np.ndarray:
        ...


class KeyboardCommandSource:
    def __init__(self, speed: float = 0.01) -> None:
        self.speed = float(speed)
        self.command = np.zeros(3, dtype=np.float32)
        self.quit_requested = False
        self.enabled = False
        self._fd: Optional[int] = None
        self._old_settings = None

    def open(self) -> None:
        if not sys.stdin.isatty():
            print("keyboard control requested, but stdin is not a TTY; command stays zero", flush=True)
            return
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self.enabled = True
        print("keyboard control: W/S forward/back, A/D left/right, SPACE stop, Q quit", flush=True)

    def close(self) -> None:
        if self.enabled and self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        self.enabled = False
        self._fd = None
        self._old_settings = None

    def poll(self) -> np.ndarray:
        if not self.enabled:
            return self.command.copy()
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = self._read_key()
            self._apply_key(key)
        return self.command.copy()

    def _read_key(self) -> str:
        key = sys.stdin.read(1)
        if key != "\x1b":
            return key
        deadline = time.monotonic() + 0.01
        while len(key) < 3 and time.monotonic() < deadline:
            if select.select([sys.stdin], [], [], 0.0)[0]:
                key += sys.stdin.read(1)
            else:
                time.sleep(0.001)
        return key

    def _apply_key(self, key: str) -> None:
        key_lower = key.lower()
        if key_lower in ("q", "\x03"):
            self.quit_requested = True
            return
        if key_lower in (" ", "x"):
            self.command[:] = 0.0
        elif key_lower in ("w", "\x1b[a"):
            self.command[:] = (self.speed, 0.0, 0.0)
        elif key_lower in ("s", "\x1b[b"):
            self.command[:] = (-self.speed, 0.0, 0.0)
        elif key_lower in ("a", "\x1b[d"):
            self.command[:] = (0.0, self.speed, 0.0)
        elif key_lower in ("d", "\x1b[c"):
            self.command[:] = (0.0, -self.speed, 0.0)


class GamepadCommandSource:
    JS_EVENT_BUTTON = 0x01
    JS_EVENT_AXIS = 0x02
    JS_EVENT_INIT = 0x80
    EVENT_STRUCT = struct.Struct("<IhBB")

    def __init__(
        self,
        *,
        device: Optional[str] = None,
        linear_speed: float = 1.0,
        yaw_speed: float = 1.0,
        deadzone: float = 0.12,
        forward_axis: int = 1,
        lateral_axis: int = 0,
        yaw_axis: int = 3,
        forward_sign: float = -1.0,
        lateral_sign: float = -1.0,
        yaw_sign: float = -1.0,
        deadman_button: int = -1,
    ) -> None:
        self.device = device
        self.linear_speed = float(linear_speed)
        self.yaw_speed = float(yaw_speed)
        self.deadzone = max(0.0, min(0.95, float(deadzone)))
        self.forward_axis = int(forward_axis)
        self.lateral_axis = int(lateral_axis)
        self.yaw_axis = int(yaw_axis)
        self.forward_sign = float(forward_sign)
        self.lateral_sign = float(lateral_sign)
        self.yaw_sign = float(yaw_sign)
        self.deadman_button = int(deadman_button)
        self.command = np.zeros(3, dtype=np.float32)
        self.quit_requested = False
        self.enabled = False
        self._fd: Optional[int] = None
        self._device_path: Optional[str] = None
        self._axes: Dict[int, float] = {}
        self._buttons: Dict[int, bool] = {}

    def open(self) -> None:
        path = self._resolve_device()
        if path is None:
            print("gamepad control requested, but no /dev/input/js* device was found; command stays zero", flush=True)
            return
        try:
            self._fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        except OSError as exc:
            print("gamepad control requested, but cannot open %s: %s; command stays zero" % (path, exc), flush=True)
            self._fd = None
            return
        self._device_path = path
        self.enabled = True
        print(
            "gamepad control: %s left stick forward/lateral, right stick X yaw, A stop, Back quit"
            % path,
            flush=True,
        )
        self.poll()

    def close(self) -> None:
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
        self.enabled = False
        self._fd = None
        self._device_path = None

    def poll(self) -> np.ndarray:
        if not self.enabled or self._fd is None:
            self.command[:] = 0.0
            return self.command.copy()
        while True:
            try:
                data = os.read(self._fd, self.EVENT_STRUCT.size)
            except BlockingIOError:
                break
            except OSError as exc:
                self.command[:] = 0.0
                if exc.errno in (errno.ENODEV, errno.EIO, errno.ENXIO):
                    print("gamepad disconnected; command reset to zero", flush=True)
                    self.close()
                    break
                raise
            if len(data) != self.EVENT_STRUCT.size:
                break
            _, value, event_type, number = self.EVENT_STRUCT.unpack(data)
            event_type &= ~self.JS_EVENT_INIT
            if event_type == self.JS_EVENT_AXIS:
                self._axes[int(number)] = self._normalize_axis(value)
            elif event_type == self.JS_EVENT_BUTTON:
                self._buttons[int(number)] = bool(value)

        self._update_command_from_state()
        return self.command.copy()

    def _resolve_device(self) -> Optional[str]:
        if self.device:
            return self.device
        devices = sorted(glob.glob("/dev/input/js*"))
        if devices:
            return devices[0]
        return "/dev/input/js0"

    def _normalize_axis(self, value: int) -> float:
        normalized = max(-1.0, min(1.0, float(value) / 32767.0))
        magnitude = abs(normalized)
        if magnitude <= self.deadzone:
            return 0.0
        return np.sign(normalized) * ((magnitude - self.deadzone) / (1.0 - self.deadzone))

    def _axis(self, axis: int) -> float:
        return self._axes.get(axis, 0.0)

    def _button(self, button: int) -> bool:
        return button >= 0 and self._buttons.get(button, False)

    def _update_command_from_state(self) -> None:
        if self._button(6):
            self.quit_requested = True
        if self._button(0):
            self.command[:] = 0.0
            return
        if self.deadman_button >= 0 and not self._button(self.deadman_button):
            self.command[:] = 0.0
            return
        self.command[:] = (
            self.forward_sign * self._axis(self.forward_axis) * self.linear_speed,
            self.lateral_sign * self._axis(self.lateral_axis) * self.linear_speed,
            self.yaw_sign * self._axis(self.yaw_axis) * self.yaw_speed,
        )


class DeploymentRunner:
    def __init__(
        self,
        model_path: str,
        *,
        yaw_model_path: Optional[str] = None,
        obs_head_order: Optional[str] = None,
        use_imu: bool = True,
        use_can: bool = True,
        startup: bool = False,
        auto_rl_after_start: bool = False,
        send_actions: bool = False,
        loop_hz: float = 50.0,
        startup_speed_scale: float = 1.0,
        require_motor_feedback: bool = True,
        startup_feedback_timeout: float = 2.0,
        feedback_timeout: float = 0.5,
        rl_action_scale: float = RL_ACTION_SCALE,
        action_clip: float = 0.0,
        print_full_action: bool = False,
        hold_after_run: float = 0.0,
        command: Optional[Sequence[float]] = None,
        play_history_bootstrap: bool = True,
        startup_hold_before_rl: float = 3.0,
        use_startup_pose_as_rl_origin: bool = False,
        switch_mit_mode: bool = True,
        keyboard_control: bool = False,
        keyboard_speed: float = 1.0,
        gamepad_control: bool = False,
        gamepad_device: Optional[str] = None,
        gamepad_linear_speed: float = 1.0,
        gamepad_yaw_speed: float = 1.0,
        gamepad_deadzone: float = 0.12,
        gamepad_forward_axis: int = 1,
        gamepad_lateral_axis: int = 0,
        gamepad_yaw_axis: int = 3,
        gamepad_forward_sign: float = -1.0,
        gamepad_lateral_sign: float = -1.0,
        gamepad_yaw_sign: float = -1.0,
        gamepad_deadman_button: int = -1,
        command_source: Optional[CommandSource] = None,
        shutdown_lie_down: bool = True,
        shutdown_lie_down_hold: float = 0.25,
        policy_gain_transition_duration: float = 0.4,
        policy_switch_pause_duration: float = 0.5,
    ) -> None:
        self.policy = PolicyDeployer(model_path, name="translation_policy", head_order=obs_head_order)
        self.yaw_policy = self.policy
        if yaw_model_path:
            self.yaw_policy = PolicyDeployer(yaw_model_path, name="yaw_policy", head_order=obs_head_order)
        self._active_policy_name = self.policy.name
        self.translation_obs_builder = ObservationBuilder(
            history_length=self.policy.history_length,
            command_offset=self.policy.command_offset,
            newest_first=self.policy.newest_first,
            per_frame=self.policy.per_frame,
            has_phase=self.policy.has_phase,
            head_order=self.policy.head_order,
        )
        self.yaw_obs_builder = self.translation_obs_builder
        if self.yaw_policy is not self.policy:
            self.yaw_obs_builder = ObservationBuilder(
                history_length=self.yaw_policy.history_length,
                command_offset=self.yaw_policy.command_offset,
                newest_first=self.yaw_policy.newest_first,
                per_frame=self.yaw_policy.per_frame,
                has_phase=self.yaw_policy.has_phase,
                head_order=self.yaw_policy.head_order,
            )
        self.obs_builder = self.translation_obs_builder
        if command is not None:
            self._update_command(command)
        self.use_imu = use_imu
        self.use_can = use_can
        self.startup = startup
        self.auto_rl_after_start = auto_rl_after_start
        self.send_actions = send_actions
        self.loop_hz = float(loop_hz)
        self.startup_speed_scale = float(startup_speed_scale)
        self.require_motor_feedback = require_motor_feedback
        self.startup_feedback_timeout = max(0.1, float(startup_feedback_timeout))
        self.feedback_timeout = float(feedback_timeout)
        self.rl_action_scale = float(rl_action_scale)
        self.action_clip = float(action_clip)
        self.print_full_action = bool(print_full_action)
        self.hold_after_run = max(0.0, float(hold_after_run))
        self.play_history_bootstrap = bool(play_history_bootstrap)
        self.startup_hold_before_rl = max(0.0, float(startup_hold_before_rl))
        self.use_startup_pose_as_rl_origin = bool(use_startup_pose_as_rl_origin)
        self.switch_mit_mode = bool(switch_mit_mode)
        self.keyboard = KeyboardCommandSource(keyboard_speed) if keyboard_control else None
        self.gamepad = (
            GamepadCommandSource(
                device=gamepad_device,
                linear_speed=gamepad_linear_speed,
                yaw_speed=gamepad_yaw_speed,
                deadzone=gamepad_deadzone,
                forward_axis=gamepad_forward_axis,
                lateral_axis=gamepad_lateral_axis,
                yaw_axis=gamepad_yaw_axis,
                forward_sign=gamepad_forward_sign,
                lateral_sign=gamepad_lateral_sign,
                yaw_sign=gamepad_yaw_sign,
                deadman_button=gamepad_deadman_button,
            )
            if gamepad_control
            else None
        )
        self.command_source = command_source
        self.shutdown_lie_down = bool(shutdown_lie_down)
        self.shutdown_lie_down_hold = max(0.0, float(shutdown_lie_down_hold))
        self.policy_gain_transition_duration = max(0.0, float(policy_gain_transition_duration))
        self.policy_switch_pause_duration = max(0.0, float(policy_switch_pause_duration))
        self._smoothed_policy_gains: Optional[Tuple[float, float, float, float]] = None
        self._gain_transition_start_gains: Optional[Tuple[float, float, float, float]] = None
        self._gain_transition_target_gains: Optional[Tuple[float, float, float, float]] = None
        self._gain_transition_start_time = time.monotonic()
        self.imu: Optional[ImuSensor] = None
        self.can: Optional[Usb2CanHub] = None
        self.motors: Optional[MotorBus] = None
        self.actions: Optional[DogPostureActionLibrary] = None
        self.startup_controller: Optional[DogStartupController] = None
        self.startup_result: Optional[StartupResult] = None
        self.rl_origin_pose: Optional[np.ndarray] = None
        self.remote = RemoteSwitch()
        self.rl_active = False
        # Operator "stop" request (gamepad stop button). While set, the policy keeps
        # running but its output is not applied - the startup pose is held instead, so
        # the dog stops and stands still. Any new motion command clears it.
        self.hold_requested = False
        self.hold_pose: Optional[np.ndarray] = None
        # Set when the stall guard trips, so shutdown skips the reverse startup
        # sequence instead of driving a jammed leg even more.
        self.stall_detected = False
        self.posture_action_active = False
        self._last_policy_skip_reason: Optional[str] = None
        # Gait-clock state for the deployable (unitree_rl_gym) observation layout:
        # the clock runs only while the RL policy is actually in control.
        self._phase_running = False
        self.last_feedback_time = time.monotonic()
        self.emergency_stop_requested = threading.Event()
        self._emergency_lock = threading.Lock()
        self._emergency_disable_sent = False

    def _validate_policy_compatibility(self, translation_policy: PolicyDeployer, yaw_policy: PolicyDeployer) -> None:
        fields = (
            ("total_input_dim", translation_policy.total_input_dim, yaw_policy.total_input_dim),
            ("per_frame", translation_policy.per_frame, yaw_policy.per_frame),
            ("history_length", translation_policy.history_length, yaw_policy.history_length),
            ("has_phase", translation_policy.has_phase, yaw_policy.has_phase),
            ("command_offset", translation_policy.command_offset, yaw_policy.command_offset),
            ("newest_first", translation_policy.newest_first, yaw_policy.newest_first),
        )
        mismatched = ["%s:%s!=%s" % item for item in fields if item[1] != item[2]]
        if mismatched:
            raise ValueError("translation/yaw policy observation formats differ: %s" % ", ".join(mismatched))

    def _obs_builders(self) -> Tuple[ObservationBuilder, ...]:
        if self.yaw_obs_builder is self.translation_obs_builder:
            return (self.translation_obs_builder,)
        return (self.translation_obs_builder, self.yaw_obs_builder)

    def _builder_for_policy(self, policy: PolicyDeployer) -> ObservationBuilder:
        return self.yaw_obs_builder if policy is self.yaw_policy else self.translation_obs_builder

    def _phase_builders(self) -> list:
        """Distinct observation builders that carry a gait clock."""
        builders = []
        for builder in (self.translation_obs_builder, self.yaw_obs_builder):
            if builder.has_phase and all(builder is not other for other in builders):
                builders.append(builder)
        return builders

    def _update_gait_clock(self) -> None:
        """Start/stop/advance the gait clock according to the RL state.

        The simulated policy sees the clock reset at every episode start; the real
        robot has no episode boundary, so we restart it whenever the policy takes
        over (first activation, and again after every stop/hold). Without this the
        phase would be arbitrary at stand-up time and the policy could try to swing
        a leg while the dog is still settling.
        """
        for builder in self._phase_builders():
            if self.rl_active and not self._phase_running:
                builder.start_phase()
            elif not self.rl_active and self._phase_running:
                builder.stop_phase()
            builder.tick_phase()
        self._phase_running = bool(self.rl_active)

    def _update_command(self, command: Sequence[float]) -> None:
        for builder in self._obs_builders():
            builder.update_command(command)

    def _update_action(self, action: np.ndarray) -> None:
        for builder in self._obs_builders():
            builder.update_action(action)

    def _update_joints(self, dof_pos: np.ndarray, dof_vel: np.ndarray, default_dof_pos: Optional[np.ndarray] = None) -> None:
        for builder in self._obs_builders():
            builder.update_joints(dof_pos, dof_vel, default_dof_pos=default_dof_pos)

    def _prime_histories(self, imu: Optional[ImuReading]) -> None:
        for builder in self._obs_builders():
            builder.prime_history(imu)

    def open(self) -> None:
        if self.use_imu:
            self.imu = ImuSensor(IMU_PORT, IMU_BAUDRATE).open()
            reading = self.imu.read_complete(timeout=1.0)
            if reading is None:
                raise RuntimeError("IMU opened but no complete sample was received")
        if self.use_can:
            self.can = Usb2CanHub(CAN_SERIALS, CAN_CHANNELS, CAN_BITRATE, CANFD_ENABLED).open()
            self.motors = MotorBus(self.can, CAN_BUS_MAP)
            self.actions = DogPostureActionLibrary(
                self.motors,
                kp=STARTUP_KP,
                kd=STARTUP_KD,
                speed_scale=self.startup_speed_scale,
                require_feedback=self.require_motor_feedback,
                feedback_timeout=self.startup_feedback_timeout,
                switch_mit_mode=self.switch_mit_mode,
                emergency_checker=self.emergency_stop_requested.is_set,
            )
        if self.startup:
            if self.motors is None:
                raise RuntimeError("startup requires CAN motor bus")
            self.stand_up()

    @property
    def is_standing(self) -> bool:
        return self.startup_result is not None

    def stand_up(self) -> None:
        if self.emergency_stop_requested.is_set():
            raise RuntimeError("stand_up blocked by emergency stop")
        if self.actions is None or self.motors is None:
            raise RuntimeError("stand_up requires CAN motor bus")
        if self.is_standing:
            return
        self.posture_action_active = True
        self._reset_policies()
        self._update_action(np.zeros(ACTION_DIM, dtype=np.float32))
        try:
            if self.emergency_stop_requested.is_set():
                raise RuntimeError("stand_up blocked by emergency stop")
            startup_result = self.actions.stand_up()
            self.startup_controller = self.actions.startup_controller
            self.startup_result = startup_result
            self.rl_origin_pose = startup_result.startup_pose.copy()
            if self.startup_hold_before_rl > 0:
                print("holding startup pose before RL for %.1fs" % self.startup_hold_before_rl, flush=True)
                self.motors.clear_feedback_history()
                if self.startup_controller is not None:
                    self.startup_controller.hold_current_pose(duration=self.startup_hold_before_rl)
                if not self.motors.wait_for_feedback(timeout=0.5, require_all=self.require_motor_feedback):
                    raise RuntimeError("missing motor feedback after startup hold: %s" % (self.motors.missing_feedback(),))
                if not self.use_startup_pose_as_rl_origin:
                    self.rl_origin_pose = self.motors.current_positions().copy()
            if self.use_startup_pose_as_rl_origin:
                self.rl_origin_pose = self._configured_rl_origin_pose(startup_result.startup_pose)
            self._print_standing_pose_for_config()
            self._update_joints(
                DEFAULT_DOF_POS_ARRAY,
                np.zeros(ACTION_DIM, dtype=np.float32),
                default_dof_pos=DEFAULT_DOF_POS_ARRAY,
            )
            imu_reading = self.imu.read_once() if self.imu is not None else None
            self._prime_histories(imu_reading)
            self.rl_active = self.auto_rl_after_start or self.keyboard is not None or self.gamepad is not None or self.command_source is not None
            self.last_feedback_time = time.monotonic()
        finally:
            self._reset_policies()
            self.posture_action_active = False

    def lie_down(self) -> None:
        if self.emergency_stop_requested.is_set():
            self._disable_after_emergency()
            return
        if self.actions is None:
            return
        self.posture_action_active = True
        self.rl_active = False
        self._reset_policies()
        self._update_action(np.zeros(ACTION_DIM, dtype=np.float32))
        self._update_command((0.0, 0.0, 0.0))
        try:
            self.actions.lie_down(hold_duration=self.shutdown_lie_down_hold)
            if self.motors is not None:
                self.motors.disable_all()
                time.sleep(0.05)
            self.startup_controller = None
            self.startup_result = None
            self.rl_origin_pose = None
            self.last_feedback_time = time.monotonic()
        finally:
            self._reset_policies()
            self.posture_action_active = False

    def close(self) -> None:
        if self.motors is not None:
            try:
                get_guard(self.motors).report()
            except Exception:
                pass
            if not self.emergency_stop_requested.is_set():
                try:
                    self._lie_down_before_disable()
                except Exception as exc:
                    print("reverse startup sequence failed before motor disable: %s" % exc, flush=True)
            try:
                self.motors.disable_all()
                time.sleep(0.05)
            except Exception:
                pass
        if self.can is not None:
            self.can.close()
            self.can = None
        self.motors = None
        if self.imu is not None:
            self.imu.close()
            self.imu = None
        if self.keyboard is not None:
            self.keyboard.close()
        if self.gamepad is not None:
            self.gamepad.close()
        if self.command_source is not None:
            self.command_source.close()

    def _lie_down_before_disable(self) -> None:
        if not self.shutdown_lie_down:
            return
        if self.stall_detected:
            print("skipping the reverse startup sequence: the stall guard tripped, "
                  "just disabling the motors", flush=True)
            return
        if self.motors is None or self.startup_result is None:
            return
        print("running reverse startup sequence before disabling motors", flush=True)
        self.lie_down()

    def run(self, steps: int = 0, print_every: int = 25) -> None:
        if self.send_actions and self.motors is None:
            raise RuntimeError("sending actions requires CAN motor bus")

        period = 1.0 / self.loop_hz if self.loop_hz > 0 else 0.0
        step = 0
        self.last_feedback_time = time.monotonic()
        if self.keyboard is not None:
            self.keyboard.open()
        if self.gamepad is not None:
            self.gamepad.open()
        if self.command_source is not None:
            self.command_source.open()
        try:
            while steps <= 0 or step < steps:
                t0 = time.monotonic()
                if self.emergency_stop_requested.is_set():
                    self._disable_after_emergency()
                    print("emergency stop active; leaving policy loop", flush=True)
                    break
                imu_reading = self.imu.read_once() if self.imu is not None else None
                if self.keyboard is not None:
                    self._update_command(self.keyboard.poll())
                    if self.keyboard.quit_requested:
                        print("keyboard quit requested", flush=True)
                        break
                if self.gamepad is not None:
                    self._update_command(self.gamepad.poll())
                    if self.gamepad.quit_requested:
                        print("gamepad quit requested", flush=True)
                        break
                if self.command_source is not None:
                    self._update_command(self.command_source.poll())
                    if self.command_source.quit_requested:
                        print("ROS command source quit requested", flush=True)
                        break
                rx_count = 0
                if self.motors is not None:
                    rx_count = self.motors.poll()
                    if rx_count > 0:
                        self.last_feedback_time = time.monotonic()
                    raw_pos = self.motors.current_positions().reshape(ACTION_DIM)
                    raw_vel = self.motors.current_velocities().reshape(ACTION_DIM)
                    if self.startup_result is not None:
                        dof_pos = self._raw_to_model_positions(raw_pos)
                        dof_vel = self._raw_to_model_velocities(raw_vel)
                        default_dof_pos = DEFAULT_DOF_POS_ARRAY
                    else:
                        dof_pos = raw_pos
                        dof_vel = raw_vel
                        default_dof_pos = None
                    self._update_joints(
                        dof_pos,
                        dof_vel,
                        default_dof_pos=default_dof_pos,
                    )

                # Gait clock must be advanced BEFORE the observation is built below.
                self._update_gait_clock()

                policy_skip_reason = self._policy_skip_reason()
                if policy_skip_reason is None:
                    if self._last_policy_skip_reason is not None:
                        self._reset_policies()
                        self._prime_histories(imu_reading)
                    active_policy = self._select_policy(self.obs_builder.command, imu_reading)
                    active_builder = self._builder_for_policy(active_policy)
                    self.obs_builder = active_builder
                    if self.play_history_bootstrap and step == 0:
                        obs_history = active_builder.history_vector()
                    else:
                        obs_history = active_builder.build(imu_reading)
                    action = active_policy.inference(obs_history)
                    effective_action = self._effective_action(action)
                    raw_action_units = self._raw_action_from_model_action(effective_action)
                    self._update_action(effective_action)
                else:
                    if policy_skip_reason != self._last_policy_skip_reason:
                        print("policy inference skipped: %s" % policy_skip_reason, flush=True)
                    obs_history = self.obs_builder.history_vector()
                    action = np.zeros(ACTION_DIM, dtype=np.float32)
                    effective_action = np.zeros(ACTION_DIM, dtype=np.float32)
                    raw_action_units = np.zeros(ACTION_DIM, dtype=np.float32)
                    if policy_skip_reason in ("lying", "posture_action_active", "rl_inactive"):
                        self._update_action(effective_action)
                        self._reset_policies()
                self._last_policy_skip_reason = policy_skip_reason
                if self.can is not None and self.motors is None:
                    rx_count = len(self.can.drain())

                if self.hold_requested:
                    # Operator stop: hold the stand pose, do not apply policy output.
                    self.rl_active = False
                elif self._command_requests_motion():
                    if self.startup_result is not None and not self.posture_action_active:
                        self.rl_active = True
                        self.hold_pose = None
                if self.remote.rl_requested():
                    if self.startup_result is not None and not self.posture_action_active:
                        self.rl_active = True
                if self.startup_result is None or self.posture_action_active:
                    self.rl_active = False

                if self.motors is not None and self.startup_result is not None and not self.posture_action_active:
                    if self.send_actions and self.rl_active:
                        if self.feedback_timeout > 0 and time.monotonic() - self.last_feedback_time > self.feedback_timeout:
                            raise RuntimeError("motor feedback timeout during RL control: %.3fs" % (time.monotonic() - self.last_feedback_time))
                        self._send_actions(effective_action)
                    else:
                        if self.hold_pose is not None:
                            # Operator stop: freeze in place, but keep the stiff gains
                            # the dog was just standing with, otherwise kp=30 is too
                            # soft and the dog sags down.
                            kp, kd, bm_kp, bm_kd = self._active_policy_gains()
                            with self.motors.bm_calf_gain_override(bm_kp, bm_kd):
                                self._send_pose(self.hold_pose, kp=kp, kd=kd)
                        else:
                            self._send_pose(self.startup_result.startup_pose, kp=STARTUP_KP, kd=STARTUP_KD)

                if self.motors is not None:
                    try:
                        get_guard(self.motors).update("rl")
                    except StallDetected as exc:
                        print("MOTOR GUARD: %s -> disabling the motors immediately" % exc, flush=True)
                        self.stall_detected = True
                        self.request_emergency_stop()
                        break

                if print_every > 0 and step % print_every == 0:
                    rpy = imu_reading.euler_deg if imu_reading is not None else None
                    gyro = imu_reading.gyro if imu_reading is not None else None
                    gravity = euler_deg_to_projected_gravity(rpy)
                    rejected = self.motors.rejected_feedback if self.motors is not None else 0
                    rejected_id = self.motors.rejected_feedback_id if self.motors is not None else 0
                    print(
                        "step=%d rl=%s active_policy=%s cmd=%s rpy=%s gyro=%s gravity=%s policy_action[0:3]=%s clipped_action[0:3]=%s raw_action_units[0:3]=%s can_rx=%d fb_rej=%d fb_rej_id=%d"
                        % (
                            step,
                            self.rl_active,
                            self._active_policy_name,
                            _fmt_tuple(self.obs_builder.command),
                            _fmt_tuple(rpy),
                            _fmt_tuple(gyro),
                            _fmt_tuple(gravity),
                            np.array2string(action[:3], precision=3),
                            np.array2string(effective_action[:3], precision=3),
                            np.array2string(raw_action_units[:3], precision=3),
                            rx_count,
                            rejected,
                            rejected_id,
                        ),
                        flush=True,
                    )
                    if self.print_full_action:
                        self._print_debug_action(action, obs_history)

                step += 1
                elapsed = time.monotonic() - t0
                if period > elapsed:
                    time.sleep(period - elapsed)
        finally:
            if self.keyboard is not None:
                self.keyboard.close()
            if self.gamepad is not None:
                self.gamepad.close()
            if self.command_source is not None:
                self.command_source.close()
            if not self.emergency_stop_requested.is_set():
                self._hold_startup_pose_after_run(period)

    def request_hold(self) -> None:
        """Operator stop: stop applying policy output and hold the standing pose.

        Cleared automatically as soon as a motion command arrives again
        (see DeploymentRunner.run), so pushing the stick resumes walking.
        """
        self.hold_requested = True
        self.rl_active = False
        self._update_action(np.zeros(ACTION_DIM, dtype=np.float32))
        self._update_command((0.0, 0.0, 0.0))
        self._reset_policies()
        # Freeze at the pose the policy was last COMMANDING (not the measured
        # positions): that keeps the same gravity compensation, so the dog stops
        # in place without sagging.
        if self.motors is not None:
            try:
                self.hold_pose = np.asarray(self.motors.commands, dtype=np.float32).copy()
            except Exception:
                self.hold_pose = None

    def clear_hold(self) -> None:
        self.hold_requested = False

    def _command_requests_motion(self) -> bool:
        command = np.asarray(self.obs_builder.command, dtype=np.float32).reshape(3)
        return bool(np.any(np.abs(command) > 1e-3))

    def request_emergency_stop(self) -> None:
        self.emergency_stop_requested.set()
        self.rl_active = False
        self._update_action(np.zeros(ACTION_DIM, dtype=np.float32))
        self._update_command((0.0, 0.0, 0.0))
        self._reset_policies()
        self._disable_after_emergency()

    def _disable_after_emergency(self) -> None:
        with self._emergency_lock:
            if self._emergency_disable_sent:
                return
            self._emergency_disable_sent = True
            self.rl_active = False
            self.posture_action_active = False
            self.startup_controller = None
            self.startup_result = None
            self.rl_origin_pose = None
            if self.actions is not None:
                self.actions.startup_controller = None
                self.actions.startup_result = None
                self.actions.is_standing = False
            if self.motors is not None:
                try:
                    self.motors.disable_all()
                    time.sleep(0.02)
                except Exception as exc:
                    print("failed to disable motors during emergency stop: %s" % exc, flush=True)

    def _send_actions(self, action: np.ndarray) -> None:
        if self.motors is None or self.startup_result is None:
            return
        target = self._raw_targets_from_model_action(action)
        kp, kd, bm_calf_kp, bm_calf_kd = self._active_policy_gains()
        with self.motors.bm_calf_gain_override(bm_calf_kp, bm_calf_kd):
            self._send_pose(target, kp=kp, kd=kd)

    def _reset_policies(self) -> None:
        self.policy.reset_state()
        if self.yaw_policy is not self.policy:
            self.yaw_policy.reset_state()

    def _select_policy(self, command: Sequence[float], imu_reading: Optional[ImuReading]) -> PolicyDeployer:
        command_array = np.asarray(command, dtype=np.float32).reshape(3)
        x = float(command_array[0])
        y_abs = abs(float(command_array[1]))
        yaw_abs = abs(float(command_array[2]))
        eps = 1e-4
        command_is_zero = abs(x) <= eps and y_abs <= eps and yaw_abs <= eps
        forward_without_lateral = x > eps and y_abs <= eps
        backward = x < -eps
        # Pure turn-in-place must use the yaw policy too: the translation model
        # produces no stepping gait for a yaw-only command (measured: 0 joint
        # direction reversals, body barely rotated), while the yaw model steps
        # properly (100+ reversals, real rotation).
        pure_yaw = abs(x) <= eps and y_abs <= eps and yaw_abs > eps
        selected = self.yaw_policy if (forward_without_lateral or backward or pure_yaw) else self.policy
        if selected.name != self._active_policy_name:
            print("switching active policy: %s -> %s for cmd=%s" % (self._active_policy_name, selected.name, _fmt_tuple(command_array)), flush=True)
            self._reset_policies()
            self._prime_histories(imu_reading)
            self._active_policy_name = selected.name
            self.obs_builder = self._builder_for_policy(selected)
            self._pause_after_policy_switch()
        return selected

    def _pause_after_policy_switch(self) -> None:
        if self.policy_switch_pause_duration <= 0.0:
            return

        hold_pose, hold_kp, hold_kd = self._current_pose_for_policy_switch_hold()
        if hold_pose is None:
            print("pausing policy output for %.2fs after model switch" % self.policy_switch_pause_duration, flush=True)
        else:
            print(
                "pausing policy output for %.2fs after model switch, holding current pose"
                % self.policy_switch_pause_duration,
                flush=True,
            )
        deadline = time.monotonic() + self.policy_switch_pause_duration
        period = 1.0 / self.loop_hz if self.loop_hz > 0 else 0.02
        period = min(max(period, 0.005), 0.05)
        while time.monotonic() < deadline:
            if self.emergency_stop_requested.is_set():
                return
            if self.motors is not None:
                rx_count = self.motors.poll()
                if rx_count > 0:
                    self.last_feedback_time = time.monotonic()
            if hold_pose is not None:
                self._send_pose_with_joint_gains(hold_pose, hold_kp, hold_kd)
            time.sleep(min(period, max(0.0, deadline - time.monotonic())))

    def _current_pose_for_policy_switch_hold(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        if self.motors is None:
            return None, None, None

        self.motors.poll(limit=4096)
        if self.motors.missing_feedback():
            return None, None, None
        return self.motors.current_positions(), self.motors.kp.copy(), self.motors.kd.copy()

    def _active_policy(self) -> PolicyDeployer:
        if self.yaw_policy.name == self._active_policy_name:
            return self.yaw_policy
        return self.policy

    def _active_policy_gains(self) -> Tuple[float, float, float, float]:
        target = self._target_policy_gains()
        return self._smooth_policy_gains(target)

    def _target_policy_gains(self) -> Tuple[float, float, float, float]:
        if self._active_policy() is self.yaw_policy and self.yaw_policy is not self.policy:
            return (
                float(YAW_POLICY_RL_KP),
                float(YAW_POLICY_RL_KD),
                float(YAW_POLICY_BM_CALF_KP),
                float(YAW_POLICY_BM_CALF_KD),
            )
        return (
            float(TRANSLATION_POLICY_RL_KP),
            float(TRANSLATION_POLICY_RL_KD),
            float(TRANSLATION_POLICY_BM_CALF_KP),
            float(TRANSLATION_POLICY_BM_CALF_KD),
        )

    def _smooth_policy_gains(self, target: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        now = time.monotonic()
        target = tuple(float(v) for v in target)
        if self.policy_gain_transition_duration <= 0.0 or self._smoothed_policy_gains is None:
            self._smoothed_policy_gains = target
            self._gain_transition_start_gains = target
            self._gain_transition_target_gains = target
            self._gain_transition_start_time = now
            return target

        if self._gain_transition_target_gains is None or any(
            abs(a - b) > 1e-6 for a, b in zip(target, self._gain_transition_target_gains)
        ):
            self._gain_transition_start_gains = self._smoothed_policy_gains
            self._gain_transition_target_gains = target
            self._gain_transition_start_time = now
            print(
                "smoothing policy gains: current=(%.3f, %.3f, %.3f, %.3f) target=(%.3f, %.3f, %.3f, %.3f) duration=%.2fs"
                % (
                    self._smoothed_policy_gains[0],
                    self._smoothed_policy_gains[1],
                    self._smoothed_policy_gains[2],
                    self._smoothed_policy_gains[3],
                    target[0],
                    target[1],
                    target[2],
                    target[3],
                    self.policy_gain_transition_duration,
                ),
                flush=True,
            )

        assert self._gain_transition_start_gains is not None
        assert self._gain_transition_target_gains is not None
        alpha = min(1.0, max(0.0, (now - self._gain_transition_start_time) / self.policy_gain_transition_duration))
        start = self._gain_transition_start_gains
        goal = self._gain_transition_target_gains
        self._smoothed_policy_gains = tuple(
            start_value + (goal_value - start_value) * alpha
            for start_value, goal_value in zip(start, goal)
        )
        return self._smoothed_policy_gains

    def _policy_skip_reason(self) -> Optional[str]:
        if self.posture_action_active:
            return "posture_action_active"
        if self.startup_result is None:
            return "lying"
        if self.use_can:
            if self.motors is None:
                return "motors_not_open"
            if not self._motor_feedback_ready():
                return "motor_feedback_missing:%s" % (self.motors.missing_feedback(),)
        if not self.rl_active:
            return "rl_inactive"
        return None

    def _motor_feedback_ready(self) -> bool:
        if self.motors is None:
            return False
        if self.require_motor_feedback:
            return not self.motors.missing_feedback()
        return any(fb.seen for fb in self.motors.feedback.values())

    def _effective_action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        if self.action_clip > 0:
            action = np.clip(action, -self.action_clip, self.action_clip)
        return action

    def _raw_action_from_model_action(self, action: np.ndarray) -> np.ndarray:
        """Map model residual action units onto raw motor residual action units.

        先按 MODEL_JOINT_PERMUTATION 把策略的关节顺序换成 dm 电机顺序，
        再乘以每关节的方向符号。默认置换是恒等，所以对 go2 策略无影响。
        """
        action = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        action = action[MODEL_JOINT_PERMUTATION_ARRAY]
        return (action * MODEL_JOINT_DIRECTIONS_FLAT).astype(np.float32)

    def _raw_targets_from_model_action(self, action: np.ndarray) -> np.ndarray:
        effective = self._effective_action(action)
        raw_action = self._raw_action_from_model_action(effective)
        residual = raw_action.reshape(4, 3) * self.rl_action_scale
        return self._rl_origin_pose() + residual

    def _raw_to_model_positions(self, raw_pos: np.ndarray) -> np.ndarray:
        """dm 关节位置 → 策略关节顺序的位置偏差。

        注意置换方向：dm 顺序 → 策略顺序用的是逆置换。
        """
        if self.startup_result is None:
            return np.asarray(raw_pos, dtype=np.float32).reshape(ACTION_DIM)
        raw_pos = np.asarray(raw_pos, dtype=np.float32).reshape(ACTION_DIM)
        startup_raw = self._rl_origin_pose().reshape(ACTION_DIM)
        raw_residual = raw_pos - startup_raw
        model_delta = raw_residual * MODEL_JOINT_DIRECTIONS_FLAT
        model_delta = model_delta[MODEL_JOINT_INV_PERMUTATION_ARRAY]
        return DEFAULT_DOF_POS_ARRAY + model_delta

    def _raw_to_model_velocities(self, raw_vel: np.ndarray) -> np.ndarray:
        raw_vel = np.asarray(raw_vel, dtype=np.float32).reshape(ACTION_DIM)
        return raw_vel * MODEL_JOINT_DIRECTIONS_FLAT

    def _print_debug_action(self, action: np.ndarray, obs_history: np.ndarray) -> None:
        if self.startup_result is None:
            return
        effective = self._effective_action(action).reshape(4, 3)
        raw_action = self._raw_action_from_model_action(effective.reshape(ACTION_DIM)).reshape(4, 3)
        q_model = self.obs_builder.dof_pos - self.obs_builder.default_dof_pos
        qd_model = self.obs_builder.dof_vel
        target = self._raw_targets_from_model_action(effective)
        estimated_vel = self._active_policy().estimate_base_velocity(obs_history)
        if estimated_vel is not None:
            print("  estimator base_lin_vel=%s" % np.array2string(estimated_vel, precision=3), flush=True)
        print("  action model FL/FR/RL/RR:", flush=True)
        _print_matrix(effective)
        print("  action raw-direction-mapped FL/FR/RL/RR:", flush=True)
        _print_matrix(raw_action)
        print("  q_delta_model FL/FR/RL/RR:", flush=True)
        _print_matrix(q_model.reshape(4, 3))
        print("  qd_model FL/FR/RL/RR:", flush=True)
        _print_matrix(qd_model.reshape(4, 3))
        print("  target_raw FL/FR/RL/RR:", flush=True)
        _print_matrix(target)

    def _print_standing_pose_for_config(self) -> None:
        if self.motors is None or self.startup_result is None:
            return
        self.motors.poll(limit=4096)
        current_pose = self.motors.current_positions()
        origin_pose = self._rl_origin_pose()
        current_text = _python_list_text(current_pose.reshape(ACTION_DIM))
        origin_text = _python_list_text(origin_pose.reshape(ACTION_DIM))
        content = (
            "# Leg order: FL hip/thigh/calf, FR hip/thigh/calf, RL hip/thigh/calf, RR hip/thigh/calf\n"
            "# Paste one of these into bs_hardware/config.py: FIXED_RL_ORIGIN_POSE_RAW\n"
            "COPY_CURRENT_FEEDBACK_POSE_RAW = %s\n"
            "COPY_RL_ORIGIN_POSE_RAW = %s\n"
            "FIXED_RL_ORIGIN_POSE_RAW = %s\n"
            % (current_text, origin_text, current_text)
        )
        path = "/tmp/dm_dog_standing_pose.txt"
        try:
            with open(path, "w", encoding="utf-8") as file:
                file.write(content)
        except OSError as exc:
            print("failed to write standing pose snapshot to %s: %s" % (path, exc), flush=True)
        print("standing pose snapshot saved to %s" % path, flush=True)
        print("COPY_FIXED_RL_ORIGIN_POSE_RAW = %s" % current_text, flush=True)

    def _configured_rl_origin_pose(self, fallback_pose: np.ndarray) -> np.ndarray:
        if FIXED_RL_ORIGIN_POSE_RAW_ARRAY.size == ACTION_DIM:
            print("using FIXED_RL_ORIGIN_POSE_RAW from config.py as RL origin pose", flush=True)
            return FIXED_RL_ORIGIN_POSE_RAW_ARRAY.reshape(4, 3).copy()
        if FIXED_RL_ORIGIN_POSE_RAW_ARRAY.size not in (0, ACTION_DIM):
            print(
                "ignoring FIXED_RL_ORIGIN_POSE_RAW: expected 12 values, got %d"
                % FIXED_RL_ORIGIN_POSE_RAW_ARRAY.size,
                flush=True,
            )
        print("using startup_result.startup_pose as RL origin pose", flush=True)
        return np.asarray(fallback_pose, dtype=np.float32).reshape(4, 3).copy()

    def _rl_origin_pose(self) -> np.ndarray:
        if self.rl_origin_pose is not None:
            return self.rl_origin_pose
        if self.startup_result is not None:
            return self.startup_result.startup_pose
        raise RuntimeError("RL origin pose is not available")

    def _send_pose(self, target: np.ndarray, *, kp: float, kd: float) -> None:
        if self.motors is None:
            return
        target = np.asarray(target, dtype=np.float32).reshape(4, 3)
        for leg in range(4):
            for joint in range(3):
                self.motors.send_mit(
                    leg,
                    joint,
                    MitCommand(pos=float(target[leg, joint]), vel=0.0, kp=float(kp), kd=float(kd), tor=0.0),
                )

    def _send_pose_with_joint_gains(
        self,
        target: np.ndarray,
        kp: Optional[np.ndarray],
        kd: Optional[np.ndarray],
    ) -> None:
        if self.motors is None:
            return
        if kp is None or kd is None:
            kp = self.motors.kp.copy()
            kd = self.motors.kd.copy()

        target = np.asarray(target, dtype=np.float32).reshape(4, 3)
        kp_matrix = np.asarray(kp, dtype=np.float32).reshape(4, 3)
        kd_matrix = np.asarray(kd, dtype=np.float32).reshape(4, 3)
        with self.motors.bm_calf_gain_override(float(kp_matrix[0, 2]), float(kd_matrix[0, 2])):
            for leg in range(4):
                for joint in range(3):
                    self.motors.send_mit(
                        leg,
                        joint,
                        MitCommand(
                            pos=float(target[leg, joint]),
                            vel=0.0,
                            kp=float(kp_matrix[leg, joint]),
                            kd=float(kd_matrix[leg, joint]),
                            tor=0.0,
                        ),
                    )

    def _hold_startup_pose_after_run(self, period: float) -> None:
        if self.hold_after_run <= 0 or self.motors is None or self.startup_result is None:
            return
        print("holding startup pose after run for %.1fs" % self.hold_after_run, flush=True)
        deadline = time.monotonic() + self.hold_after_run
        while time.monotonic() < deadline:
            t0 = time.monotonic()
            self.motors.poll()
            self._send_pose(self.startup_result.startup_pose, kp=STARTUP_KP, kd=STARTUP_KD)
            elapsed = time.monotonic() - t0
            if period > elapsed:
                time.sleep(period - elapsed)


def check_hardware() -> int:
    print("Checking IMU...")
    with ImuSensor(IMU_PORT, IMU_BAUDRATE) as imu:
        reading = imu.read_complete(timeout=2.0)
        if reading is None:
            raise RuntimeError("no complete IMU sample")
        print("  IMU accel=%s gyro=%s rpy_deg=%s" % (_fmt_tuple(reading.accel), _fmt_tuple(reading.gyro), _fmt_tuple(reading.euler_deg)))

    print("Checking USB2FDCAN devices...")
    with Usb2CanHub(CAN_SERIALS, CAN_CHANNELS, CAN_BITRATE, CANFD_ENABLED) as can:
        for index, serial, channels in can.describe():
            print("  device[%d] serial=%s channels=%s bitrate=%d" % (index, serial, channels, CAN_BITRATE))
    return 0


def _fmt_tuple(values: Optional[Sequence[float]]) -> str:
    if values is None:
        return "None"
    return "(" + ", ".join("%.3f" % float(v) for v in values) + ")"


def _print_matrix(values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32).reshape(4, 3)
    names = ("FL", "FR", "RL", "RR")
    for leg, name in enumerate(names):
        print(
            "    %s hip=% .4f thigh=% .4f calf=% .4f"
            % (name, values[leg, 0], values[leg, 1], values[leg, 2]),
            flush=True,
        )


def _print_python_matrix(values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32).reshape(4, 3)
    print("[", flush=True)
    for row in values:
        print("    [% .6f, % .6f, % .6f]," % (row[0], row[1], row[2]), flush=True)
    print("]", flush=True)


def _print_python_list(values: np.ndarray) -> None:
    print(_python_list_text(values), flush=True)


def _python_list_text(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    return "[" + ", ".join("%.6f" % float(v) for v in values) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RL quadruped deployment entry point.")
    parser.add_argument("--model", default=str(Path(__file__).with_name("policy.pt")))
    parser.add_argument("--yaw-model", default="", help="optional TorchScript model used for pure yaw commands")
    parser.add_argument("--check-hardware", action="store_true")
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--no-can", action="store_true")
    parser.add_argument("--startup", action="store_true", help="run the STM32 dog_start_init sequence on the motors")
    parser.add_argument("--auto-rl-after-start", action="store_true", help="switch to policy output after startup; remote switch will replace this later")
    parser.add_argument("--send-actions", action="store_true", help="send policy MIT commands after RL is active")
    parser.add_argument("--startup-speed-scale", type=float, default=1.0, help=">1 runs startup faster, <1 slower")
    parser.add_argument("--startup-feedback-timeout", type=float, default=2.0, help="seconds to wait for all motor feedback before startup")
    parser.add_argument("--feedback-timeout", type=float, default=0.5, help="stop and disable if no motor feedback arrives during RL for this many seconds; <=0 disables the guard")
    parser.add_argument("--rl-action-scale", type=float, default=RL_ACTION_SCALE, help="policy action scale in rad per action unit")
    parser.add_argument("--action-clip", type=float, default=0.0, help="clip raw policy actions before scaling; <=0 disables clipping")
    parser.add_argument("--print-full-action", action="store_true", help="print full 12D action, model-space feedback, and raw targets")
    parser.add_argument("--hold-after-run", type=float, default=0.0, help="return to startup pose and hold this many seconds before disabling")
    parser.set_defaults(shutdown_lie_down=True)
    parser.add_argument("--shutdown-lie-down", dest="shutdown_lie_down", action="store_true", help="replay startup in reverse before disabling motors")
    parser.add_argument("--no-shutdown-lie-down", dest="shutdown_lie_down", action="store_false", help="disable motors without replaying startup in reverse")
    parser.add_argument("--shutdown-lie-down-hold", type=float, default=0.25, help="hold the final lying pose before disabling")
    parser.add_argument("--policy-switch-pause-duration", type=float, default=0.5, help="seconds to pause policy output after switching active models")
    parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "YAW"), help="physical command: forward m/s, lateral m/s, yaw rad/s")
    parser.add_argument("--repeat-current-history", action="store_true", help="old behavior: prefill all history frames with the current standing observation")
    parser.add_argument("--startup-hold-before-rl", type=float, default=3.0, help="hold startup pose and latch encoder origin before switching to RL")
    parser.set_defaults(switch_mit_mode=True)
    parser.add_argument("--switch-mit-mode", dest="switch_mit_mode", action="store_true", help="send MIT mode switch frames before startup")
    parser.add_argument("--no-switch-mit-mode", dest="switch_mit_mode", action="store_false", help="skip MIT mode switch frames before startup")
    parser.add_argument("--allow-missing-feedback", action="store_true")
    parser.add_argument("--keyboard-control", action="store_true", help="read W/S/A/D or arrow keys as RL velocity commands after startup")
    parser.add_argument("--keyboard-speed", type=float, default=1.0, help="keyboard command speed in m/s")
    parser.add_argument("--gamepad-control", action="store_true", help="read an Xbox-style gamepad as RL velocity commands after startup")
    parser.add_argument("--gamepad-device", help="Linux joystick device path, default: first /dev/input/js* or /dev/input/js0")
    parser.add_argument("--gamepad-linear-speed", type=float, default=1.0, help="max forward/lateral gamepad command in m/s")
    parser.add_argument("--gamepad-yaw-speed", type=float, default=1.0, help="max yaw gamepad command in rad/s")
    parser.add_argument("--gamepad-deadzone", type=float, default=0.12, help="normalized stick deadzone")
    parser.add_argument("--gamepad-forward-axis", type=int, default=1, help="joystick axis index for forward/back, default Xbox left stick Y")
    parser.add_argument("--gamepad-lateral-axis", type=int, default=0, help="joystick axis index for left/right, default Xbox left stick X")
    parser.add_argument("--gamepad-yaw-axis", type=int, default=3, help="joystick axis index for yaw, default Xbox right stick X")
    parser.add_argument("--gamepad-forward-sign", type=float, default=-1.0, help="axis sign for forward command")
    parser.add_argument("--gamepad-lateral-sign", type=float, default=-1.0, help="axis sign for lateral command")
    parser.add_argument("--gamepad-yaw-sign", type=float, default=-1.0, help="axis sign for yaw command")
    parser.add_argument("--gamepad-deadman-button", type=int, default=-1, help="button index that must be held to output commands; -1 disables")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument("--steps", type=int, default=100, help="0 means run forever")
    parser.add_argument("--print-every", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.keyboard_control and args.gamepad_control:
        raise ValueError("use only one command source: --keyboard-control or --gamepad-control")
    if args.check_hardware:
        return check_hardware()

    runner = DeploymentRunner(
        args.model,
        yaw_model_path=args.yaw_model or None,
        use_imu=not args.no_imu,
        use_can=not args.no_can,
        startup=args.startup,
        auto_rl_after_start=args.auto_rl_after_start,
        send_actions=args.send_actions,
        loop_hz=args.hz,
        startup_speed_scale=args.startup_speed_scale,
        require_motor_feedback=not args.allow_missing_feedback,
        startup_feedback_timeout=args.startup_feedback_timeout,
        feedback_timeout=args.feedback_timeout,
        rl_action_scale=args.rl_action_scale,
        action_clip=args.action_clip,
        print_full_action=args.print_full_action,
        hold_after_run=args.hold_after_run,
        command=args.command,
        play_history_bootstrap=not args.repeat_current_history,
        startup_hold_before_rl=args.startup_hold_before_rl,
        switch_mit_mode=args.switch_mit_mode,
        keyboard_control=args.keyboard_control,
        keyboard_speed=args.keyboard_speed,
        gamepad_control=args.gamepad_control,
        gamepad_device=args.gamepad_device,
        gamepad_linear_speed=args.gamepad_linear_speed,
        gamepad_yaw_speed=args.gamepad_yaw_speed,
        gamepad_deadzone=args.gamepad_deadzone,
        gamepad_forward_axis=args.gamepad_forward_axis,
        gamepad_lateral_axis=args.gamepad_lateral_axis,
        gamepad_yaw_axis=args.gamepad_yaw_axis,
        gamepad_forward_sign=args.gamepad_forward_sign,
        gamepad_lateral_sign=args.gamepad_lateral_sign,
        gamepad_yaw_sign=args.gamepad_yaw_sign,
        gamepad_deadman_button=args.gamepad_deadman_button,
        shutdown_lie_down=args.shutdown_lie_down,
        shutdown_lie_down_hold=args.shutdown_lie_down_hold,
        policy_switch_pause_duration=args.policy_switch_pause_duration,
    )
    try:
        runner.open()
        runner.run(steps=args.steps, print_every=args.print_every)
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
