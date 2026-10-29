#!/usr/bin/env python3
from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Int32MultiArray
from std_srvs.srv import SetBool

from dm_dog_controller.deployment import DeploymentRunner, check_hardware


def _normalize_head_order(value) -> Optional[str]:
    """把 obs_head_order 参数统一成字符串。

    注意：ROS 2 CLI 里写 `-p obs_head_order:=gravity,cmd,ang_vel` 时，逗号会被当成
    数组分隔符，参数会变成 ['gravity','cmd','ang_vel']；launch 里写字符串则正常。
    两种都收，避免"参数看起来给了、其实没用上"。
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        joined = ",".join(str(item) for item in value)
    else:
        joined = str(value)
    joined = joined.strip()
    return joined or None


class RosCommandSource:
    def __init__(
        self,
        node: Node,
        *,
        cmd_vel_topic: str,
        buttons_topic: str,
        command_timeout: float,
        quit_button: int,
        min_linear_x: float,
        min_linear_y: float,
        min_angular_z: float,
        max_linear_x: float,
        max_linear_y: float,
        max_angular_z: float,
        input_deadband: float,
        process_callbacks: bool = True,
    ) -> None:
        self.node = node
        self.command_timeout = max(0.0, float(command_timeout))
        self.quit_button = int(quit_button)
        self.min_linear_x = max(0.0, float(min_linear_x))
        self.min_linear_y = max(0.0, float(min_linear_y))
        self.min_angular_z = max(0.0, float(min_angular_z))
        self.max_linear_x = max(self.min_linear_x, float(max_linear_x))
        self.max_linear_y = max(self.min_linear_y, float(max_linear_y))
        self.max_angular_z = max(self.min_angular_z, float(max_angular_z))
        self.input_deadband = max(0.0, float(input_deadband))
        self.process_callbacks = bool(process_callbacks)
        self.quit_requested = False
        self._command = np.zeros(3, dtype=np.float32)
        self._last_command_time = node.get_clock().now()
        self._cmd_sub = node.create_subscription(Twist, cmd_vel_topic, self._on_cmd_vel, 10)
        self._button_sub = node.create_subscription(Int32MultiArray, buttons_topic, self._on_buttons, 10)

    def open(self) -> None:
        self.node.get_logger().info("using ROS cmd_vel command source")

    def close(self) -> None:
        pass

    def poll(self) -> np.ndarray:
        if self.process_callbacks:
            rclpy.spin_once(self.node, timeout_sec=0.0)
        if self.command_timeout > 0.0:
            age = (self.node.get_clock().now() - self._last_command_time).nanoseconds * 1e-9
            if age > self.command_timeout:
                self._command[:] = 0.0
        return self._command.copy()

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._command[:] = (
            self._shape_axis(float(msg.linear.x), self.min_linear_x, self.max_linear_x),
            self._shape_axis(float(msg.linear.y), self.min_linear_y, self.max_linear_y),
            self._shape_axis(float(msg.angular.z), self.min_angular_z, self.max_angular_z),
        )
        self._last_command_time = self.node.get_clock().now()

    def _shape_axis(self, value: float, min_abs: float, max_abs: float) -> float:
        magnitude = abs(float(value))
        if magnitude <= self.input_deadband:
            return 0.0
        magnitude = min(1.0, magnitude)
        shaped = min_abs + (max_abs - min_abs) * magnitude
        return float(np.sign(value) * shaped)

    def _on_buttons(self, msg: Int32MultiArray) -> None:
        if self.quit_button >= 0 and self.quit_button < len(msg.data) and bool(msg.data[self.quit_button]):
            self.quit_requested = True


class DogPolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("dm_dog_policy_node")
        share_dir = get_package_share_directory("dm_dog_controller")

        self._share_dir = share_dir
        self.declare_parameter("model", f"{share_dir}/policy_m.pt")
        self.declare_parameter("use_dual_policy", False)
        self.declare_parameter("yaw_model", f"{share_dir}/policy.pt")
        # 观测帧前 9 维的顺序（必须与训练时一致）。空串=用 config.py 的 OBS_HEAD_ORDER。
        # 例："gravity,cmd,ang_vel" 是 SB3 那个 go2dog 策略要的顺序。
        self.declare_parameter("obs_head_order", "")
        self.declare_parameter("policy_gain_transition_duration", 0.4)
        self.declare_parameter("policy_switch_pause_duration", 0.5)
        self.declare_parameter("check_hardware", False)
        self.declare_parameter("use_imu", True)
        self.declare_parameter("use_can", True)
        self.declare_parameter("startup", True)
        self.declare_parameter("auto_rl_after_start", False)
        self.declare_parameter("send_actions", True)
        self.declare_parameter("hz", 50.0)
        self.declare_parameter("steps", 0)
        self.declare_parameter("print_every", 10)
        self.declare_parameter("startup_speed_scale", 1.0)
        self.declare_parameter("allow_missing_feedback", False)
        self.declare_parameter("startup_feedback_timeout", 2.0)
        self.declare_parameter("feedback_timeout", 0.5)
        self.declare_parameter("rl_action_scale", 0.25)
        self.declare_parameter("action_clip", 1.0)
        self.declare_parameter("print_full_action", False)
        self.declare_parameter("hold_after_run", 5.0)
        self.declare_parameter("shutdown_lie_down", True)
        self.declare_parameter("shutdown_lie_down_hold", 0.25)
        self.declare_parameter("command", [0.0, 0.0, 0.0])
        self.declare_parameter("repeat_current_history", False)
        self.declare_parameter("startup_hold_before_rl", 3.0)
        self.declare_parameter("use_startup_pose_as_rl_origin", False)
        self.declare_parameter("switch_mit_mode", True)
        self.declare_parameter("keyboard_control", False)
        self.declare_parameter("keyboard_speed", 1.0)
        self.declare_parameter("gamepad_control", False)
        self.declare_parameter("gamepad_device", "")
        self.declare_parameter("gamepad_linear_speed", 1.0)
        self.declare_parameter("gamepad_yaw_speed", 1.0)
        self.declare_parameter("gamepad_deadzone", 0.12)
        self.declare_parameter("gamepad_forward_axis", 1)
        self.declare_parameter("gamepad_lateral_axis", 0)
        self.declare_parameter("gamepad_yaw_axis", 3)
        self.declare_parameter("gamepad_forward_sign", -1.0)
        self.declare_parameter("gamepad_lateral_sign", -1.0)
        self.declare_parameter("gamepad_yaw_sign", -1.0)
        self.declare_parameter("gamepad_deadman_button", -1)
        self.declare_parameter("ros_command_control", True)
        self.declare_parameter("cmd_vel_topic", "cmd_vel")
        self.declare_parameter("buttons_topic", "joy_buttons")
        self.declare_parameter("command_timeout", 0.5)
        self.declare_parameter("cmd_min_linear_x", 0.22)
        self.declare_parameter("cmd_min_linear_y", 0.33)
        self.declare_parameter("cmd_min_angular_z", 0.30)
        self.declare_parameter("cmd_max_linear_x", 1.0)
        self.declare_parameter("cmd_max_linear_y", 1.0)
        self.declare_parameter("cmd_max_angular_z", 1.0)
        self.declare_parameter("cmd_input_deadband", 1e-4)
        self.declare_parameter("quit_button", 6)
        self.declare_parameter("posture_service", "dog_posture")
        self.declare_parameter("emergency_stop_topic", "/emergency_stop")
        self.declare_parameter("hold_topic", "/dog_hold")
        self.runner: Optional[DeploymentRunner] = None
        self._callback_group = ReentrantCallbackGroup()
        self._executor: Optional[MultiThreadedExecutor] = None
        self._executor_thread: Optional[threading.Thread] = None
        emergency_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._emergency_sub = self.create_subscription(
            Bool,
            str(self.get_parameter("emergency_stop_topic").value),
            self._on_emergency_stop,
            emergency_qos,
            callback_group=self._callback_group,
        )
        # Operator stop (gamepad stop button): hold the stand pose instead of
        # applying policy output. Non-latching - a new motion command resumes.
        self._hold_sub = self.create_subscription(
            Bool,
            str(self.get_parameter("hold_topic").value),
            self._on_hold,
            emergency_qos,
            callback_group=self._callback_group,
        )

    def run(self) -> int:
        if self._param_bool("check_hardware"):
            return check_hardware()

        keyboard_control = self._param_bool("keyboard_control")
        gamepad_control = self._param_bool("gamepad_control")
        ros_command_control = self._param_bool("ros_command_control")
        enabled_sources = sum((keyboard_control, gamepad_control, ros_command_control))
        if enabled_sources > 1:
            raise ValueError("use only one command source: keyboard_control, gamepad_control, or ros_command_control")

        command = [float(v) for v in self.get_parameter("command").value]
        gamepad_device = self._param_optional_string("gamepad_device")
        command_source = None
        if ros_command_control:
            command_source = RosCommandSource(
                self,
                cmd_vel_topic=str(self.get_parameter("cmd_vel_topic").value),
                buttons_topic=str(self.get_parameter("buttons_topic").value),
                command_timeout=self._param_float("command_timeout"),
                quit_button=self._param_int("quit_button"),
                min_linear_x=self._param_float("cmd_min_linear_x"),
                min_linear_y=self._param_float("cmd_min_linear_y"),
                min_angular_z=self._param_float("cmd_min_angular_z"),
                max_linear_x=self._param_float("cmd_max_linear_x"),
                max_linear_y=self._param_float("cmd_max_linear_y"),
                max_angular_z=self._param_float("cmd_max_angular_z"),
                input_deadband=self._param_float("cmd_input_deadband"),
                process_callbacks=False,
            )
        self.runner = DeploymentRunner(
            self._resolve_model_path(str(self.get_parameter("model").value)),
            yaw_model_path=(
                self._resolve_model_path(str(self.get_parameter("yaw_model").value))
                if self._param_bool("use_dual_policy")
                else None
            ),
            obs_head_order=_normalize_head_order(self.get_parameter("obs_head_order").value),
            use_imu=self._param_bool("use_imu"),
            use_can=self._param_bool("use_can"),
            startup=self._param_bool("startup"),
            auto_rl_after_start=self._param_bool("auto_rl_after_start"),
            send_actions=self._param_bool("send_actions"),
            loop_hz=self._param_float("hz"),
            startup_speed_scale=self._param_float("startup_speed_scale"),
            require_motor_feedback=not self._param_bool("allow_missing_feedback"),
            startup_feedback_timeout=self._param_float("startup_feedback_timeout"),
            feedback_timeout=self._param_float("feedback_timeout"),
            rl_action_scale=self._param_float("rl_action_scale"),
            action_clip=self._param_float("action_clip"),
            print_full_action=self._param_bool("print_full_action"),
            hold_after_run=self._param_float("hold_after_run"),
            command=command,
            play_history_bootstrap=not self._param_bool("repeat_current_history"),
            startup_hold_before_rl=self._param_float("startup_hold_before_rl"),
            use_startup_pose_as_rl_origin=self._param_bool("use_startup_pose_as_rl_origin"),
            switch_mit_mode=self._param_bool("switch_mit_mode"),
            keyboard_control=keyboard_control,
            keyboard_speed=self._param_float("keyboard_speed"),
            gamepad_control=gamepad_control,
            gamepad_device=gamepad_device,
            gamepad_linear_speed=self._param_float("gamepad_linear_speed"),
            gamepad_yaw_speed=self._param_float("gamepad_yaw_speed"),
            gamepad_deadzone=self._param_float("gamepad_deadzone"),
            gamepad_forward_axis=self._param_int("gamepad_forward_axis"),
            gamepad_lateral_axis=self._param_int("gamepad_lateral_axis"),
            gamepad_yaw_axis=self._param_int("gamepad_yaw_axis"),
            gamepad_forward_sign=self._param_float("gamepad_forward_sign"),
            gamepad_lateral_sign=self._param_float("gamepad_lateral_sign"),
            gamepad_yaw_sign=self._param_float("gamepad_yaw_sign"),
            gamepad_deadman_button=self._param_int("gamepad_deadman_button"),
            command_source=command_source,
            shutdown_lie_down=self._param_bool("shutdown_lie_down"),
            shutdown_lie_down_hold=self._param_float("shutdown_lie_down_hold"),
            policy_gain_transition_duration=self._param_float("policy_gain_transition_duration"),
            policy_switch_pause_duration=self._param_float("policy_switch_pause_duration"),
        )
        self.create_service(
            SetBool,
            str(self.get_parameter("posture_service").value),
            self._on_posture_request,
            callback_group=self._callback_group,
        )
        self.get_logger().info("starting DM dog policy deployment loop")
        self._start_callback_executor()
        try:
            self.runner.open()
            self.runner.run(steps=self._param_int("steps"), print_every=self._param_int("print_every"))
        finally:
            self.runner.close()
            self._stop_callback_executor()
        return 0

    def _start_callback_executor(self) -> None:
        if self._executor is not None:
            return
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self)
        self._executor_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._executor_thread.start()

    def _stop_callback_executor(self) -> None:
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None
        if self._executor_thread is not None:
            self._executor_thread.join(timeout=1.0)
            self._executor_thread = None

    def _on_emergency_stop(self, msg: Bool) -> None:
        if not msg.data:
            return
        self.get_logger().fatal("EMERGENCY STOP received, disabling dog motors")
        if self.runner is not None:
            self.runner.request_emergency_stop()

    def _on_hold(self, msg: Bool) -> None:
        if self.runner is None:
            return
        if bool(msg.data):
            self.runner.request_hold()
            self.get_logger().warn(
                "STOP requested: policy output paused, holding the stand pose "
                "(push the stick to resume)"
            )
        else:
            self.runner.clear_hold()

    def _on_posture_request(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        if self.runner is None:
            response.success = False
            response.message = "policy runner is not initialized"
            return response
        if self.runner.emergency_stop_requested.is_set():
            response.success = False
            response.message = "dog posture blocked by emergency stop"
            return response
        try:
            if request.data:
                self.runner.stand_up()
                response.message = "dog posture: standing"
            else:
                self.runner.lie_down()
                response.message = "dog posture: lying"
        except Exception as exc:
            response.success = False
            response.message = "dog posture action failed: %s" % exc
            return response
        response.success = True
        return response

    def _param_bool(self, name: str) -> bool:
        return bool(self.get_parameter(name).value)

    def _param_int(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _param_float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def _param_optional_string(self, name: str) -> Optional[str]:
        value = str(self.get_parameter(name).value)
        return value if value else None

    def _resolve_model_path(self, value: str) -> str:
        path = value.strip()
        if not path:
            return path
        if path.startswith("/") or "/" in path:
            return path
        return f"{self._share_dir}/{path}"


def main() -> int:
    rclpy.init()
    node = DogPolicyNode()
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
