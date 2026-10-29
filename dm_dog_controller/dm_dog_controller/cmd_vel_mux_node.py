#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

from dm_dog_controller.cmd_vel_shaping import continuous_speed_mode, shape_joy_cmd


class CmdVelMuxNode(Node):
    def __init__(self) -> None:
        super().__init__("dm_dog_cmd_vel_mux_node")
        self.declare_parameter("joy_cmd_topic", "joy_cmd")
        self.declare_parameter("aim_cmd_topic", "aim_cmd")
        self.declare_parameter("cmd_vel_topic", "cmd_vel")
        self.declare_parameter("publish_rate", 50.0)
        self.declare_parameter("joy_active_timeout", 0.25)
        self.declare_parameter("aim_timeout", 0.5)
        self.declare_parameter("motion_epsilon", 1e-4)
        self.declare_parameter("min_output_linear_x", 0.3)
        self.declare_parameter("min_output_linear_y", 0.2)
        self.declare_parameter("min_output_angular_z", 0.5)
        self.declare_parameter("rotation_blocks_translation", True)
        self.declare_parameter("rotation_block_epsilon", 1e-4)
        self.declare_parameter("forward_blocks_lateral", True)
        self.declare_parameter("forward_block_epsilon", 1e-4)
        # Forward/backward stick handling.
        #   "fixed"      - legacy behaviour: any non-zero stick becomes a constant
        #                  +-fixed_forward_speed (the trained policy of that era only
        #                  knew one speed).
        #   "continuous" - the stick magnitude is mapped linearly onto
        #                  [0, continuous_forward_max] m/s, which is what a
        #                  velocity-tracking policy (unitree_rl_gym) needs. Lateral
        #                  and yaw are scaled the same way so the commanded values
        #                  stay inside the range the policy was trained on.
        self.declare_parameter("forward_speed_mode", "fixed")
        self.declare_parameter("fixed_forward_speed", 0.5)
        self.declare_parameter("continuous_forward_max", 0.6)
        # 2026-10-15：前进/后退分别限幅。linear_x>0 用 forward_max，
        # linear_x<0 用 backward_max。后退更慢更保守（看不清后方且易后仰摔倒）。
        self.declare_parameter("continuous_backward_max", 0.3)
        self.declare_parameter("continuous_lateral_max", 0.4)
        self.declare_parameter("continuous_yaw_max", 1.0)

        self.last_joy_cmd: Optional[Twist] = None
        self.last_aim_cmd: Optional[Twist] = None
        self.last_joy_motion_time: Optional[float] = None
        self.last_aim_time: Optional[float] = None

        self.cmd_pub = self.create_publisher(Twist, str(self.get_parameter("cmd_vel_topic").value), 10)
        self.create_subscription(Twist, str(self.get_parameter("joy_cmd_topic").value), self._on_joy_cmd, 10)
        self.create_subscription(Twist, str(self.get_parameter("aim_cmd_topic").value), self._on_aim_cmd, 10)

        rate = float(self.get_parameter("publish_rate").value)
        period = 1.0 / rate if rate > 0.0 else 0.02
        self.create_timer(period, self._publish_selected_cmd)
        self.get_logger().info(
            "cmd_vel mux ready: joy=%s aim=%s output=%s speed_mode=%s"
            % (
                str(self.get_parameter("joy_cmd_topic").value),
                str(self.get_parameter("aim_cmd_topic").value),
                str(self.get_parameter("cmd_vel_topic").value),
                "continuous (fwd<=%.2f / back<=%.2f m/s, lat<=%.2f m/s, yaw<=%.2f rad/s)"
                % (
                    float(self.get_parameter("continuous_forward_max").value),
                    float(self.get_parameter("continuous_backward_max").value),
                    float(self.get_parameter("continuous_lateral_max").value),
                    float(self.get_parameter("continuous_yaw_max").value),
                )
                if self._continuous_speed()
                else "fixed (fwd=+-%.2f m/s)" % float(self.get_parameter("fixed_forward_speed").value),
            )
        )

    def _on_joy_cmd(self, msg: Twist) -> None:
        shaped = shape_joy_cmd(
            float(msg.linear.x),
            float(msg.linear.y),
            float(msg.angular.z),
            mode=str(self.get_parameter("forward_speed_mode").value),
            fixed_speed=float(self.get_parameter("fixed_forward_speed").value),
            forward_max=float(self.get_parameter("continuous_forward_max").value),
            backward_max=float(self.get_parameter("continuous_backward_max").value),
            lateral_max=float(self.get_parameter("continuous_lateral_max").value),
            yaw_max=float(self.get_parameter("continuous_yaw_max").value),
        )
        msg.linear.x, msg.linear.y, msg.angular.z = shaped
        self.last_joy_cmd = msg
        
        if self._has_motion(msg):
            self.last_joy_motion_time = self._now_sec()

    def _continuous_speed(self) -> bool:
        return continuous_speed_mode(str(self.get_parameter("forward_speed_mode").value))

    def _on_aim_cmd(self, msg: Twist) -> None:
        self.last_aim_cmd = msg
        self.last_aim_time = self._now_sec()

    def _publish_selected_cmd(self) -> None:
        now = self._now_sec()
        if self._joy_is_active(now) and self.last_joy_cmd is not None:
            self.cmd_pub.publish(self._shape_output_cmd(self.last_joy_cmd))
            return
        if self._aim_is_fresh(now) and self.last_aim_cmd is not None:
            self.cmd_pub.publish(self._shape_output_cmd(self.last_aim_cmd))
            return
        self.cmd_pub.publish(Twist())

    def _joy_is_active(self, now: float) -> bool:
        if self.last_joy_motion_time is None:
            return False
        timeout = max(0.0, float(self.get_parameter("joy_active_timeout").value))
        return now - self.last_joy_motion_time <= timeout

    def _aim_is_fresh(self, now: float) -> bool:
        if self.last_aim_time is None:
            return False
        timeout = max(0.0, float(self.get_parameter("aim_timeout").value))
        return timeout <= 0.0 or now - self.last_aim_time <= timeout

    def _has_motion(self, msg: Twist) -> bool:
        eps = max(0.0, float(self.get_parameter("motion_epsilon").value))
        values = (
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.x,
            msg.angular.y,
            msg.angular.z,
        )
        return any(math.fabs(float(value)) > eps for value in values)

    def _shape_output_cmd(self, msg: Twist) -> Twist:
        shaped = Twist()
        shaped.linear.x = self._apply_min_abs(
            float(msg.linear.x),
            max(0.0, float(self.get_parameter("min_output_linear_x").value)),
        )
        shaped.linear.y = self._apply_min_abs(
            float(msg.linear.y),
            max(0.0, float(self.get_parameter("min_output_linear_y").value)),
        )
        shaped.linear.z = msg.linear.z
        shaped.angular.x = msg.angular.x
        shaped.angular.y = msg.angular.y
        shaped.angular.z = self._apply_min_abs(
            float(msg.angular.z),
            max(0.0, float(self.get_parameter("min_output_angular_z").value)),
        )
        if self._rotation_blocks_translation(shaped):
            shaped.linear.x = 0.0
            shaped.linear.y = 0.0
        if self._forward_blocks_lateral(shaped):
            shaped.linear.y = 0.0
        return shaped

    def _rotation_blocks_translation(self, msg: Twist) -> bool:
        if not bool(self.get_parameter("rotation_blocks_translation").value):
            return False
        eps = max(0.0, float(self.get_parameter("rotation_block_epsilon").value))
        return math.fabs(float(msg.angular.z)) > eps

    def _forward_blocks_lateral(self, msg: Twist) -> bool:
        if not bool(self.get_parameter("forward_blocks_lateral").value):
            return False
        eps = max(0.0, float(self.get_parameter("forward_block_epsilon").value))
        return float(msg.linear.x) > eps

    @staticmethod
    def _apply_min_abs(value: float, minimum: float) -> float:
        if value == 0.0 or minimum <= 0.0:
            return value
        if math.fabs(value) < minimum:
            return math.copysign(minimum, value)
        return value

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9


def main() -> int:
    rclpy.init()
    node = CmdVelMuxNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
