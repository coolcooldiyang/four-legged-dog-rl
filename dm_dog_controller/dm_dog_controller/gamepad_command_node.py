#!/usr/bin/env python3
from __future__ import annotations

from typing import List

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from std_msgs.msg import Int32MultiArray


class GamepadCommandNode(Node):
    def __init__(self) -> None:
        super().__init__("dm_dog_gamepad_command_node")
        self.declare_parameter("joy_topic", "joy")
        self.declare_parameter("cmd_vel_topic", "joy_cmd")
        self.declare_parameter("buttons_topic", "joy_buttons")
        self.declare_parameter("linear_speed", 1.0)
        self.declare_parameter("yaw_speed", 1.0)
        self.declare_parameter("deadzone", 0.12)
        self.declare_parameter("publish_zero_on_release", True)
        self.declare_parameter("forward_axis", 1)
        self.declare_parameter("lateral_axis", 0)
        self.declare_parameter("yaw_axis", 3)
        self.declare_parameter("forward_sign", 1.0)
        self.declare_parameter("lateral_sign", 1.0)
        self.declare_parameter("yaw_sign", 1.0)
        self.declare_parameter("deadman_button", -1)
        self.declare_parameter("stop_button", 0)
        self.declare_parameter("hold_topic", "/dog_hold")
        self.declare_parameter("emergency_stop_topic", "/emergency_stop")
        self.declare_parameter("emergency_lb_button", 4)
        self.declare_parameter("emergency_rb_button", 5)
        self.declare_parameter("emergency_a_button", 0)

        self._cmd_pub = self.create_publisher(Twist, str(self.get_parameter("cmd_vel_topic").value), 10)
        self._buttons_pub = self.create_publisher(Int32MultiArray, str(self.get_parameter("buttons_topic").value), 10)
        emergency_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._emergency_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("emergency_stop_topic").value),
            emergency_qos,
        )
        # Non-latching stop: while the stop button is held the policy holds the stand
        # pose instead of applying policy output; releasing clears the request and the
        # next motion command resumes walking.
        self._hold_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("hold_topic").value),
            emergency_qos,
        )
        self.create_subscription(Joy, str(self.get_parameter("joy_topic").value), self._on_joy, 10)
        self._was_moving = False
        self._emergency_was_pressed = False
        self._emergency_latched = False
        self._stop_was_pressed = False

    def _on_joy(self, msg: Joy) -> None:
        self._buttons_pub.publish(Int32MultiArray(data=[int(v) for v in msg.buttons]))

        twist = Twist()
        emergency_pressed = self._emergency_combo_pressed(msg.buttons)
        emergency_rising = emergency_pressed and not self._emergency_was_pressed
        self._emergency_was_pressed = emergency_pressed
        if emergency_rising and not self._emergency_latched:
            self._emergency_latched = True
            self._emergency_pub.publish(Bool(data=True))
            self._cmd_pub.publish(twist)
            self._was_moving = False
            self.get_logger().fatal("EMERGENCY STOP requested by LB+RB+A")
            return
        if self._emergency_latched:
            self._cmd_pub.publish(twist)
            self._was_moving = False
            return

        if self._button(msg.buttons, self._param_int("stop_button")):
            self._cmd_pub.publish(twist)
            self._was_moving = False
            if not self._stop_was_pressed:
                self._stop_was_pressed = True
                self._hold_pub.publish(Bool(data=True))
                self.get_logger().warn("STOP pressed: zero velocity, holding the stand pose")
            return
        if self._stop_was_pressed:
            self._stop_was_pressed = False
            self._hold_pub.publish(Bool(data=False))
        deadman = self._param_int("deadman_button")
        if deadman >= 0 and not self._button(msg.buttons, deadman):
            if self._was_moving and bool(self.get_parameter("publish_zero_on_release").value):
                self._cmd_pub.publish(twist)
            self._was_moving = False
            return

        twist.linear.x = self._param_float("forward_sign") * self._axis(msg.axes, self._param_int("forward_axis")) * self._param_float("linear_speed")
        twist.linear.y = self._param_float("lateral_sign") * self._axis(msg.axes, self._param_int("lateral_axis")) * self._param_float("linear_speed")
        twist.angular.z = self._param_float("yaw_sign") * self._axis(msg.axes, self._param_int("yaw_axis")) * self._param_float("yaw_speed")
        moving = self._twist_has_motion(twist)
        if moving or (self._was_moving and bool(self.get_parameter("publish_zero_on_release").value)):
            self._cmd_pub.publish(twist)
        self._was_moving = moving

    @staticmethod
    def _twist_has_motion(twist: Twist) -> bool:
        return abs(twist.linear.x) > 1e-6 or abs(twist.linear.y) > 1e-6 or abs(twist.angular.z) > 1e-6

    def _axis(self, axes: List[float], index: int) -> float:
        if index < 0 or index >= len(axes):
            return 0.0
        value = max(-1.0, min(1.0, float(axes[index])))
        deadzone = max(0.0, min(0.95, self._param_float("deadzone")))
        magnitude = abs(value)
        if magnitude <= deadzone:
            return 0.0
        sign = 1.0 if value >= 0.0 else -1.0
        return sign * ((magnitude - deadzone) / (1.0 - deadzone))

    def _button(self, buttons: List[int], index: int) -> bool:
        return index >= 0 and index < len(buttons) and bool(buttons[index])

    def _emergency_combo_pressed(self, buttons: List[int]) -> bool:
        return (
            self._button(buttons, self._param_int("emergency_lb_button")) and
            self._button(buttons, self._param_int("emergency_rb_button")) and
            self._button(buttons, self._param_int("emergency_a_button"))
        )

    def _param_int(self, name: str) -> int:
        return int(self.get_parameter(name).value)

    def _param_float(self, name: str) -> float:
        return float(self.get_parameter(name).value)


def main() -> int:
    rclpy.init()
    node = GamepadCommandNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
