#!/usr/bin/env python3
from __future__ import annotations

from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import SetBool


class GamepadPostureToggleNode(Node):
    def __init__(self) -> None:
        super().__init__("dm_dog_gamepad_posture_toggle_node")
        self.declare_parameter("buttons_topic", "joy_buttons")
        self.declare_parameter("posture_service", "dog_posture")
        self.declare_parameter("modifier_button", 4)
        self.declare_parameter("action_button", 0)
        self.declare_parameter("blocker_button", 5)
        self.declare_parameter("initial_standing", False)

        self._standing = bool(self.get_parameter("initial_standing").value)
        self._combo_was_pressed = False
        self._request_in_flight = False
        self._client = self.create_client(SetBool, str(self.get_parameter("posture_service").value))
        self.create_subscription(Int32MultiArray, str(self.get_parameter("buttons_topic").value), self._on_buttons, 10)

    def _on_buttons(self, msg: Int32MultiArray) -> None:
        combo_pressed = (
            self._button(msg.data, self._param_int("modifier_button")) and
            self._button(msg.data, self._param_int("action_button")) and
            not self._button(msg.data, self._param_int("blocker_button"))
        )
        rising_edge = combo_pressed and not self._combo_was_pressed
        self._combo_was_pressed = combo_pressed
        if not rising_edge or self._request_in_flight:
            return

        target_standing = not self._standing
        if not self._client.service_is_ready():
            self.get_logger().warn("posture service is not ready")
            return

        request = SetBool.Request()
        request.data = target_standing
        self._request_in_flight = True
        future = self._client.call_async(request)
        future.add_done_callback(lambda fut: self._on_response(fut, target_standing))

    def _on_response(self, future, target_standing: bool) -> None:
        self._request_in_flight = False
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error("posture service call failed: %s" % exc)
            return
        if response.success:
            self._standing = target_standing
            self.get_logger().info(response.message)
        else:
            self.get_logger().warn(response.message)

    def _button(self, buttons: List[int], index: int) -> bool:
        return index >= 0 and index < len(buttons) and bool(buttons[index])

    def _param_int(self, name: str) -> int:
        return int(self.get_parameter(name).value)


def main() -> int:
    rclpy.init()
    node = GamepadPostureToggleNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
