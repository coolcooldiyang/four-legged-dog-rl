#!/usr/bin/env python3
from __future__ import annotations

import time

import rclpy
from rclpy.node import Node

from bs_hardware.can_hub import Usb2CanHub
from bs_hardware.config import CAN_BITRATE, CAN_BUS_MAP, CAN_CHANNELS, CAN_SERIALS, CANFD_ENABLED
from bs_hardware.motor_bus import MotorBus


class MotorFeedbackCheckNode(Node):
    def __init__(self) -> None:
        super().__init__("dm_dog_motor_feedback_check_node")
        self.declare_parameter("switch_mit_mode", True)
        self.declare_parameter("query_rounds", 3)
        self.declare_parameter("wait_timeout", 1.0)
        self.declare_parameter("print_raw_idle_frames", False)
        self.declare_parameter("idle_drain_seconds", 0.5)

    def run(self) -> int:
        self.get_logger().info("opening DaMiao USB2FDCAN devices")
        with Usb2CanHub(CAN_SERIALS, CAN_CHANNELS, CAN_BITRATE, CANFD_ENABLED) as can:
            for index, serial, channels in can.describe():
                self.get_logger().info("device[%d] serial=%s channels=%s bitrate=%d" % (index, serial, channels, CAN_BITRATE))
            self._print_bus_map()

            if bool(self.get_parameter("print_raw_idle_frames").value):
                self._drain_idle_frames(can)

            motors = MotorBus(can, CAN_BUS_MAP)
            if bool(self.get_parameter("switch_mit_mode").value):
                self.get_logger().info("sending MIT mode switch frames")
                motors.switch_mit_mode_all()

            motors.clear_feedback_history()
            rounds = max(1, int(self.get_parameter("query_rounds").value))
            for index in range(rounds):
                self.get_logger().info("sending zero-query round %d/%d" % (index + 1, rounds))
                motors.send_zero_query_all()
                time.sleep(0.05)
                motors.poll(limit=4096)

            timeout = max(0.0, float(self.get_parameter("wait_timeout").value))
            got_all = motors.wait_for_feedback(timeout=timeout, require_all=True)
            self._print_feedback(motors)
            if not got_all:
                self.get_logger().error("missing feedback: %s" % (motors.missing_feedback(),))
                return 2

        self.get_logger().info("all 12 motor feedback frames were received")
        return 0

    def _print_bus_map(self) -> None:
        for leg, item in enumerate(CAN_BUS_MAP):
            self.get_logger().info("leg%d -> serial=%s channel=%d motor_ids=1,2,3" % (leg, item[0], item[1]))

    def _drain_idle_frames(self, can: Usb2CanHub) -> None:
        deadline = time.monotonic() + max(0.0, float(self.get_parameter("idle_drain_seconds").value))
        seen = 0
        while time.monotonic() < deadline:
            for frame in can.drain(limit=256):
                seen += 1
                self.get_logger().info(
                    "idle frame serial=%s ch=%d id=0x%X data=%s tx_echo=%s"
                    % (frame.serial, frame.channel, frame.can_id, frame.data.hex(" "), frame.is_tx_echo)
                )
            time.sleep(0.01)
        self.get_logger().info("idle raw frame count=%d" % seen)

    def _print_feedback(self, motors: MotorBus) -> None:
        for leg in range(4):
            for joint in range(3):
                fb = motors.feedback[(leg, joint)]
                if fb.seen:
                    self.get_logger().info(
                        "leg%d joint%d id=%d state=%d pos=%.4f vel=%.4f tor=%.4f t_mos=%.1f t_coil=%.1f"
                        % (leg, joint, fb.motor_id, fb.state, fb.pos, fb.vel, fb.tor, fb.t_mos, fb.t_coil)
                    )
                else:
                    self.get_logger().warn("leg%d joint%d feedback missing" % (leg, joint))
        self.get_logger().info("rejected_feedback=%d rejected_feedback_id=%d" % (motors.rejected_feedback, motors.rejected_feedback_id))


def main() -> int:
    rclpy.init()
    node = MotorFeedbackCheckNode()
    try:
        return node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
