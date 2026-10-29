#!/usr/bin/env python3
"""test_cmd_vel_shaping.py —— 手柄→速度映射的离线单元测试（不需要 ROS/硬件）。

重点：
  * fixed 模式必须**逐位保持**旧行为（非零摇杆 -> ±0.5，零 -> 零），
    否则改这一处就等于偷偷改了真机行为；
  * continuous 模式要把摇杆幅度线性映射到训练时用的速度区间上限，
    并且夹住越界输入；
  * 非法模式值一律退回 fixed（安全侧）。
"""

from __future__ import annotations

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dm_dog_controller.cmd_vel_shaping import (  # noqa: E402
    continuous_speed_mode, fixed_forward, map_stick, shape_joy_cmd,
)

_results: list[tuple[bool, str, str]] = []


def check(condition: bool, item: str, detail: str = "") -> None:
    _results.append((bool(condition), item, detail))
    print(f"[{' ok ' if condition else 'FAIL'}] {item}" + (f"\n        {detail}" if detail else ""))


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    print("-" * 72)
    print("1) fixed 模式（旧行为，必须逐位一致）")
    check(fixed_forward(0.0, 0.5) == 0.0, "摇杆归零 -> 0（不会残留速度）")
    check(approx(fixed_forward(0.2, 0.5), 0.5), "小幅前推 -> +0.5（旧硬编码值）", f"实际 {fixed_forward(0.2, 0.5)}")
    check(approx(fixed_forward(1.0, 0.5), 0.5), "满推 -> +0.5")
    check(approx(fixed_forward(-0.03, 0.5), -0.5), "小幅后拉 -> -0.5（保留方向）")
    check(approx(fixed_forward(-1.0, 0.5), -0.5), "满拉 -> -0.5")
    shaped = shape_joy_cmd(0.7, 0.9, -0.4, mode="fixed", fixed_speed=0.5,
                           forward_max=0.6, lateral_max=0.4, yaw_max=1.0)
    check(shaped == (0.5, 0.9, -0.4), "fixed 模式下横移/转向原样透传", f"实际 {shaped}")

    print("-" * 72)
    print("2) continuous 模式（新策略要的速度跟踪）")
    check(approx(map_stick(0.0, 0.6), 0.0), "摇杆归零 -> 0")
    check(approx(map_stick(0.5, 0.6), 0.3), "半推 -> 0.3 m/s（线性映射）", f"实际 {map_stick(0.5, 0.6)}")
    check(approx(map_stick(1.0, 0.6), 0.6), "满推 -> 上限 0.6 m/s")
    check(approx(map_stick(-1.0, 0.6), -0.6), "满拉 -> -0.6 m/s（能后退）")
    check(approx(map_stick(1.7, 0.6), 0.6), "越界输入被夹到上限", f"实际 {map_stick(1.7, 0.6)}")
    check(approx(map_stick(-3.0, 0.4), -0.4), "负向越界同样被夹")
    check(approx(map_stick(0.5, -1.0), 0.0), "负的上限参数 -> 0（不会反号）")
    shaped = shape_joy_cmd(1.0, -0.5, 0.25, mode="continuous", fixed_speed=0.5,
                           forward_max=0.6, lateral_max=0.4, yaw_max=1.0)
    check(all(approx(a, b) for a, b in zip(shaped, (0.6, -0.2, 0.25))),
          "三轴同时映射（前进/横移/转向）", f"实际 {shaped}")

    print("-" * 72)
    print("3) 模式解析（非法值退回 fixed = 安全侧）")
    check(continuous_speed_mode("continuous") is True, "'continuous' -> 连续模式")
    check(continuous_speed_mode(" Continuous ") is True, "大小写与空格容错")
    check(continuous_speed_mode("fixed") is False, "'fixed' -> 固定模式")
    check(continuous_speed_mode("garbage") is False, "非法值 -> 固定模式（不会误开连续模式）")
    check(continuous_speed_mode("") is False, "空值 -> 固定模式")

    failed = [item for ok, item, _ in _results if not ok]
    print("=" * 72)
    print(f"总计 {len(_results)} 项，通过 {len(_results) - len(failed)}，失败 {len(failed)}")
    for item in failed:
        print(f"  FAILED: {item}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
