"""cmd_vel_shaping.py —— 手柄摇杆 → 速度指令的映射（纯函数，便于离线测试）。

历史背景：
  旧策略只会一个前进速度，所以 `cmd_vel_mux_node` 里把任何非零的 linear.x 都
  硬压成常数 ±0.5（真机实测约 0.61 m/s）。新策略（unitree_rl_gym 训出来的速度
  跟踪器）需要连续可调的指令，否则训练出来的调速能力完全用不上。

这里把两种映射抽成纯函数，节点只负责读参数与发布：
  * fixed      —— 旧行为，逐位保持（默认 0.5，对应原来的硬编码值）
  * continuous —— 摇杆幅度线性映射到 [0, max]，再交给后面的最小速度塑形
"""

from __future__ import annotations

FIXED = "fixed"
CONTINUOUS = "continuous"


def continuous_speed_mode(mode: str) -> bool:
    """参数值 -> 是否连续模式。非法值一律退回 fixed（安全侧）。"""
    return str(mode).strip().lower() == CONTINUOUS


def map_stick(value: float, magnitude: float) -> float:
    """摇杆值（-1..1，节点上游已做死区）线性映射到 ±magnitude。"""
    if value == 0.0:
        return 0.0
    clipped = max(-1.0, min(1.0, float(value)))
    return clipped * max(0.0, float(magnitude))


def fixed_forward(value: float, speed: float) -> float:
    """旧行为：非零摇杆 -> 固定大小的速度（保留方向）。"""
    if value == 0.0:
        return 0.0
    return (-1.0 if float(value) < 0.0 else 1.0) * abs(float(speed))


def shape_joy_cmd(linear_x: float, linear_y: float, angular_z: float, *, mode: str,
                  fixed_speed: float, forward_max: float, lateral_max: float, yaw_max: float,
                  backward_max: float = None):
    """返回 (linear_x, linear_y, angular_z) 三个映射后的值。

    前进/后退**分别限幅**（2026-10-15 新增）：
      linear_x > 0 用 ``forward_max``，linear_x < 0 用 ``backward_max``。
    这样可以让前进更快、后退更保守（后退时看不清后方，且极易后仰摔倒）。

    ``backward_max`` 为 None 时退回旧的对称行为（两侧都用 ``forward_max``），
    保证不破坏既有调用方。
    """
    if continuous_speed_mode(mode):
        back = forward_max if backward_max is None else backward_max
        x_max = forward_max if float(linear_x) >= 0.0 else back
        return (
            map_stick(linear_x, x_max),
            map_stick(linear_y, lateral_max),
            map_stick(angular_z, yaw_max),
        )
    return (fixed_forward(linear_x, fixed_speed), float(linear_y), float(angular_z))
