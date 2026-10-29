#!/usr/bin/env python3
"""make_test_policy47.py —— 生成 47 维"零输出"测试策略。

用途（两个，都不需要等训练完成）：
  1) **离线验证部署链路**：让 dm_dog_policy_node 加载一个 47 维模型，确认布局判定、
     观测拼接、推理、相位时钟全部按新契约工作。
  2) **真机第一步的安全测试**：零权重 → 动作恒为 0 → 目标角就是 default_dof_pos
     （站立零位），狗只会站着不动。这是"打通链路但不动"的最安全一次上机验证：
     如果这一步狗乱动，说明观测/动作映射有问题，而不是策略问题。

用法：
    python tools/make_test_policy47.py --out /tmp/policy_test47_zero.pt
    python tools/make_test_policy47.py --out /tmp/policy_test47_random.pt --mode random

--mode zero   （默认）权重全零，动作恒为 0
--mode random 小随机权重，动作幅值约 ±0.1（比 default_dof_pos 的变化尺度 0.25 小，
              狗会轻微晃腿；只在吊起来/有人扶的情况下用）
"""

from __future__ import annotations

import argparse
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Windows 上没有 termios/tty，而 bs_hardware/__init__ 会连带导入 imu_l1；给个空壳即可。
if os.name == "nt":
    for _name in ("termios", "tty"):
        if _name not in sys.modules:
            sys.modules[_name] = types.ModuleType(_name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="输出的 .pt 路径")
    parser.add_argument("--mode", choices=("zero", "random"), default="zero")
    parser.add_argument("--obs-dim", type=int, default=None, help="默认取 config 里的 OBS_DIM_DEPLOYABLE(47)")
    parser.add_argument("--action-dim", type=int, default=None)
    args = parser.parse_args()

    import torch
    from bs_hardware.config import ACTION_DIM, OBS_DIM_DEPLOYABLE

    obs_dim = args.obs_dim or OBS_DIM_DEPLOYABLE
    action_dim = args.action_dim or ACTION_DIM

    # 完全照 unitree_rl_gym 的导出方式：actor 是一个 Sequential MLP，
    # 用 torch.jit.script 保存，入口是 forward（部署侧 policy(obs) 就是这么调的）。
    actor = torch.nn.Sequential(
        torch.nn.Linear(obs_dim, 64),
        torch.nn.ELU(),
        torch.nn.Linear(64, action_dim),
    ).eval()

    with torch.no_grad():
        if args.mode == "zero":
            for param in actor.parameters():
                param.zero_()
        else:
            for param in actor.parameters():
                param.uniform_(-0.05, 0.05)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    torch.jit.script(actor).save(args.out)

    probe = os.path.commonprefix(["x"])  # noqa: F841  (keep flake quiet)
    import numpy as np
    rollout = actor(torch.zeros(1, obs_dim)).detach().numpy().reshape(-1)
    print("已写出 %s" % args.out)
    print("  观测维 %d  动作维 %d  模式 %s" % (obs_dim, action_dim, args.mode))
    print("  零输入下的动作: min=%.4f max=%.4f max|a|=%.4f"
          % (float(rollout.min()), float(rollout.max()), float(abs(rollout).max())))
    print("  部署侧启动日志应显示: input_dim=%d per_frame=%d history=1 has_phase=True" % (obs_dim, obs_dim))
    return 0


if __name__ == "__main__":
    sys.exit(main())
