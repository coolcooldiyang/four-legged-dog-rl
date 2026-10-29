#!/usr/bin/env python3
"""test_obs_layout.py —— 观测布局与步态时钟的离线单元测试（不需要硬件）。

为什么要有这个测试：
  * 这条狗同时存在两代策略，观测布局不同（legacy 45 维 × 5 帧 = 225；
    unitree_rl_gym 可部署布局 47 维 × 1 帧）。
  * 旧代码从 TorchScript 里找 "command = torch.slice(...)" 来推断命令位置；
    rsl_rl 导出的纯 actor **没有这一行**，旧代码会静默地按"命令在最前面"处理 ——
    和实际训练布局正好相反。这个 bug 一旦上真机极难定位（症状只是"狗不听话"）。
  * 本测试把 47 维那条路径钉死，并同时验证 225 维老路径没有被改坏。

用法：
    python tools/test_obs_layout.py                       # 用仓库自带的模型文件
    python tools/test_obs_layout.py --policy policy.pt    # 指定 legacy 模型

退出码 0 = 全部通过。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Windows 上没有 termios/tty；这两个模块只在函数内部使用，给个空壳即可。
if os.name == "nt":
    for name in ("termios", "tty"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)

_results: list[tuple[bool, str, str]] = []


def check(condition: bool, item: str, detail: str = "") -> None:
    _results.append((bool(condition), item, detail))
    print(f"[{' ok ' if condition else 'FAIL'}] {item}" + (f"\n        {detail}" if detail else ""))


def section(title: str) -> None:
    print("-" * 72)
    print(title)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=os.path.join(REPO_ROOT, "policy.pt"),
                        help="legacy 策略文件（默认仓库根目录的 policy.pt）")
    args = parser.parse_args()

    import numpy as np
    import torch

    from bs_hardware.config import (
        ACTION_DIM, COMMAND_SCALE, DEFAULT_DOF_POS, OBS_DIM, OBS_DIM_DEPLOYABLE,
        OBS_PHASE_PERIOD_S, resolve_observation_spec,
    )
    from dm_dog_controller.deployment import DeploymentRunner, ObservationBuilder, PolicyDeployer

    # ---------------------------------------------------------------- 布局解析
    section("1) 观测布局解析（按输入宽度判定，45 与 47 互质所以无歧义）")
    cases = {
        225: (45, 5, False, "legacy-45"),
        45: (45, 1, False, "legacy-45"),
        90: (45, 2, False, "legacy-45"),
        47: (47, 1, True, "deployable-47"),
        235: (47, 5, True, "deployable-47"),
    }
    for width, expected in cases.items():
        got = resolve_observation_spec(width)
        check(got == expected, f"width {width} -> {expected}", f"实际: {got}")
    for width in (0, -1, 46, 100, 200):
        check(resolve_observation_spec(width) is None, f"width {width} 应判为未知布局")

    # ------------------------------------------------- 可部署布局（47 维单帧）
    section("2) 可部署布局构造（unitree_rl_gym：命令 offset 6，末尾 2 维步态相位）")
    builder = ObservationBuilder(history_length=1, command_offset=6, newest_first=True,
                                 per_frame=OBS_DIM_DEPLOYABLE, has_phase=True)
    builder.update_command((0.5, 0.0, -0.2))
    builder.update_action(np.arange(ACTION_DIM, dtype=np.float32) * 0.01)
    obs = builder.build(None)
    check(obs.shape == (47,), "build() 返回 47 维向量", f"实际 {obs.shape}")
    expected_cmd = np.asarray((0.5, 0.0, -0.2), dtype=np.float32) * np.asarray(COMMAND_SCALE, dtype=np.float32)
    check(np.allclose(obs[6:9], expected_cmd), "命令位于 offset 6..8 且已乘 COMMAND_SCALE",
          f"实际 {obs[6:9]}，期望 {expected_cmd}")
    check(np.allclose(obs[9:21], -np.asarray(DEFAULT_DOF_POS, dtype=np.float32)),
          "关节位置误差 = dof_pos - default_dof_pos（此处 dof_pos 初值为 0）")
    check(np.allclose(obs[33:45], np.arange(ACTION_DIM, dtype=np.float32) * 0.01),
          "上一帧动作位于 33..44")
    check(np.allclose(obs[45:47], (math.sin(0.0), math.cos(0.0)), atol=1e-6),
          "相位未启动时 [45:47] = (sin 0, cos 0) = (0, 1)", f"实际 {obs[45:47]}")

    # 相位时钟
    builder.start_phase(now=100.0)
    builder.tick_phase(now=100.0 + OBS_PHASE_PERIOD_S / 4.0)
    phase = builder.tick_phase(now=100.0 + OBS_PHASE_PERIOD_S / 4.0)
    check(abs(phase - 0.25) < 1e-6, "相位按 0.8 s 周期推进（1/4 周期 -> 0.25）", f"实际 {phase}")
    pair = builder.phase_pair()
    check(np.allclose(pair, (math.sin(math.pi / 2), math.cos(math.pi / 2)), atol=1e-6),
          "1/4 周期相位对 = (1, 0)", f"实际 {pair}")
    builder.tick_phase(now=100.0 + OBS_PHASE_PERIOD_S * 1.5)
    check(abs(builder.phase - 0.5) < 1e-6, "相位按周期取模（1.5 周期 -> 0.5）", f"实际 {builder.phase}")
    builder.stop_phase()
    frozen = builder.tick_phase(now=100.0 + OBS_PHASE_PERIOD_S * 9.0)
    check(abs(frozen - 0.5) < 1e-6, "stop_phase() 后相位冻结（狗停下来时不该继续走时钟）", f"实际 {frozen}")
    builder.start_phase(now=200.0)
    check(builder.phase == 0.0, "start_phase() 把相位复位到 0")

    # ------------------------------------------------------- legacy 45×5 布局
    section("3) legacy 布局回归（45 维 × 5 帧，不能因为新布局把老路径改坏）")
    legacy = ObservationBuilder(history_length=5, command_offset=6, newest_first=False, per_frame=OBS_DIM, has_phase=False)
    legacy.update_command((1.0, 0.0, 0.0))
    vec = legacy.build(None)
    check(vec.shape == (225,), "history_vector() 为 225 维", f"实际 {vec.shape}")
    scale = np.asarray(COMMAND_SCALE, dtype=np.float32)
    expected_cmd = np.asarray((1.0, 0.0, 0.0), dtype=np.float32) * scale   # = [2, 0, 0]
    check(np.allclose(vec[-OBS_DIM:][6:9], expected_cmd),
          "oldest_first：最新一帧在末尾，命令在帧内 offset 6..8", f"实际 {vec[-OBS_DIM:][6:9]}")
    check(np.allclose(vec[:OBS_DIM], 0.0),
          "oldest_first：最早一帧（初始化补的零帧）仍在开头，说明拼接方向是旧->新")
    check(np.allclose(legacy.phase_pair(), (0.0, 1.0)), "legacy 布局不含相位（phase_pair 不参与拼接）")

    newest_first_builder = ObservationBuilder(history_length=5, command_offset=6, newest_first=True,
                                              per_frame=OBS_DIM, has_phase=False)
    newest_first_builder.update_command((1.0, 0.0, 0.0))
    vec_newest = newest_first_builder.build(None)
    check(np.allclose(vec_newest[:OBS_DIM][6:9], expected_cmd),
          "newest_first：最新一帧在开头（与 oldest_first 相反，不能混）",
          f"实际 {vec_newest[:OBS_DIM][6:9]}")

    # --------------------------------------------------- 真实 legacy 策略文件
    section("4) 真实 legacy 策略（policy.pt）兼容性")
    if os.path.isfile(args.policy):
        deployer = PolicyDeployer(args.policy, name="legacy")
        check(deployer.total_input_dim == 225, "输入维 225", f"实际 {deployer.total_input_dim}")
        check((deployer.per_frame, deployer.history_length) == (OBS_DIM, 5), "per_frame=45 history=5",
              f"实际 per_frame={deployer.per_frame} history={deployer.history_length}")
        check(deployer.has_phase is False, "legacy 布局 has_phase=False")
        check(deployer.command_offset == 6, "命令 offset=6（沿用 traced slice 的判定）",
              f"实际 {deployer.command_offset}（traced={deployer.traced_layout_name}）")
    else:
        check(True, f"跳过：找不到 {args.policy}")

    # ------------------------------------------- 合成的 47 维策略（关键回归测试）
    section("5) 合成 47 维策略：验证没有 traced slice 时不再误判成 command-first")
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "policy_1.pt")
        # 完全照 unitree_rl_gym 的导出方式：export_policy_as_jit 把 actor（一个
        # Sequential MLP）用 torch.jit.script 存成 policy_1.pt，入口是 forward。
        actor = torch.nn.Sequential(
            torch.nn.Linear(OBS_DIM_DEPLOYABLE, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, ACTION_DIM),
        ).eval()
        torch.jit.script(actor).save(path)

        deployer = PolicyDeployer(path, name="dog4438")
        check(deployer.total_input_dim == OBS_DIM_DEPLOYABLE, "探测到输入维 47（不是 225、不是 45）",
              f"实际 {deployer.total_input_dim}")
        check((deployer.per_frame, deployer.history_length, deployer.has_phase) == (47, 1, True),
              "解析为 deployable-47：单帧 + 相位",
              f"实际 per_frame={deployer.per_frame} history={deployer.history_length} has_phase={deployer.has_phase}")
        check(deployer.command_offset == 6, "命令 offset 落在 6（修复前会静默误判为 0）",
              f"实际 {deployer.command_offset}，traced={deployer.traced_layout_name}")

        builder2 = ObservationBuilder(history_length=deployer.history_length,
                                      command_offset=deployer.command_offset,
                                      newest_first=deployer.newest_first,
                                      per_frame=deployer.per_frame,
                                      has_phase=deployer.has_phase)
        builder2.start_phase(now=0.0)
        builder2.tick_phase(now=0.1)
        vec2 = builder2.build(None)
        check(vec2.shape == (47,), "构造出的观测长度与模型输入一致", f"实际 {vec2.shape}")
        action = deployer.inference(vec2)
        check(action.shape == (ACTION_DIM,), "推理输出 12 维动作", f"实际 {action.shape}")

        # 反向验证：必须不能把 47 维模型当成 45 维老布局（那会喂错数据）
        check(resolve_observation_spec(deployer.total_input_dim)[0] != OBS_DIM,
              "47 维模型不会被误解析成 45 维 legacy 布局")

    # ------------------------------------------------------ runner 的时钟接线
    section("6) DeploymentRunner 步态时钟接线（不构造真实 runner，只测胶水逻辑）")
    runner = DeploymentRunner.__new__(DeploymentRunner)   # 绕过需要硬件的构造函数
    clocked = ObservationBuilder(history_length=1, command_offset=6, newest_first=True,
                                 per_frame=OBS_DIM_DEPLOYABLE, has_phase=True)
    runner.translation_obs_builder = clocked
    runner.yaw_obs_builder = clocked
    runner.rl_active = False
    runner._phase_running = False
    runner._update_gait_clock()
    check(clocked._phase_started_at is None, "RL 未接管时不启动时钟")
    runner.rl_active = True
    runner._update_gait_clock()
    check(clocked._phase_started_at is not None, "RL 接管的第一个周期启动时钟")
    runner.rl_active = False
    runner._update_gait_clock()
    check(clocked._phase_started_at is None and runner._phase_running is False,
          "停止（hold/急停）时冻结时钟，恢复时从 0 重新计时")

    failed = [item for ok, item, _ in _results if not ok]
    print("=" * 72)
    print(f"总计 {len(_results)} 项，通过 {len(_results) - len(failed)}，失败 {len(failed)}")
    for item in failed:
        print(f"  FAILED: {item}")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
