# go2_rl_gym 产出的策略能否导入 dm_dog_controller

结论速览：**接口层面兼容，普通 `go2`（PPO）策略可以被加载；MoE/CTS 系列不能。
但默认关节角、PD 增益、机型三项必须处理，否则不能上真机。**

---

## 1. go2_rl_gym 产出的是什么

`play.py` 结束后导出到 `logs/<experiment_name>/exported/policies/`：

| 文件 | 格式 | 用途 |
| --- | --- | --- |
| `policy.pt` | TorchScript | Mujoco Sim2Sim / Python 部署 |
| `policy.onnx` | ONNX opset 11 | Sim2Real（C++ 部署） |
| `policy.pkl` | 原始 `state_dict` | 备用 |

### 观测：45 维，单帧，无相位时钟

| 区间 | 内容 | 缩放 |
| --- | --- | --- |
| `0:3` | base_ang_vel | ×0.25 |
| `3:6` | projected_gravity | — |
| `6:9` | command (vx, vy, wz) | ×[2.0, 2.0, 0.25] |
| `9:21` | joint_pos − default_joint_angles | ×1.0 |
| `21:33` | joint_vel | ×0.05 |
| `33:45` | last_action | — |

### 动作：关节位置残差

```text
target_dof_pos = action * 0.25 + default_joint_angles
```

- 网络：`ActorCritic` MLP `45 → 512 → 256 → 128 → 12`
- 控制频率：`dt=0.005` × `decimation=4` = **50 Hz**
- 训练 PD 增益：`stiffness = 20.0`，`damping = 0.5`

---

## 2. dm_dog_controller 的接口契约

来自 `bs_hardware/config.py`：

| 项目 | dm_dog_controller | go2_rl_gym | 是否一致 |
| --- | --- | --- | --- |
| 观测总维 | `OBS_DIM=45`（另支持 47） | 45 | ✅ |
| 前 9 维顺序 | `OBS_HEAD_ORDER="ang_vel,gravity,cmd"` | ang_vel, gravity, cmd | ✅ |
| 角速度缩放 | `ANGULAR_VELOCITY_SCALE=0.25` | 0.25 | ✅ |
| 速度指令缩放 | `COMMAND_SCALE=[2,2,0.25]` | [2.0, 2.0, 0.25] | ✅ |
| 关节速度缩放 | `DOF_VELOCITY_SCALE=0.05` | 0.05 | ✅ |
| 关节位置缩放 | 1.0（代码里不减缩放） | 1.0 | ✅ |
| 动作缩放 | `RL_ACTION_SCALE=0.25` | 0.25 | ✅ |
| 动作语义 | `model_target = default + action*scale` | 同 | ✅ |
| 控制频率 | `hz=50` | 50 Hz | ✅ |
| 网络入口 | `forward` / `act_inference` | `forward` | ✅ |
| **默认关节角** | `[0, 0.8, -1.5] × 4` | `[0.1,0.8,-1.5, -0.1,0.8,-1.5, 0.1,1.0,-1.5, -0.1,1.0,-1.5]` | ❌ |
| **PD 增益** | 120/2.2（平移）、65/1.2（偏航） | 训练用 20/0.5 | ❌ |

### 观测拼接实测对齐

`deployment.py` 的 `build()` 在 `head_order=None` 且 `command_offset=6` 时走这一支：

```python
parts = [base_ang_vel, gravity, command,
         dof_pos - default_dof_pos,
         dof_vel * DOF_VELOCITY_SCALE,
         last_action]
```

`has_phase=False` 时不追加相位。**与 go2_rl_gym 逐位一致。**

---

## 3. 用 dm 的加载器实测

把 go2_rl_gym 导出的策略直接喂给 `dm_dog_controller.deployment.PolicyDeployer`：

```text
loaded test: path=.../go2_cts_150k.pt entry=forward
  input_dim=45 per_frame=45 history=1 has_phase=False
  obs_layout=legacy-45 traced_layout=None
  command_offset=6 head_order=ang_vel,gravity,cmd history_order=newest_first
>>> 加载成功
```

```text
go2_moe_cts_137k_0.6739.pt
>>> 加载失败: AttributeError 'tuple' object has no attribute 'detach'
```

### 各 task 导出物的兼容性

| task | `policy_class_name` | 导出 `forward` 返回 | 能否被 dm 加载 |
| --- | --- | --- | --- |
| **`go2`** | `ActorCritic` | **单个 tensor (1,12)** | ✅ |
| `go2_cts` | `ActorCriticCTS` | tuple `(action,(None,latent))` | ❌ |
| `go2_moe_cts` | `ActorCriticMoECTS` | tuple `(action,(weights,latent))` | ❌ |
| `go2_moe_ng_cts` | `ActorCriticMoENGCTS` | tuple | ❌ |
| `go2_mcp_cts` | `ActorCriticMCPCTS` | tuple | ❌ |
| `go2_ac_moe_cts` | `ActorCriticACMoECTS` | tuple | ❌ |
| `go2_dual_moe_cts` | `ActorCriticDualMoECTS` | tuple | ❌ |

> 仓库里发布的 `deploy/pre_train/go2/go2_cts_150k.pt` 是**旧版导出件**，
> 恰好是单张量，所以能加载。用**当前代码**重新导出 `go2_cts` 会得到 tuple，不能加载。
>
> 失败根因在 `deployment.py:255`：`action_tensor.detach()` 对 tuple 不可用。
> 要支持 MoE，需改 `_validate_inference()` / `inference()` 取 `out[0]`，
> 并注意 MoE 的 TorchScript 内部维护 `history` 缓冲，需要每回合调 `reset()`。

---

## 4. 要落地必须处理的四项

### 4.1 默认关节角（必须改）

`bs_hardware/config.py` 的 `DEFAULT_DOF_POS` 要改成 go2_rl_gym 训练时用的值，
否则观测里的 `joint_pos_error` 和动作目标角**同时**偏掉：

```python
DEFAULT_DOF_POS = [
     0.1,  0.8, -1.5,   # FL
    -0.1,  0.8, -1.5,   # FR
     0.1,  1.0, -1.5,   # RL
    -0.1,  1.0, -1.5,   # RR
]
```

### 4.2 PD 增益（必须改）

策略是在 `kp=20 / kd=0.5` 的动力学下训练的，dm 当前用 `120/2.2`（6 倍刚度）。
要么把部署增益降到 20/0.5，要么重新训练时把 `go2_config.py` 的
`stiffness` / `damping` 改成 dm 实机值再训。

### 4.3 机型差异（最大不确定项）

go2_rl_gym 训练对象是 **Unitree Go2**；dm_dog_controller 驱动的是一台
**达妙(DM)电机 + MIT URDF + BM  calf 电机**的自研四足。
连杆长度、质量分布、电机减速比、通信延迟若与 Go2 不同，
即使接口全对，零样本迁移也可能直接摔倒。

**建议先在 MuJoCo 里用 dm 的 URDF 跑 Sim2Sim**，通过后再上电。

### 4.4 关节符号标定

`MODEL_JOINT_DIRECTIONS` / `STARTUP_JOINT_DIRECTIONS` 是给旧策略标定的。
换策略后需重新确认每个关节的残差符号方向（`launch` 注释建议先 `action_clip:=0.3` 试方向）。

---

## 5. launch 侧要改的四处（作者原话）

`launch/dm_dog_policy.launch.py:113-126` 已写明：

1. `"model"` 改成新导出的策略文件
2. `"use_dual_policy": False`，删掉 `"yaw_model"`
3. `cmd_vel_mux_node` 的 `"forward_speed_mode": "continuous"`
4. 首次试机 `action_clip` 先降到 `0.3`，确认迈腿方向后回到 `1.0`

### ⚠️ 启动日志的预期值与实际会不一样

该注释要求看到：

```text
loaded translation_policy: ... input_dim=47 per_frame=47 history=1 has_phase=True
```

那是 **官方 unitree_rl_gym**（带步态相位时钟，47 维）的契约。
**本仓库 `go2_rl_gym` 产出的是 45 维、无相位**，所以日志会显示：

```text
input_dim=45 per_frame=45 history=1 has_phase=False obs_layout=legacy-45
```

**这是正常的，不是错误**，只要同时看到
`command_offset=6 head_order=ang_vel,gravity,cmd` 就说明布局契约对上了。

---

## 6. 上真机前的检查清单

```bash
# 1) 用 dm 自带的离线测试确认布局没被改坏
cd ~
python tools/test_obs_layout.py

# 2) 无硬件的冒烟测试（不发送动作）
ros2 run dm_dog_controller dm_dog_policy_node --ros-args \
  -p use_imu:=false -p use_can:=false -p startup:=false \
  -p send_actions:=false -p gamepad_control:=false -p steps:=1
# 启动日志里核对 input_dim / per_frame / has_phase / command_offset

# 3) MuJoCo Sim2Sim（用 dm 的 URDF）确认步态正确

# 4) 上电，action_clip 先 0.3，手放急停
```

---

## 7. 一句话总结

> **`go2`（PPO）导出的 `policy.pt` 可以导入 dm_dog_controller，
> 观测/动作/频率接口逐位对齐，实测加载成功。**
> **但必须同步修改 `DEFAULT_DOF_POS` 和 PD 增益，并先在 MuJoCo 里验证机型差异。**
> **`go2_moe_cts` 等 MoE/CTS 策略返回 tuple，需改部署代码才能用。**
