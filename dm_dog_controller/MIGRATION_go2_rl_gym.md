# 迁移到 go2_rl_gym（go2 PPO）策略 — 状态与上真机清单

本文档记录 dm_dog_controller 从 legacy htdw_4438 策略迁移到
go2_rl_gym 训练的 go2 PPO 策略的全部改动、验证结果与上真机步骤。

---

## 1. 已完成的代码改动

| 文件 | 改动 | 备份 |
| --- | --- | --- |
| `bs_hardware/config.py` | `DEFAULT_DOF_POS`、PD 增益 | `bs_hardware/config.py.bak-before-go2-policy` |
| **`policies/go2_cts_150k.pt`** | **当前使用**（仓库自带预训练，最稳定） | — |
| `policies/go2_him_21k_0.5379.pt` | 备选，更快但更晃 | — |
| `policies/go2_dwaq_119.5k_0.5054.pt` | 备选，最快但最晃 | — |
| `policies/go2_ppo_7000.pt` 等 | 自训 checkpoint，留作对比 | — |
| `training_backup/` | 训练备份（tar + model_8500.pt） | — |
| `policies/go2_ppo_500.pt` | 备选（更早的 checkpoint） | — |

### 1.1 `DEFAULT_DOF_POS`

```python
# 旧（legacy htdw_4438）：[0.0, 0.8, -1.5] x 4
# 新（go2_rl_gym go2_config.py 的 default_joint_angles）：
DEFAULT_DOF_POS = [
     0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,   # FR
     0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5,   # RR
]
```

### 1.2 PD 增益 —— kp 保持 20，kd 提到 1.5

| 项 | 现值 | 旧值（legacy） |
| --- | --- | --- |
| `TRANSLATION_POLICY_RL_KP / KD` | **20.0 / 1.5** | 120.0 / 2.2 |
| `TRANSLATION_POLICY_BM_CALF_KP / KD` | **20.0 / 1.5** | 100.0 / 20.0 |
| `YAW_POLICY_RL_KP / KD` | **20.0 / 1.5** | 65.0 / 1.2 |
| `YAW_POLICY_BM_CALF_KP / KD` | **20.0 / 1.5** | 75.0 / 20.0 |

> `YAW_POLICY_*` 在 `use_dual_policy: False` 时不会被使用
> （`_target_policy_gains` 走 `TRANSLATION_*` 分支）。

#### ⭐ 实机试机结论：**增益越低实机越稳**（与直觉相反）

| 增益 | 实机表现 |
| --- | --- |
| 120 / 2.2（legacy） | 偏硬 |
| 20 / 0.5（训练值） | **腿在自己"响"** → 用户报"腿动得太快" |
| 60 / 1.5（已否决） | 狗身抖动严重（kp 抬高把闭环带宽抬上去了） |
| **20 / 1.5（当前）** | 只提阻尼不提刚度 |

**关键：kp 决定闭环带宽，kd 决定阻尼。** 关节固有频率
`omega_n = sqrt(kp/J) ≈ 7 Hz`（大腿 J≈0.01 kg·m²），而步态只有 1.67 Hz。
`kd=0.5` 时 `zeta≈0.56`（欠阻尼），腿会在 7 Hz 自激 → 看起来"动得太快"。
`kd=1.5` 时 `zeta≈1.68`（过阻尼），自激被压住，而 `omega_n` 不变，
所以不会像 60/1.5 那样把高频耦合进机身。

> **策略本身不快**：实测四个候选策略的步态主频都是 1.67 Hz，
> 关节行程 go2_cts_150k 最小（大腿 0.622 rad）。所以"腿快"是控制环问题。

#### ⚠️ kp 与 action_clip 是耦合的

**峰值需求力矩 ≈ `kp × action_clip × 0.25`**：

| kp | action_clip | 峰值力矩 | 实机结果 |
| --- | --- | --- | --- |
| **20** | **4.0** | **20 Nm** | ← 当前，可用 |
| 60 | 2.0 | 30 Nm | 已否决：狗身抖动严重 |
| 120 | 4.0 | 120 Nm | 已否决：上电过猛 |
| 120 | 1.0 | 30 Nm | 已否决：偏硬 |

改 kp 必须同步改 launch 里的 `action_clip`。

**`STARTUP_KP/KD = 30.0/1.0` 与 `BM_CALF_STARTUP_KP/KD = 40.0/20.0` 未改动**——
那是启动序列撑站姿用的，与 RL 策略增益是两套值。

> 改完**必须 `colcon build`**：`bs_hardware` 是通过
> `ament_python_install_package` **复制**安装的，不重建的话 ROS 里跑的还是旧值。

### 1.3 关节方向符号试验（已排除）

2026-10 实机试验：把 `MODEL_JOINT_DIRECTIONS` 的**髋列翻转**
（`[-1,-1,+1,+1]` → `[+1,+1,-1,-1]`，大腿/小腿不动）后试机。

**结果：实机表现更怪。** 已改回原标定值。

**结论：关节方向符号不是 Go2 系列策略在实机上异常的原因。**

至此已排除的假设：

| 假设 | 排除方式 | 结果 |
| --- | --- | --- |
| kd 太低导致腿自激 | 实机试 kd=1.5 | 用户判断不是 |
| action_clip 太大 | 实机试 clip=2.0 | 用户判断不是 |
| 髋方向符号反了 | 实机翻转髋列 | **更怪，已回退** |
| 站姿比 Go2 窄 | 实测左右腿间距 | ~27 cm，与 Go2 相当 |
| 摇杆漂移导致自己走 | 改 deadzone + min_output | 已修，但"停不下来"仍在 |

**剩余解释：跨机器人零样本迁移**——这些策略训练在 Unitree Go2 的 URDF 上
（不是本狗），关节符号以外的**机型参数与执行器特性**（DM 电机 + BM 小腿电机
vs Go2 电机）不匹配，且离线无法验证。

---

---

## 2. 已完成的验证

| 验证项 | 结果 | 证据 |
| --- | --- | --- |
| `tools/test_obs_layout.py` | **38 项通过 / 0 失败** | — |
| 无硬件冒烟（CLI 路径） | 通过 | `python -m dm_dog_controller.deployment --no-imu --no-can --steps 1` |
| 无硬件冒烟（ROS 真路径） | 通过 | `ros2 run dm_dog_controller dm_dog_policy_node ...` |
| MuJoCo Sim2Sim | **通过** | 基座 0.293 m，前向 0.693 m/s，不摔 |
| 观测构造一致性 | **逐位相同，max diff = 0.0** | dm `ObservationBuilder` vs go2_rl_gym deploy 公式 |

加载日志（应与此一致）：

```text
loaded translation_policy: entry=forward input_dim=45 per_frame=45 history=1
  has_phase=False obs_layout=legacy-45 command_offset=6
  head_order=ang_vel,gravity,cmd history_order=newest_first
```

---

## 3. launch 已改好（4 处）

`launch/dm_dog_policy.launch.py` 已按作者注释（原第 113-126 行）完成迁移：

| # | 参数 | 旧值 | 现值 |
| --- | --- | --- | --- |
| 1 | `dm_dog_policy_node` 的 `model` | `policy_stair.pt` | **`policies/go2_cts_150k.pt`** |
| 2 | `use_dual_policy` | `True` | **`False`**（并删掉 `yaw_model`） |
| 3 | `cmd_vel_mux_node` 的 `forward_speed_mode` | `"fixed"` | **`"continuous"`** |
| 4 | `action_clip` | `1.0` | **`4.0`**（与 kp=20 耦合，见 1.2 节） |

配套改动：`CMakeLists.txt` 增加了

```cmake
install(DIRECTORY policies DESTINATION share/${PROJECT_NAME})
```

否则 launch 在 `share/dm_dog_controller/` 下找不到 `policies/` 里的策略文件。
**改完必须重新 `colcon build`**，launch 与文件才会同步到 install 目录。

### 关于 `action_clip = 4.0`

代码里 `deployment.py:1421` 是 `if self.action_clip > 0: action = np.clip(...)`，
即 **`<=0` 表示关闭裁剪**。

实测策略的原始动作幅值（MuJoCo 平地，稳态段 95 个控制步）：

| 指标 | 值 |
| --- | --- |
| 最大 \|action\| | **3.52** |
| 99 分位 | 3.17 |
| 中位数 | 0.73 |
| \|action\| > 0.3 比例 | 79.3% |
| \|action\| > 1.0 比例 | 37.7% |

训练时 `clip_actions = 100.`（不裁剪），所以 clip 越小越偏离训练条件。
取 **2.0** 是为了配合 kp=60 把峰值力矩压在 ~30 Nm（见第 1.2 节的耦合关系）。

各取值的换算（残差 = \|action\| × `action_scale` 0.25）：

| `action_clip` | 残差上限 | 被截断动作 | 配合 kp |
| --- | --- | --- | --- |
| 0.3 | 4.3° | 79% | kp=120 时动作几乎看不出来 |
| 1.0 | 14.3° | 38% | kp=120 时偏硬 |
| 2.0 | 28.6° | ~25% | kp=60 时狗身抖动严重 |
| **4.0（当前）** | **57°** | ~0% | **kp=20** |
| 4.0 | 57° | ~0% | kp=120 时 ~120 Nm，实机过猛 |
| 0.0 | 不限 | 0% | 完全关闭裁剪 |

#### 实机试机历程（clip 与 kp 必须一起看）

| clip | kp | 实机表现 |
| --- | --- | --- |
| 0.3 | 120 | 动作几乎看不出来（截断 79%） |
| 4.0 | 120 | 上电动作过猛（≈120 Nm） |
| 1.0 | 120 | 好转，但仍偏硬（≈30 Nm） |
| **4.0** | **20（当前）** | **可用**：大腿关节有摇摆，但整体正常 |
| 2.0 | 60 | 已否决：狗身抖动严重 |

> 1.0 会截断 38% 的动作，属刻意限幅。现在增益已对齐训练，
> 若实机确认 1.0 下步态正常且动作偏小，可以逐步升到 1.5~2.0 再观察。

### 摇了摇杆之后狗一直自己往前走（已修）

**现象**：手柄回中不动，狗仍然持续前进。

**根因**：两个节点的组合效应（`gamepad_command_node` + `cmd_vel_mux_node`）。

```python
# gamepad_command_node：死区 0.12，超出后重新线性映射到 0~1
return sign * ((magnitude - deadzone) / (1.0 - deadzone))

# cmd_vel_mux_node：把小于 minimum 的值"吸"到 minimum
def _apply_min_abs(value, minimum):
    if math.fabs(value) < minimum:
        return math.copysign(minimum, value)
```

这只手柄回中后仍有漂移。只要漂移略微超过死区，比如 0.13：

```
(0.13 - 0.12) / 0.88 = 0.011  →  ×0.6 = 0.0068 m/s  →  被 min 塑形成 0.2 m/s
```

又因为 `_has_motion` 判定为真会不断刷新 `last_joy_motion_time`，
mux 永远认为手柄"活跃"，于是**持续以 0.2 m/s 前进**。

**⚠️ 关键：命令链上有两层"最小速度"，只堵一层无效。**

| 层 | 参数 | 旧值 | 现值 |
| --- | --- | --- | --- |
| ① `gamepad_command_node` | `deadzone` | 0.12 | **0.15** |
| ② `cmd_vel_mux_node` | `min_output_linear_x / y / angular_z` | 0.2 / 0.1 / 0.2 | **0 / 0 / 0** |
| ③ **`dm_dog_policy_node`** | **`cmd_min_linear_x / y / angular_z`** | **0.22 / 0.33 / 0.30** | **0 / 0 / 0** |
| ③ | `cmd_input_deadband` | 0.0001 | **0.02** |

第 ③ 层是第二次才发现的坑。`dog_policy_node.py:95` 的 `_shape_axis()`：

```python
magnitude = abs(value)
if magnitude <= self.input_deadband: return 0.0
shaped = min_abs + (max_abs - min_abs) * magnitude   # ← min_abs 就是 cmd_min_*
return sign(value) * shaped
```

**只要指令超过 `input_deadband`，输出就至少是 `min_abs`。**
实测：摇杆漂移 0.16 → ①后剩 0.0118 → ②后 0.0071 m/s → ③被抬到 **0.2255 m/s**（狗自己往前走）。
归零后同一路径输出 **0.0000**。

**修法**（launch 已改，共三层）：

| 参数 | 旧值 | 现值 |
| --- | --- | --- |
| `gamepad_command_node` 的 `deadzone` | 0.12 | **0.15** |
| `cmd_vel_mux_node` 的 `min_output_linear_x` | 0.2 | **0.0** |
| `cmd_vel_mux_node` 的 `min_output_linear_y` | 0.1 | **0.0** |
| `cmd_vel_mux_node` 的 `min_output_angular_z` | 0.2 | **0.0** |

归零后同样的漂移只产生约 0.007 m/s，可忽略。

**诊断方法**（不需要上电，这两个节点不依赖 CAN）。摇杆回中后：

```bash
ros2 topic echo /cmd_vel
```

若 `linear.x` 稳定在 ±0.2，就是这个问题。

> 如果之后发现"轻推摇杆狗不动"（策略自身有启动死区），
> 再把 `min_output_linear_x` 单独调到 0.1 左右，不要回到 0.2 以上。

### ⚠️ 启动日志的预期值会与注释不同

launch 注释要求看到 `input_dim=47 ... has_phase=True`，那是**官方 unitree_rl_gym**
（带步态相位时钟，47 维）的契约。**本仓库 go2_rl_gym 产出的是 45 维无相位**，
所以日志显示 `input_dim=45 ... has_phase=False` 是**正常的**。
只要同时有 `command_offset=6 head_order=ang_vel,gravity,cmd` 就说明契约对上了。

---

## 4. ⚠️ 尚未解决的风险：启动序列终点位姿

### 4.1 问题

代码里 RL 的零动作位姿是这样定义的：

```text
raw_target(action=0) = rl_origin
model_q = DEFAULT_DOF_POS + dirs * (raw_q - rl_origin)
观测 joint_pos_error = model_q - DEFAULT_DOF_POS = dirs * (raw_q - rl_origin)
```

两个推论：

1. **`DEFAULT_DOF_POS` 在 RL 路径中被完全抵消**——改它不改变狗的实际姿势。
   （已用数值实验验证：改与不改，观测和 raw_target 逐位相同。）
2. **`FIXED_RL_ORIGIN_POSE_RAW` 默认不生效**——`use_startup_pose_as_rl_origin`
   在节点和 launch 里默认都是 `False`，此时 `rl_origin` 取的是
   **启动序列结束后的实测编码器位姿**。

所以：**零动作时狗摆成什么姿势，完全由 `bs_hardware/startup.py` 的启动序列决定**
（髋展开 → 大腿微调 940 步 → IK 下沉）。而策略是在 Go2 的 `default_joint_angles`
（髋 ±0.1、后大腿 1.0）附近训练的。两者是否一致，**当前无法离线判定**。

### 4.2 处理办法（按顺序做）

**第一次上电时不要开 RL**，先只跑启动序列：

```bash
ros2 run dm_dog_controller dm_dog_policy_node --ros-args \
  -p startup:=true -p auto_rl_after_start:=false -p send_actions:=false
```

观察站姿，并与 Go2 默认角对比：

| 关节 | Go2 默认 | 关注点 |
| --- | --- | --- |
| 髋 (hip) | FL/RR +0.1，FR/RL −0.1 | 是否有外八/内八的展开量 |
| 前大腿 | 0.8 | 前腿是否偏伸 |
| **后大腿** | **1.0** | 与 0.8 差 11.5°，是最需要核对的一项 |
| 小腿 (calf) | −1.5 | 小腿折角 |

程序会打印可粘贴的位姿快照：

```text
standing pose snapshot saved to /tmp/dm_dog_standing_pose.txt
COPY_FIXED_RL_ORIGIN_POSE_RAW = [...]
```

若站姿与 Go2 默认明显不符，两条路：

- **调整启动序列**（改 `startup.py` 的 `HIP_SPREAD_STEP_RAD/HIP_SPREAD_STEPS`、
  `_thigh_adjust` 的迭代次数），让终点位姿匹配 Go2 默认；
- **或者改训练侧**：把 `go2_config.py` 的 `default_joint_angles` 改成实机启动
  序列的位姿后重训，部署端不动。**这是更安全的路线**，因为完全不碰已标定的启动序列。

---

### 4.3 量化风险：策略对名义位姿偏移的容忍度

为了判断这个风险到底有多大，我做了一个鲁棒性实验：在 MuJoCo 里把策略的名义关节角
`default_angles` 整体偏移 Δ（等价于"实机站姿 ≠ 训练默认角"），看策略还能不能走。
用 `go2_ppo_1000.pt`，指令 `vx = 1.0 m/s`：

| 场景 | 名义角偏移 | 基座高 | 前向速度 | 结果 |
| --- | --- | --- | --- | --- |
| 基准 | 0（Go2 默认角） | 0.305 m | +0.791 m/s | ✅ 未摔 |
| 实机站姿对应 legacy 默认角 | 后大腿 ±0.2 rad 等 | 0.324 m | +0.633 m/s | ✅ 未摔 |

**结论：即使实机启动序列的终点位姿对应的是 legacy 默认角（而非 Go2 默认角），
策略仍然能稳定行走**，只是跟踪速度略低（0.633 vs 0.791 m/s），基座高度正常。

所以这一项**不构成上电阻断风险**。首次上电按第 5 节流程正常进行即可；
若实测步态明显不佳（拖腿、后腿蹬不直、身体前倾），再考虑调整启动序列或改成
"用实机站姿重训"的路线。

---

## 5. 首次上电操作清单

> ⚠️ **默认 launch 没有"只站不给 RL"的中间态。**
> launch 里 `ros_command_control: True`，而接管判定是
> `rl_active = auto_rl_after_start or keyboard or gamepad or command_source`
> （`deployment.py`）。只要命令源存在，狗一站稳就**立刻交给 RL**。
> 所以"先只看站姿"必须绕开 launch，用下面的 `ros2 run` 直接起节点。

### 第 0 步：先确认站姿（策略只算不发）

狗**架空或用支架托住**，急停在手，场地清空：

```bash
source install/setup.bash        # 你的 colcon 工作空间
SHARE=$(ros2 pkg prefix dm_dog_controller)/share/dm_dog_controller

ros2 run dm_dog_controller dm_dog_policy_node --ros-args \
  -p model:=$SHARE/policies/go2_cts_150k.pt \
  -p startup:=true -p auto_rl_after_start:=false \
  -p ros_command_control:=false -p send_actions:=false
```

- `startup:=true`：节点启动即跑启动序列，把狗撑起来
- `send_actions:=false`：策略**只算不发**，狗由启动序列保持站姿
- 对照第 4.2 节的 Go2 默认角表核对站姿
- 位姿快照会写到 `/tmp/dm_dog_standing_pose.txt`

### 第 1 步：正式跑（带 RL，限幅 0.3）

```bash
ros2 launch dm_dog_controller dm_dog_policy.launch.py
```

狗初始是趴着的（`initial_standing: false`），**按 LB + A 站起来**，
站起来后立刻进入 RL。观察前 3~5 秒：

- 迈腿方向是否正确（方向错立刻急停）
- 是否高频抖动（增益问题，回看第 1.2 节 PD 增益）
- 是否向一侧倾倒

### 第 2 步：动作幅值与增益（两者必须一起调）

当前组合：**`kp=20 / kd=0.5` + `action_clip=4.0`** → 峰值需求力矩 ≈ 20 Nm。

> **改 `action_clip` 前一定要先确认当前的 PD 增益。**
> 峰值力矩 ≈ `kp × action_clip × 0.25`。
> **实机已验证：升增益会让狗身抖动，所以维持 kp=20。**
> 若之后要动 kp，必须同步调 clip 使 `kp × clip` 仍在 80 左右。
>
> 理想情况下应该在**训练侧**把 `stiffness/damping` 改成实机值重训，
> 这样部署端就能直接用训练值；但在那之前，这组手调值是实用的折中。

### 第 3 步：逐步给速度指令

前进（慢 → 快）→ 转向 → 后退/横移。
注意 `continuous_forward_max: 0.6` 是首飞限速。

### 急停

- 手柄：**LB + RB**（`emergency_lb_button: 6` / `emergency_rb_button: 7`）
- 话题：`/emergency_stop`

### 监控

```bash
# 电机堵转/过载守卫（config.py）
MOTOR_GUARD_ABORT = True                              # 已启用
MOTOR_GUARD_ABORT_POS_ERROR_RAD = 1.3                 # 主判据
```

---

## 6. 回滚到 legacy 策略

```bash
cp bs_hardware/config.py.bak-before-go2-policy bs_hardware/config.py
# launch 里改回：model=policy_stair.pt, use_dual_policy=True, yaw_model=policy.pt
#               forward_speed_mode="fixed", action_clip=1.0
```

---

## 7. 当前策略的来源与复现

- 训练：云端 `go2_rl_gym`，`python legged_gym/scripts/train.py --task=go2 --num_envs 4096 --headless`
- 导出：`play.py` → `logs/go2_ppo/exported/policies/policy.pt`
- 已下载并验证的 checkpoint（MuJoCo 平地，指令 `vx = 1.0 m/s`）：

| 策略 | 前向速度 | 基座高度 | 结果 |
| --- | --- | --- | --- |
| `go2_ppo_500.pt` | 0.693 m/s | 0.293 m | ✅ |
| `go2_ppo_1000.pt` | 0.791 m/s | 0.305 m | ✅ |
| `go2_ppo_5000.pt` | 0.695 m/s | 0.333 m | ✅ |
| **`go2_ppo_7000.pt`（当前）** | **0.899 m/s** | 0.344 m | ✅ |

> ⚠️ **选 checkpoint 不要只看单条 MuJoCo 轨迹。** 上面几个值差异主要来自轨迹差异
> 而非策略质量——训练日志里的 `Episode/rew_tracking_lin_vel` 显示 500~7000 几乎持平
> （0.79~0.83），而 `rew_tracking_ang_vel` 是**单调上升**的（0.390 → 0.415）。
> **结论：用训练日志的 tracking 指标排序，最新的 checkpoint 最好。**

从训练日志提取指标的方法（云端）：

```python
# 在迭代 X 到 X+200 窗口内对 rew_tracking_* 取均值，抑制单迭代噪声
```

- 训练仍在继续，`save_interval=500`
- 云端有现成脚本：`/root/export_and_validate.sh <run> <checkpoint> [地形] [秒数]`
  一步完成「导出 TorchScript + MuJoCo 验证」

### 换 checkpoint 的完整步骤

```bash
# 1) 云端导出（play.py 会把 policy.pt 写到 logs/go2_ppo/exported/policies/）
ssh 183.147.142.40 '/root/export_and_validate.sh <run> <新迭代数> flat.xml 15'

# 2) 拉到本地
scp 183.147.142.40:/root/go2_rl_gym/logs/go2_ppo/exported/policies/policy.pt \
    ~/project/dm_dog_controller/policies/go2_ppo_<新迭代数>.pt

# 3) 改 launch 里的 model 文件名，然后必须重新构建（策略文件是复制安装的）
cd <ros2_ws> && colcon build --packages-select dm_dog_controller
```
