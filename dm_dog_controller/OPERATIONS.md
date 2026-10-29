# 操作手册：换策略 / 改参数

> 面向日常自己动手调试。**改完只需重启节点，通常不用 `colcon build`。**

---

## 0. 核心机制：工作空间已用 `--symlink-install` 构建

安装目录里的文件是**符号链接**，指向源码：

```
install/dm_dog_controller/local/lib/python3.10/dist-packages/bs_hardware/config.py
    -> src/dm_dog_controller/bs_hardware/config.py

install/.../share/dm_dog_controller/launch/dm_dog_policy.launch.py
    -> src/dm_dog_controller/launch/dm_dog_policy.launch.py

install/.../share/dm_dog_controller/policies/*.pt
    -> src/dm_dog_controller/policies/*.pt
```

**所以：改源码 = 立即生效。** 只有**新增**文件（比如往 `policies/` 里加一个新的 `.pt`）才需要重建。

---

## 1. 换策略（最常用）

我留了一个**固定文件名** `policies/active.pt`，launch 只引用它。

```bash
# 1) 把想用的策略复制成 active.pt（覆盖即可）
cp ~/project/dm_dog_controller/policies/go2_cts_150k.pt \
   ~/project/dm_dog_controller/policies/active.pt

# 2) 重启节点（Ctrl-C 后重新 launch，或只重启 dm_dog_policy_node）
ros2 launch dm_dog_controller dm_dog_policy.launch.py
```

**不用 `colcon build`。** 确认是否生效，看启动日志：

```
loaded translation_policy: path=.../policies/active.pt entry=forward
  input_dim=45 per_frame=45 history=1 has_phase=False obs_layout=legacy-45
  command_offset=6 head_order=ang_vel,gravity,cmd
```

想确认当前是哪个策略：

```bash
md5sum ~/project/dm_dog_controller/policies/active.pt
md5sum ~/project/dm_dog_controller/policies/*.pt   # 对上哪个就是哪个
```

**可用的策略清单**（都在 `policies/`）：

| 文件 | 输入维 | 相位 | 备注 |
| --- | --- | --- | --- |
| `go2_cts_150k.pt` | 45 | 无 | 已测：腿打架、停不下来 |
| `go2_him_21k_0.5379.pt` | 45 | 无 | 已测：更怪 |
| `go2_dwaq_119.5k_0.5054.pt` | 45 | 无 | 已测 |
| `go2_ppo_*.pt` | 45 | 无 | 自训 |
| `go2dog_policy_1.15M.pt`（在上级目录） | **47** | **有** | 需改 `OBS_HEAD_ORDER`，见第 5 节 |
| `policy.pt` / `policy_stair.pt`（上级目录） | 225 | 无 | **你自己的旧策略，已知能走** |

> 你包根目录（`~/project/dm_dog_controller/`）下的 `.pt` 也要先 `cp` 进 `policies/` 才能被 launch 引用。

---

## 2. 参数改在哪

### A. `bs_hardware/config.py`（Python，改完直接生效）

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `TRANSLATION_POLICY_RL_KP / KD` | `20.0 / 1.5` | **RL 关节增益** |
| `TRANSLATION_POLICY_BM_CALF_KP / KD` | `20.0 / 1.5` | BM 小腿电机增益 |
| `YAW_POLICY_*` | 同上 | `use_dual_policy=False` 时不生效 |
| `STARTUP_KP / KD` | `30.0 / 1.0` | **启动序列**撑站姿（与 RL 独立） |
| `BM_CALF_STARTUP_KP / KD` | `40.0 / 20.0` | 同上 |
| `MODEL_JOINT_DIRECTIONS` | `[ [-1,-1,-1], [-1,1,1], [1,-1,-1], [1,1,1] ]` | 关节方向标定（**只在实机验证后才改**） |
| `DEFAULT_DOF_POS` | Go2 默认角 | ⚠️ 在 RL 路径里会被抵消，改了没用 |
| `OBS_HEAD_ORDER` | `"ang_vel,gravity,cmd"` | 观测前 9 维顺序，改它等于换策略接口 |
| `HIP_SPREAD_STEPS`（在 `startup.py`） | `139` | 站立髋外展量 |

### B. `launch/dm_dog_policy.launch.py`（改完直接生效）

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `model` | `policies/active.pt` | **策略文件** |
| `action_clip` | `2.0` | 动作限幅。**与 kp 耦合**：峰值力矩 ≈ `kp × clip × 0.25` |
| `hz` | `50.0` | 控制频率（必须 50，训练就是 50） |
| `deadzone`（gamepad 节点） | `0.15` | 摇杆死区 |
| `min_output_*`（mux 节点） | `0 / 0 / 0` | 最小输出塑形，**已归零**（见第 4 节） |
| `cmd_min_linear_x / y / angular_z` | `0 / 0 / 0` | dm 侧最小速度，**已归零**（见第 4 节） |
| `cmd_input_deadband` | `0.02` | dm 侧指令死区 |
| `use_dual_policy` | `False` | 双策略开关 |
| `forward_speed_mode` | `"continuous"` | 摇杆幅度线性映射 |
| `continuous_forward_max` | `0.6` | 前进最大速度（训练范围到 1.0，可提到 1.0） |
| `continuous_lateral_max` / `yaw_max` | `0.4` / `1.0` | 同上 |
| `startup` | `False` | 启动时是否自动跑启动序列 |
| `send_actions` | `True` | 是否真的发指令给电机 |

---

## 3. 标准流程

```bash
# 改文件（config.py 或 launch）
vim ~/project/dm_dog_controller/bs_hardware/config.py

# 重启节点
cd <ros2_ws> && source install/setup.bash
ros2 launch dm_dog_controller dm_dog_policy.launch.py
```

**只有当新增了文件**时：

```bash
cd <ros2_ws> && source /opt/ros/humble/setup.bash
colcon build --packages-select dm_dog_controller --symlink-install
```

> ⚠️ **务必带 `--symlink-install`**。不带的话会把安装目录变回**副本**，之后改源码就不再自动生效了。

### 安全冒烟测试（不接硬件，验证能加载）

```bash
cd <ros2_ws> && source install/setup.bash
SHARE=$(ros2 pkg prefix dm_dog_controller)/share/dm_dog_controller
ros2 run dm_dog_controller dm_dog_policy_node --ros-args \
  -p model:=$SHARE/policies/active.pt \
  -p use_imu:=false -p use_can:=false -p startup:=false \
  -p send_actions:=false -p gamepad_control:=false -p steps:=1
```

---

## 4. ⚠️ 命令链上有三层"最小值/死区"，改任意一层都要看另外两层

"摇杆没动狗却往前走"就是这三层的组合造成的。

| 层 | 文件 | 参数 | 现值 |
| --- | --- | --- | --- |
| ① 摇杆 | `config.py`/launch `deadzone` | 死区 | `0.15` |
| ② mux | launch（`cmd_vel_mux_node`） | `min_output_linear_x/y/angular_z` | `0 / 0 / 0` |
| ③ dm 节点 | launch（`dm_dog_policy_node`） | `cmd_min_linear_x/y/angular_z` | `0 / 0 / 0` |
| ③ dm 节点 | 同上 | `cmd_input_deadband` | `0.02` |

第 ③ 层的逻辑（`dog_policy_node.py:95`）：

```python
if magnitude <= self.input_deadband: return 0.0
shaped = min_abs + (max_abs - min_abs) * magnitude   # ← min_abs 就是 cmd_min_*
```

**只要指令超过 `input_deadband`，输出就至少是 `min_abs`。** 三层任意一层不为 0，都会让微小漂移变成可见的前进指令。

**排查手法**：

```bash
ros2 topic echo /cmd_vel          # ① 摇杆回中后应稳定为 0
ros2 topic echo /joy --once       # ② 看原始轴是否停在 0（手柄硬件漂移）
```

---

## 5. 常用配方

### 换增益（实机抖动/太软时）

```python
# config.py
TRANSLATION_POLICY_RL_KP = 20.0
TRANSLATION_POLICY_RL_KD = 1.5
TRANSLATION_POLICY_BM_CALF_KP = 20.0
TRANSLATION_POLICY_BM_CALF_KD = 1.5
```

> **kp 决定带宽，kd 决定阻尼。** 腿自激/抖 → 提 kd；整机高频抖 → 说明 kp 太高了。
> 已试过：`120/2.2` 偏硬、`60/1.5` 整机抖、`20/0.5` 腿自激、`20/1.5` 当前。

### 改动作幅值

```python
# launch
"action_clip": 2.0,
```

**必须与 kp 一起看**：峰值力矩 ≈ `kp × action_clip × 0.25`
- kp=20 + clip=2.0 → 10 Nm
- kp=20 + clip=4.0 → 20 Nm
- kp=120 + clip=4.0 → 120 Nm（实机"上电过猛"）

### 用 47 维策略（如 `go2dog_policy_1.15M.pt`）

需要同时改两处：

```python
# config.py
OBS_HEAD_ORDER = "gravity,cmd,ang_vel"   # 原值 "ang_vel,gravity,cmd"
```
```bash
cp ~/project/dm_dog_controller/go2dog_policy_1.15M.pt \
   ~/project/dm_dog_controller/policies/active.pt
```

> `config.py` 注释里写明：这个 SB3 策略用契约顺序时"会把重力当成偏航指令，一直顶大腿力矩"。
> **换回 45 维策略时必须把 `OBS_HEAD_ORDER` 改回来。**

### 用你自己旧的双策略

```python
# launch
"model": .../policies/policy_stair.pt,
"use_dual_policy": True,
"yaw_model": .../policies/policy.pt,
```

---

## 6. 回退

| 改动 | 回退 |
| --- | --- |
| PD 增益 | 改回 `120.0 / 2.2`，BM calf `100.0 / 20.0` |
| `action_clip` | 改回 `4.0` |
| `MODEL_JOINT_DIRECTIONS` | 髋列改回 `[-1, -1, 1, 1]` |
| 策略 | `cp policies/go2_cts_150k.pt policies/active.pt` |
| 参数整体 | `cp bs_hardware/config.py.bak-before-go2-policy bs_hardware/config.py` |

---

## 7. 训练（如需继续）

训练已暂停，备份在 `training_backup/`（本地）+ 云端持久卷。

```bash
ssh 183.147.142.40
bash /root/resume_go2_train.sh 30000       # 追加 30000 迭代
source /opt/conda/etc/profile.d/conda.sh && conda activate isaacgym
tensorboard --logdir /root/gpufree-data/go2_rl_gym_logs --port 6006 --bind_all
```

导出新 checkpoint 并放进 `policies/`：

```bash
ssh 183.147.142.40 '/root/export_and_validate.sh Oct02_19-08-27_ <迭代数> flat.xml 15'
scp 183.147.142.40:/root/go2_rl_gym/logs/go2_ppo/exported/policies/policy.pt \
    ~/project/dm_dog_controller/policies/new.pt
# 然后 cp 到 active.pt
```
