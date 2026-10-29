# dm_dog_controller 机器狗调参手册

**怎么调狗的行为**（PD 增益、速度、姿态、步幅……）
**当前配置**：go2 策略（`go2_ppo_65000_planA.pt`）
**最后更新**：2026-10-16

---

## 零、先搞清楚：参数分两层

**这是最重要的一张图。** 调参前必须知道自己想改的东西在哪一层。

```
┌─────────────────────────────────────────────────────────┐
│  A 层：部署层（可以随时改，改完重启，不用重训）            │
│                                                          │
│   PD 增益        → 腿的僵硬程度 / 软硬                    │
│   RL_ACTION_SCALE→ 动作幅度                                │
│   DEFAULT_DOF_POS→ 站立姿态（站姿几何）                    │
│   速度上限        → 走多快                                 │
│   死区            → 手柄灵敏度                             │
│   action_clip    → 动作限幅（安全）                        │
└─────────────────────────────────────────────────────────┘
                         ↓ 策略推理
┌─────────────────────────────────────────────────────────┐
│  B 层：策略层（★ 改了必须重新训练）                        │
│                                                          │
│   ★ 步幅 ★        ← 学出来的，没有直接参数                  │
│   步态（抬腿高度、迈腿频率、相位）                          │
│   重心偏好 / 姿态倾向                                      │
│   能耗偏好 / 抗扰策略                                      │
│                                                          │
│   由训练侧的【奖励权重】决定：                              │
│     tracking_lin_vel = 1.0     跟踪速度（越大走得越积极）    │
│     feet_regulation  = -0.05   足端规矩（抑制拖步）         │
│     correct_base_height = -1.0 站高（影响步幅/步频）        │
│     torques = -1e-4            力矩惩罚（越大越省力、步幅小）│
│     action_smoothness = -0.01  动作平滑                     │
│     hip_to_default = -0.3      髋回中（影响腿的展开）        │
└─────────────────────────────────────────────────────────┘
```

### ⚠️ 直接回答"步幅怎么调"

| 你的目标 | 能不能直接调 | 怎么做 |
| --- | --- | --- |
| **让狗走快 → 步幅自然变大** | ✅ 能 | 调 `continuous_forward_max` |
| **同样速度下步幅更大** | ❌ 不能直接调 | 必须改训练奖励（`torques` ⬆️ 或 `feet_regulation` 调）后**重训** |
| **抬腿更高（跨越障碍）** | ❌ 不能 | 训练侧加 `feet_clearance` 类奖励，**重训** |
| **步频更快/更慢** | ⚠️ 部分 | 47 维布局有 `OBS_PHASE_PERIOD_S`；45 维布局由策略决定 |

**一句话**：**A 层只能"调外观和手感"，B 层才决定"怎么走"。**

---

## 一、速度类参数（最常用、最安全）

**文件**：`launch/dm_dog_policy.launch.py`

| 参数 | 当前值 | 作用 | 安全范围 |
| --- | --- | --- | --- |
| `continuous_forward_max` | **0.4** | 前进上限 m/s | ≤ 策略训练上限 |
| `continuous_backward_max` | **0.4** | 后退上限 m/s | 建议比前进小 |
| `continuous_lateral_max` | 0.4 | 横移上限 m/s | ≤ 0.4 |
| `continuous_yaw_max` | 1.5 | 转向上限 rad/s | ≤ 2.0 |
| `linear_speed` | 1.0 | 手柄→平动速度的整体缩放 | 0.2~1.0 |
| `yaw_speed` | 1.0 | 手柄→转速的整体缩放 | 0.2~1.0 |
| `forward_speed_mode` | `continuous` | 速度随摇杆连续变化 | `continuous` / `fixed` |
| `fixed_forward_speed` | 0.5 | 仅 `fixed` 模式：任何推动=这个速度 | ≤ 0.6 |
| `deadzone` | **0.20** | 手柄死区 | 0.15~0.30 |
| `cmd_input_deadband` | **0.15** | 控制器侧再挡一次小指令 | 0.05~0.20 |

### ⚠️ 速度上限不能随便提

**策略只在它训练过的速度范围内可靠。** 当前 go2 策略的训练范围：

| 绝对迭代 | 训练速度范围 |
| --- | --- |
| 0 ~ 20000 | ±0.5 m/s |
| 20000 ~ 50000 | **±1.0 m/s** |
| 50000 以后 | ±2.0 m/s |

**当前策略（65000 迭代）训练过 ±2.0**，所以 0.4~0.6 都很安全 ✓

**如果换成只训到 ±0.5 的策略，`continuous_forward_max` 就不能超过 0.5。**

### 改法（launch 免重建）

```bash
cd <repo>/dm_dog_controller
cp -a launch/dm_dog_policy.launch.py launch/dm_dog_policy.launch.py.bak-before-speed
sed -i 's/"continuous_forward_max": 0.4,/"continuous_forward_max": 0.5,/' launch/dm_dog_policy.launch.py
grep -n "continuous_forward_max" launch/dm_dog_policy.launch.py
# 直接重启 ros2 launch 即可生效（用了 --symlink-install）
```

---

## 二、PD 增益（腿的软硬）

**文件**：`bs_hardware/config.py`

| 常量 | 当前值 | 什么时候用 |
| --- | --- | --- |
| `STARTUP_KP` / `STARTUP_KD` | **30.0 / 1.0** | 启动站立序列 |
| `BM_CALF_STARTUP_KP` / `_KD` | **40.0 / 20.0** | 启动时小腿（BM 电机）专用 |
| `TRANSLATION_POLICY_RL_KP` / `_KD` | **25.0 / 2.0** | **正常走路** |
| `TRANSLATION_POLICY_BM_CALF_KP` / `_KD` | **25.0 / 2.0** | 走路时小腿专用 |
| `YAW_POLICY_RL_KP` / `_KD` | 20.0 / 1.5 | 转向策略（双策略模式） |
| `RL_ACTION_SCALE` | **0.125** | 动作 → 目标角偏移的缩放 |

### 🔴 改之前必须算峰值力矩

```
最大偏移   = action_clip × RL_ACTION_SCALE          [rad]
最大 P 力矩 = kp × action_clip × RL_ACTION_SCALE     [N·m]
```

**当前**：`25.0 × 2.0 × 0.125 = 6.25 N·m`

**真机电机能力：**

| | 持续额定 | 峰值 |
| --- | --- | --- |
| 髋 / 大腿（550g） | **6 N·m** | 23.5 N·m |
| 小腿（730g） | **8 N·m** | 40 N·m |

**6.25 N·m 刚好压在髋的持续额定上 —— 安全。**

| 如果你改 | 峰值力矩 | 后果 |
| --- | --- | --- |
| `kp` → 40 | 10.0 N·m | 超持续额定 1.7 倍，**长跑会烫** |
| `kp` → 50 | 12.5 N·m | 超 2 倍，**有烧的风险** |
| `action_clip` → 4 | 12.5 N·m | 同上 |
| `RL_ACTION_SCALE` → 0.25 | 12.5 N·m | 同上，且偏离训练设定 |

**快速验算：**
```bash
python3 -c "
kp=25.0; clip=2.0; scale=0.125
peak = kp*clip*scale
print(f'峰值力矩 = {peak:.2f} N·m   (髋持续 6 / 小腿持续 8)')
print('⚠️ 超过 6 就要盯温度' if peak>6 else '✅ 安全')
"
```

### 调增益的方向

| 想达到 | 改 | 方向 |
| --- | --- | --- |
| 腿更硬、支撑更强 | `TRANSLATION_POLICY_RL_KP` | ⬆️ |
| 腿更软、更柔顺 | `TRANSLATION_POLICY_RL_KP` | ⬇️ |
| 抑制高频抖动 | `TRANSLATION_POLICY_RL_KD` | ⬆️（先降 KP） |
| 抑制小腿抖动 | `TRANSLATION_POLICY_BM_CALF_KD` | ⬆️ |
| 站起来更缓和 | `STARTUP_KP` | ⬇️ |
| 动作幅度更大 | `RL_ACTION_SCALE` | ⬆️（**重算力矩**） |

**⚠️ 小腿必须单独调**：小腿是 BM（达妙）电机，和髋/大腿的电机
型号、减速比不同，共用增益会振或软。`motor_bus.py` 里通过
`bm_calf_gain_override` 单独设置。

### 改法（config 必须重建）

```bash
cd <repo>/dm_dog_controller
cp -a bs_hardware/config.py bs_hardware/config.py.bak-before-kp-$(date +%m%d-%H%M)
sed -i 's/^TRANSLATION_POLICY_RL_KP = 25.0/TRANSLATION_POLICY_RL_KP = 30.0/' bs_hardware/config.py
sed -i 's/^TRANSLATION_POLICY_BM_CALF_KP = 25.0/TRANSLATION_POLICY_BM_CALF_KP = 30.0/' bs_hardware/config.py
grep -nE "^TRANSLATION_POLICY" bs_hardware/config.py

cd <ros2_ws> && source /opt/ros/humble/setup.bash
colcon build --packages-select dm_dog_controller --symlink-install
source install/setup.bash

# 验证真的生效了
python3 -c "
import sys; sys.path.insert(0,'$(ros2 pkg prefix dm_dog_controller)/local/lib/python3.10/dist-packages')
from bs_hardware.config import TRANSLATION_POLICY_RL_KP, RL_ACTION_SCALE
print('KP =', TRANSLATION_POLICY_RL_KP, ' ACTION_SCALE =', RL_ACTION_SCALE)
"
```

---

## 三、姿态参数（站立姿势）

**文件**：`bs_hardware/config.py`

### 3.1 `DEFAULT_DOF_POS` —— 模型坐标系下的站立默认角

```python
DEFAULT_DOF_POS = [
     0.1, 0.8, -1.5,   # FL 髋/大腿/小腿
    -0.1, 0.8, -1.5,   # FR
     0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5,   # RR
]
```

**作用**：
- 观测里的 `(dof_pos − DEFAULT_DOF_POS)` 用它
- 动作目标 `DEFAULT_DOF_POS + action × RL_ACTION_SCALE` 的**起点**

**⚠️ 必须等于训练时 URDF 的 `default_joint_angles`**，否则观测有常值偏置。

**能怎么调**：
| 想达到 | 改哪个数 |
| --- | --- |
| 站得更高 | 大腿角 ⬆️（减小）、小腿角绝对值 ⬇️ |
| 站得更低 | 反之 |
| 腿更外展（防碰） | 髋角（第 1、3 个腿的前 3 项）绝对值 ⬆️ |
| 前后腿更分开 | RL/RR 的大腿角 ⬆️ |

**⚠️ 改这个会同时改变观测和动作起点，是"大改"，要小心。**

### 3.2 `FIXED_RL_ORIGIN_POSE_RAW` —— 真机的站立姿态（原始电机弧度）

```python
FIXED_RL_ORIGIN_POSE_RAW = [-0.904669, 3.151560, 0.449655, ...]
```

**作用**：策略动作的**基准姿态**（真机电机原始坐标）。

**为什么存在**：正常流程是启动序列站好后由 `startup_result` 提供；
这个常量是**手动固化**一个已经验证过的站立姿态，让每次启动姿态**可复现**。

**注意**：这是**原始电机弧度**，不是模型弧度（受 `MODEL_JOINT_DIRECTIONS` 影响）。

### 3.3 `MODEL_JOINT_DIRECTIONS` —— 模型 → 电机的方向符号

```python
MODEL_JOINT_DIRECTIONS = [
    [-1.0, -1.0, -1.0],   # FL
    [-1.0,  1.0,  1.0],   # FR
    [ 1.0, -1.0, -1.0],   # RL
    [ 1.0,  1.0,  1.0],   # RR
]
```

**作用**：把"模型坐标系的正方向"翻译成"电机坐标系的正方向"。

**🔴 不要乱改** —— 改错了腿会朝反方向运动（和 2026-10-15 那次"上电发癫"类似）。
改之前先看 `STARTUP_JOINT_DIRECTIONS` 的注释。

---

## 四、安全保护参数

**文件**：`bs_hardware/config.py`

| 常量 | 当前值 | 作用 |
| --- | --- | --- |
| `MOTOR_GUARD_WARN_POS_ERROR_RAD` | 0.8 | 位置误差告警阈值 |
| `MOTOR_GUARD_WARN_TORQUE_NM` | 12.0 | 力矩告警阈值 |
| **`MOTOR_GUARD_ABORT_POS_ERROR_RAD`** | **1.3** | **位置误差超这个 → 中止** |
| **`MOTOR_GUARD_ABORT_TORQUE_NM`** | **15.5** | **力矩超这个 → 中止** |
| `MOTOR_GUARD_ABORT_SECONDS` | 0.3 | 要连续超这么久才中止 |
| `MOTOR_GUARD_ABORT` | True | 总开关 |
| `MOTOR_INIT_SETTLE_SECONDS` | 5.0 | 电机初始化后的稳定等待 |

**这些是保命的，一般不要关。** 如果你发现狗在正常走路时被误中止，
可以把 `ABORT_TORQUE_NM` 适当提到 18（BM 小腿的额定就是 ±18）。

**launch 里的相关项：**

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `action_clip` | **2.0** | 策略动作裁剪（改了要算峰值力矩） |
| `feedback_timeout` | 0.5 | 收不到电机反馈多久后停 |
| `cmd_max_linear_x` | 1.0 | 控制器侧硬上限 |
| `cmd_min_linear_x` | 0.0 | 最小速度下限（0=不抬速） |
| `shutdown_lie_down` | True | 退出时趴下而不是直接断电 |

---

## 五、观测相关（一般不动）

**文件**：`bs_hardware/config.py`

| 常量 | 当前值 | 作用 |
| --- | --- | --- |
| `OBS_DIM` | 45 | 观测维度（**必须和策略匹配**） |
| `OBS_HEAD_ORDER` | `"ang_vel,gravity,cmd"` | 观测头顺序 |
| `COMMAND_SCALE` | [2.0, 2.0, 0.25] | 速度指令在观测里的缩放 |
| `ANGULAR_VELOCITY_SCALE` | 0.25 | 角速度缩放 |
| `DOF_VELOCITY_SCALE` | 0.05 | 关节速度缩放 |
| `HISTORY_LENGTH` | 5 | 历史帧数（本策略=1） |
| `OBS_PHASE_PERIOD_S` | 0.8 | 步态时钟周期（**仅 47 维布局用**） |

**⚠️ 这些改了就加载不了策略** —— 它们必须和训练时完全一致。

---

## 六、症状 → 调哪个参数（速查表）

| 症状 | 调 | 方向 | 注意 |
| --- | --- | --- | --- |
| 软趴趴、站不住 | `TRANSLATION_POLICY_RL_KP` | ⬆️ 25→30 | **算峰值力矩** |
| 高频抖动 / 嗡嗡响 | `TRANSLATION_POLICY_RL_KD` | ⬆️ 2.0→2.5 | 先降 KP 再提 KD |
| 只有小腿抖 | `TRANSLATION_POLICY_BM_CALF_KD` | ⬆️ | 小腿是 BM 电机 |
| 站起来猛冲一下 | `STARTUP_KP` | ⬇️ 30→25 | 或 `BM_CALF_STARTUP_KP` ⬇️ |
| 走不动、动作幅度小 | `RL_ACTION_SCALE` | ⬆️ 0.125→0.15 | **峰值力矩等比涨** |
| 最快速度不够 | `continuous_forward_max` | ⬆️ 0.4→0.5 | **不能超训练范围** |
| 狗自己慢慢走（手柄没动） | `deadzone` | ⬆️ 0.20→0.25 | 手柄漂移 |
| 推一点点不动，推多了猛走 | `cmd_input_deadband` | ⬇️ 0.15→0.10 | |
| 转向太灵敏 | `continuous_yaw_max` | ⬇️ 1.5→1.0 | |
| 走直线会歪 | `MODEL_JOINT_DIRECTIONS` | 检查符号 | **危险，别乱改** |
| 走路被误中止 | `MOTOR_GUARD_ABORT_TORQUE_NM` | ⬆️ 15.5→18 | 仅当确认是误报 |
| 想改**步幅** | ❌ 不能直接调 | — | **改训练奖励后重训** |
| 想改**抬腿高度** | ❌ 不能直接调 | — | **改训练奖励后重训** |

---

## 七、改完的验证流程（每次都要做）

```bash
cd <repo>/dm_dog_controller

# 1. 语法检查
python3 -c "import ast; ast.parse(open('bs_hardware/config.py').read()); print('config.py OK')"
python3 -c "import ast; ast.parse(open('launch/dm_dog_policy.launch.py').read()); print('launch OK')"

# 2. 峰值力矩
python3 -c "
kp=25.0; clip=2.0; scale=0.125
print(f'峰值力矩 = {kp*clip*scale:.2f} N·m')"

# 3. 构建（只改 launch 可跳过）
cd <ros2_ws> && source /opt/ros/humble/setup.bash
colcon build --packages-select dm_dog_controller --symlink-install
source install/setup.bash

# 4. 冒烟测试（不接硬件）
S=$(ros2 pkg prefix dm_dog_controller)/share/dm_dog_controller
ros2 run dm_dog_controller dm_dog_policy_node --ros-args \
  -p model:=$S/policies/go2_ppo_65000_planA.pt -p use_imu:=false -p use_can:=false \
  -p startup:=false -p send_actions:=false -p ros_command_control:=false -p steps:=1

# 5. 上机（架空、急停在手、先只站起来）
ros2 launch dm_dog_controller dm_dog_policy.launch.py
```

---

## 八、回退

所有改动前都建了备份：

```bash
cd <repo>/dm_dog_controller
ls -la bs_hardware/config.py.bak* launch/dm_dog_policy.launch.py.bak*

# 回退某个改动
cp -a bs_hardware/config.py.bak-before-kp-1016-2030 bs_hardware/config.py
cd <ros2_ws> && source /opt/ros/humble/setup.bash
colcon build --packages-select dm_dog_controller --symlink-install
```

**备份命名规律**：`*.bak-before-<改动名>`

---

## 九、文件位置速查

| 项 | 路径 |
| --- | --- |
| 工程根 | `<repo>/dm_dog_controller` |
| **PD 增益 / 姿态 / 安全** | `bs_hardware/config.py` |
| **速度 / 死区 / action_clip** | `launch/dm_dog_policy.launch.py` |
| 电机总线（小腿专用增益） | `bs_hardware/motor_bus.py` |
| 启动序列 | `bs_hardware/startup.py` |
| 策略推理 / 动作管线 | `dm_dog_controller/deployment.py` |
| 手柄映射 | `dm_dog_controller/gamepad_command_node.py` |
| 速度整形（前进/后退限幅） | `dm_dog_controller/cmd_vel_shaping.py` |
| 策略文件 | `policies/*.pt` |
| ROS 工作区 | `<ros2_ws>` |
| **训练侧奖励权重（改步幅的地方）** | `go2_rl_gym/legged_gym/envs/go2/go2_config.py` |

---

## 十、一句话总结

> **A 层（部署）能调"软硬、快慢、站姿、灵敏度"—— 改完重启就行。**
> **B 层（策略）决定"怎么迈腿、步幅多大、抬多高"—— 想改必须重训。**
>
> 想调行为前先问自己："这是手感问题，还是步态问题？"
