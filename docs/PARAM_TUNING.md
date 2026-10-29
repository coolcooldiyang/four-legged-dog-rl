# dm_dog_controller 参数修改手册

**当前生效**：go2 策略（`go2_ppo_65000_planA.pt`）+ 经验调出的增益
**最后更新**：2026-10-16

---

## 零、先记住三条

1. **改 PD 增益前先算峰值力矩**（第一节有公式），别把电机烧了
2. **一次只改一个参数**，改完测试，再改下一个 —— 不然不知道是哪个起的作用
3. **改前先备份**，每个改动都要能回退

---

## 一、🔑 峰值力矩公式（安全第一）

控制器给电机的力矩是：

```
τ = kp × (目标角 − 当前角) + kd × (0 − 当前角速度)
```

最大目标偏移由 **动作裁剪 × 动作缩放** 决定：

```
最大偏移 = action_clip × RL_ACTION_SCALE          [rad]
最大 P 力矩 = kp × action_clip × RL_ACTION_SCALE   [N·m]
```

### 当前值代入

```
kp=25.0, action_clip=2.0, RL_ACTION_SCALE=0.125
最大偏移  = 2.0 × 0.125 = 0.25 rad
最大 P 力矩 = 25.0 × 2.0 × 0.125 = 6.25 N·m
```

### 与真机电机的对比

| | 持续额定 | 峰值 |
| --- | --- | --- |
| 髋（550g 电机） | 6 N·m | 23.5 N·m |
| 大腿（550g） | 6 N·m | 23.5 N·m |
| **小腿（730g）** | **8 N·m** | **40 N·m** |

**当前 6.25 N·m 刚好压在髋/大腿的持续额定上** —— 这个配置是安全的。

### ⚠️ 三条红线

| 如果你把... | 后果 |
| --- | --- |
| `kp` 提到 **50** | 峰值 12.5 N·m → 超过髋持续额定 2 倍，**长时间跑会烧** |
| `action_clip` 提到 **4** | 峰值也是 12.5 N·m，同上下场 |
| `RL_ACTION_SCALE` 提到 **0.25** | 峰值 12.5 N·m，且**策略行为和训练时偏差更大** |

**每次改完都要重算这个公式。** 快速算法：

```bash
python3 -c "
kp=25.0; clip=2.0; scale=0.125
print(f'峰值力矩 = {kp*clip*scale:.2f} N·m  (髋持续 6 / 小腿持续 8)')
"
```

---

## 二、PD 增益在哪、怎么用

**全部在**：`bs_hardware/config.py`

| 常量 | 当前值 | 什么时候用 |
| --- | --- | --- |
| `STARTUP_KP` / `STARTUP_KD` | **30.0 / 1.0** | **启动站立序列**（慢慢站起来那一段） |
| `BM_CALF_STARTUP_KP` / `_KD` | **40.0 / 20.0** | 启动时小腿（BM 电机）的**专用覆盖增益** |
| `TRANSLATION_POLICY_RL_KP` | **25.0** | **正常走路**（前后平移策略） |
| `TRANSLATION_POLICY_RL_KD` | **2.0** | 同上 |
| `TRANSLATION_POLICY_BM_CALF_KP` | **25.0** | 走路时小腿（BM 电机）的专用增益 |
| `TRANSLATION_POLICY_BM_CALF_KD` | **2.0** | 同上 |
| `YAW_POLICY_RL_KP` / `_KD` | **20.0 / 1.5** | 转向策略（`use_dual_policy: True` 时才用） |
| `YAW_POLICY_BM_CALF_KP` / `_KD` | 20.0 / 1.5 | 同上 |
| `RL_ACTION_SCALE` | **0.125** | **动作 → 目标角偏移的缩放** |

**⚠️ 为什么小腿有单独的增益？**
因为小腿用的是 **BM（达妙）电机**，和髋/大腿的电机型号、减速比不同，
用同一组增益会**振**或者**软**。代码里通过 `motor_bus.py` 的
`bm_calf_gain_override` 单独设置。

**注意别的文件不要动**：`bs_hardware/config.sync-conflict-*.py` 是 Syncthing
冲突残留文件，**不被任何代码引用**（增益值也不同，别混淆）。

---

## 三、怎么改（以调 KP 为例）

### 1. 备份

```bash
cd <repo>/dm_dog_controller
cp -a bs_hardware/config.py bs_hardware/config.py.bak-before-kp-$(date +%m%d-%H%M)
```

### 2. 改

```bash
# 例：把走路 kp 从 25 改成 30
sed -i 's/^TRANSLATION_POLICY_RL_KP = 25.0/TRANSLATION_POLICY_RL_KP = 30.0/' bs_hardware/config.py
# 小腿同步改（一般要一起改，保持比例）
sed -i 's/^TRANSLATION_POLICY_BM_CALF_KP = 25.0/TRANSLATION_POLICY_BM_CALF_KP = 30.0/' bs_hardware/config.py
```

### 3. 核对

```bash
grep -nE "^TRANSLATION_POLICY" bs_hardware/config.py
python3 -c "
kp=30.0; clip=2.0; scale=0.125
print(f'峰值力矩 = {kp*clip*scale:.2f} N·m  ← 别超过 6~8')
"
```

### 4. 构建（**改 Python 常量需要重建才能生效**）

```bash
cd <ros2_ws>
source /opt/ros/humble/setup.bash
colcon build --packages-select dm_dog_controller --symlink-install
source install/setup.bash
```

### 5. 验证 install 里真的生效了

```bash
python3 -c "
import sys; sys.path.insert(0,'$(ros2 pkg prefix dm_dog_controller)/local/lib/python3.10/dist-packages')
from bs_hardware.config import TRANSLATION_POLICY_RL_KP, RL_ACTION_SCALE
print('KP =', TRANSLATION_POLICY_RL_KP, ' ACTION_SCALE =', RL_ACTION_SCALE)
"
```

### 6. 回退

```bash
cd <repo>/dm_dog_controller
cp -a bs_hardware/config.py.bak-before-kp-* bs_hardware/config.py
cd <ros2_ws> && source /opt/ros/humble/setup.bash
colcon build --packages-select dm_dog_controller --symlink-install
```

---

## 四、症状 → 调哪个参数（速查表）

| 症状 | 先调 | 方向 | 注意 |
| --- | --- | --- | --- |
| 狗**软趴趴**、站不住、腿打弯 | `TRANSLATION_POLICY_RL_KP` | ⬆️ 25→30 | **先算峰值力矩** |
| 狗**高频抖动/嗡嗡响** | `TRANSLATION_POLICY_RL_KD` | ⬆️ 2.0→2.5 | 先降 KP 再提 KD |
| **只有小腿**抖 | `TRANSLATION_POLICY_BM_CALF_KD` | ⬆️ | 小腿是 BM 电机 |
| 站起来**猛冲一下** | `STARTUP_KP` | ⬇️ 30→25 | 或 `BM_CALF_STARTUP_KP` ⬇️ |
| 动作**幅度太小**、走不动 | `RL_ACTION_SCALE` | ⬆️ 0.125→0.15 | **峰值力矩会等比涨** |
| 狗**自己慢慢走**（手柄没动） | `deadzone`（launch） | ⬆️ 0.20→0.25 | 手柄漂移 |
| 推一点点**不动**，推多了**猛走** | `cmd_input_deadband`（launch） | ⬇️ 0.15→0.10 | |
| **最快速度不够** | `continuous_forward_max`（launch） | ⬆️ 0.4→0.5 | **不能超过策略训练范围** |
| 狗**走直线会歪** | `MODEL_JOINT_DIRECTIONS` | 检查符号 | **危险，别乱改** |

---

## 五、launch 里的参数（免重建，改完重启即可）

**文件**：`launch/dm_dog_policy.launch.py`

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `linear_speed` | 1.0 | 手柄 → 速度的整体缩放 |
| `yaw_speed` | 1.0 | 手柄 → 转向的整体缩放 |
| `deadzone` | **0.20** | 手柄死区（小于它输出 0） |
| `continuous_forward_max` | **0.4** | 前进上限 m/s |
| `continuous_backward_max` | **0.4** | 后退上限 m/s |
| `continuous_lateral_max` | 0.4 | 横移上限 m/s |
| `continuous_yaw_max` | 1.5 | 转向上限 rad/s |
| `forward_speed_mode` | `continuous` | `continuous`=速度随摇杆 / `fixed`=固定速度 |
| `fixed_forward_speed` | 0.5 | 仅 `fixed` 模式生效 |
| `cmd_input_deadband` | **0.15** | 控制器侧再挡一次小指令 |
| `action_clip` | **2.0** | 策略动作裁剪（**改了要算峰值力矩**） |
| `cmd_max_linear_x` | 1.0 | 控制器侧硬上限 |
| `policy_gain_transition_duration` | 0.4 | 切策略时增益渐变时间（秒） |

**改 launch 后**：因为用了 `--symlink-install`，**重启 `ros2 launch` 即可生效**
（不需要 `colcon build`）。

---

## 六、⚠️ 重要：部署增益 ≠ 训练增益

| | KP | KD | action_scale | 峰值力矩 |
| --- | --- | --- | --- | --- |
| **go2 训练时** | 20.0 | 0.5 | 0.25 | 10.0 N·m |
| **go1dm 训练时** | 40.0 | 1.0 | 0.25 | 20.0 N·m |
| **部署当前** | **25.0** | **2.0** | **0.125** | **6.25 N·m** |

**三者都不一致。**

**这意味着**：
- 部署的增益是**经验调出来的**（你之前试出来的能走的组合），不是照搬训练值
- 策略在部署时的**行为会和仿真里有偏差** —— 因为它训练时假设的 PD 响应不同
- 这不是"bug"，是**现实妥协**：训练用大 gain 学出激进的动作，
  部署时用小 gain 是为了**真机安全**（不烧电机）

**如果你想让策略表现更接近仿真**：把部署的 `kp` 和 `RL_ACTION_SCALE`
往训练值靠 —— 但**峰值力矩会涨到 20 N·m，远超电机持续额定**，
只适合**短时间**测试，且**必须盯温度**。

---

## 七、改完的验证流程（每次都要做）

```bash
# 1. 语法检查
python3 -c "import ast; ast.parse(open('bs_hardware/config.py').read()); print('OK')"

# 2. 构建
cd <ros2_ws> && source /opt/ros/humble/setup.bash
colcon build --packages-select dm_dog_controller --symlink-install

# 3. 冒烟测试（不接硬件）
source install/setup.bash
S=$(ros2 pkg prefix dm_dog_controller)/share/dm_dog_controller
ros2 run dm_dog_controller dm_dog_policy_node --ros-args \
  -p model:=$S/policies/go2_ppo_65000_planA.pt -p use_imu:=false -p use_can:=false \
  -p startup:=false -p send_actions:=false -p ros_command_control:=false -p steps:=1

# 4. 上机（架空、急停在手）
ros2 launch dm_dog_controller dm_dog_policy.launch.py
```

---

## 八、文件位置速查

| 项 | 路径 |
| --- | --- |
| 工程根 | `<repo>/dm_dog_controller` |
| **PD 增益** | `bs_hardware/config.py` |
| **速度/死区** | `launch/dm_dog_policy.launch.py` |
| 电机总线 | `bs_hardware/motor_bus.py` |
| 启动序列 | `bs_hardware/startup.py` |
| 策略推理/动作管线 | `dm_dog_controller/deployment.py` |
| 策略文件 | `policies/*.pt` |
| ROS 工作区 | `<ros2_ws>` |

**备份命名规律**：`*.bak-before-<改动名>`
