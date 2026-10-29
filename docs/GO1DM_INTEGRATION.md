# go1dm 策略集成到 dm_dog_controller

**适用**：把 `go2_rl_gym` 里用 **go1dm**（真狗尺寸 Go1 URDF）训练出来的策略
加载到 `dm_dog_controller` 上跑实机。

**为什么要单独写这个**：go1dm 的**关节顺序**和 dm 电机顺序**不一致**，
直接加载会把腿命令送错腿（和 2026-10-15 那次"上电发癫"类似的后果）。
另外**默认关节角**也不同，观测会出现常值偏置。

---

## 一、两处必须同时改的配置

**文件**：`bs_hardware/config.py`

### 1. 关节顺序置换

| | go1dm（策略输出） | dm（电机顺序） |
| --- | --- | --- |
| 索引 0-2 | **LF**（左前） | **FL**（左前） |
| 索引 3-5 | **LH**（左后） | **FR**（右前） |
| 索引 6-8 | **RF**（右前） | **RL**（左后） |
| 索引 9-11 | **RH**（右后） | **RR**（右后） |

```python
# 默认（go2 策略）：与 dm 顺序一致
MODEL_JOINT_PERMUTATION = list(range(ACTION_DIM))

# ★ 加载 go1dm 策略时改成：
MODEL_JOINT_PERMUTATION = [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]
```

**语义**：`dm_action[i] = policy_action[PERM[i]]`

这个置换**恰好是对合**（正置换 = 逆置换），所以观测侧用同一个数组即可
（`deployment.py` 里已显式算逆置换，一般情形也支持）。

### 2. 默认关节角

```python
# 默认（go2.urdf）：
DEFAULT_DOF_POS = [
     0.1, 0.8, -1.5,   # FL
    -0.1, 0.8, -1.5,   # FR
     0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5,   # RR
]

# ★ 加载 go1dm 策略时改成（按 dm 的 FL/FR/RL/RR 顺序重排）：
DEFAULT_DOF_POS = [
     0.119837, 0.994838, -1.780236,   # FL (← go1dm LF)
    -0.088174, 0.994838, -1.780236,   # FR (← go1dm RF)
     0.090845, 0.994838, -1.780236,   # RL (← go1dm LH)
    -0.090082, 0.994838, -1.780236,   # RR (← go1dm RH)
]
```

**为什么**：训练时观测是 `dof_pos - default_joint_angles`，动作目标是
`default + residual × action_scale`。**default 不匹配会有常值偏置。**

---

## 二、操作步骤

```bash
D=<repo>/dm_dog_controller
cd $D

# ① 备份
cp -a bs_hardware/config.py bs_hardware/config.py.bak-before-go1dm
cp -a launch/dm_dog_policy.launch.py launch/dm_dog_policy.launch.py.bak-before-go1dm

# ② 装入策略
cp -a <导出的 policy.pt> policies/go1dm_policy.pt

# ③ 改 launch 指向
sed -i 's|"go2_ppo_65000_planA.pt"|"go1dm_policy.pt"|' launch/dm_dog_policy.launch.py

# ④ 改 config.py 的两处（手动编辑，见上面第一节）

# ⑤ 构建
source /opt/ros/humble/setup.bash && cd <ros2_ws>
colcon build --packages-select dm_dog_controller --symlink-install

# ⑥ 冒烟测试（不接硬件）
source install/setup.bash
S=$(ros2 pkg prefix dm_dog_controller)/share/dm_dog_controller
ros2 run dm_dog_controller dm_dog_policy_node --ros-args \
  -p model:=$S/policies/go1dm_policy.pt -p use_imu:=false -p use_can:=false \
  -p startup:=false -p send_actions:=false -p ros_command_control:=false -p steps:=1
```

**预期输出**：
```
loaded translation_policy: .../go1dm_policy.pt entry=forward input_dim=45 ...
step=0 rl=False ... cmd=(0.000, 0.000, 0.000) gravity=(0.000, 0.000, -1.000)
```

---

## 三、上电前的必查项

### ⚠️ 1. 关节顺序实测验证（最重要）

**不要凭代码正确就上电。** 用下面的方法实测：

1. **架空、急停在手**
2. 启动后进入站立姿态，**用手轻推某条腿**，看 **dm 节点日志里对应位置的
   `policy_action`** 是否变化 —— 应该只有那条腿的分量变
3. 或者：**只给一条腿的命令**（用键盘/手柄的横移或转向），确认动的是预期的腿

**如果动错腿 → 立刻急停，检查 `MODEL_JOINT_PERMUTATION`。**

### ⚠️ 2. 扭矩与温度

go1dm 训练时 URDF 限幅改成了 **真机峰值 23.5 / 23.5 / 40 N·m**，
实测策略的 **RMS 扭矩约 12.9 N·m**（占限幅 55%）。

而真机电机：
- 髋/大腿（550g）：持续 **6 N·m** / 峰值 **23.5 N·m**
- 小腿（730g）：持续 **8 N·m** / 峰值 **40 N·m**

**12.9 N·m 的 RMS 已超过持续额定（6~8 N·m）** —— **跑几分钟必须摸电机温度**，
烫手立刻停。

### ⚠️ 3. 站立高度

go1dm 训练用的站高是 **0.2906 m**（URDF 正运动学算出）。
如果实机启动姿态高度明显不同，`correct_base_height` 对应的观测会偏。

---

## 四、回退

```bash
cd <repo>/dm_dog_controller
cp -a bs_hardware/config.py.bak-before-go1dm bs_hardware/config.py
cp -a launch/dm_dog_policy.launch.py.bak-before-go1dm launch/dm_dog_policy.launch.py
source /opt/ros/humble/setup.bash && cd <ros2_ws>
colcon build --packages-select dm_dog_controller --symlink-install
```

---

## 五、go1dm 训练出的策略指标（参考）

| 指标 | go1dm | Go2 65000（对照） |
| --- | --- | --- |
| 速度跟踪 | **0.72~0.73** | 0.66 |
| 身体晃动 `ang_vel_xy` | **-0.058**（更小） | -0.109 |
| 摔倒率 | **~7.9%** | ~12% |
| 内八 `hip_to_default` | **-0.061**（更少） | -0.140 |
| episode 长度 | 1180 | 1220 |
| 奖励 | ~12.2 | 12.8 |
| RMS 扭矩 | 12.9 N·m（55% 限幅） | 5.9 N·m（25%） |

**go1dm 在"晃动"和"摔倒率"上更好，但用扭矩更足**（真机要注意发热）。
