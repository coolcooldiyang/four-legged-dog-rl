# dm_dog_controller 控制指令手册（go1dm 策略版）

**当前部署**：`policies/go1dm_policy_iter23500.pt`（真狗实测尺寸 Go1 训练，迭代 23500）
**最后更新**：2026-10-16

---

## 零、上电前的三条硬规矩

1. **狗必须架空**（四轮离地或用支架垫起），急停在手
2. **第一次跑新策略，先只做「站立」不做「走路」**，观察 30 秒有没有异常
3. **跑几分钟就摸电机温度** —— 新策略 RMS 扭矩 12.9 N·m，
   而真机电机**持续额定只有 6~8 N·m**，烫手立刻停

---

## 一、启动

### 1. 开终端，加载环境

```bash
source /opt/ros/humble/setup.bash
source <ros2_ws>/install/setup.bash
```

### 2. 确认硬件在位

```bash
ls /dev/input/js0                                    # 手柄
ls /dev/serial/by-id/ | grep -i imu                  # IMU
ip link show can0 2>/dev/null || ls /dev/ttyACM*     # 电机 CAN
```

### 3. 启动

```bash
ros2 launch dm_dog_controller dm_dog_policy.launch.py
```

**注意**：launch 里 `startup: False` —— **启动时狗不会自己站起来**，
要用手柄或服务命令触发（见第二节）。

---

## 二、让狗站起来 / 趴下

### 方式 A：手柄（推荐）

**按住 `LB` + 按 `A`（不要按 `RB`）** —— 站立/趴下切换。

### 方式 B：命令行服务

```bash
# 站起来
ros2 service call /dog_posture std_srvs/srv/SetBool "{data: true}"

# 趴下
ros2 service call /dog_posture std_srvs/srv/SetBool "{data: false}"
```

**站起来的瞬间要有心理准备** —— 如果策略有问题，狗会立刻有异常动作，随时急停。

---

## 三、控制方式

### 手柄布局（shanwan "Android GamePad"，D-input 模式）

```
        左摇杆 (axis 0/1)              右摇杆 (axis 2/3)
        前后 = axis 1                  左右 = axis 2
        左右 = axis 0                  （上下 axis 3 未使用）

        LB = button 6                  RB = button 7
        A  = button 0                  （button 10 = 退出）
```

### 操作表

| 摇杆 | 作用 | 上限 |
| --- | --- | --- |
| **左摇杆 上下**（axis 1） | **前进 / 后退** | ±0.4 m/s |
| **左摇杆 左右**（axis 0） | **左右横移** | ±0.4 m/s |
| **右摇杆 左右**（axis 2） | **原地转向** | ±1.5 rad/s |

**规则**：
- 摇杆幅度 **< 0.20（死区）→ 输出 0**（防止手柄漂移让狗自己走）
- 死区之外**线性映射**：推一半 = 一半速度
- **前进时横移被屏蔽**（`forward_blocks_lateral: True`）
- 最小速度下限是 **0**（不会因为指令小就被抬到某个速度）

### 按钮

| 按钮 | 作用 | 说明 |
| --- | --- | --- |
| **LB + A**（不按 RB） | **站立 / 趴下 切换** | 通过 `dog_posture` 服务 |
| **A**（单独按） | **停车保持** | 零速度，保持站立姿态；松开恢复 |
| **LB + RB + A**（三个同时） | **🚨 急停** | 立即停 |
| **button 10** | 退出程序 | |

⚠️ **急停必须三个键同时按**（`emergency_lb_button=6`, `emergency_rb_button=7`,
`emergency_a_button=0`）。**别记错。**

---

## 四、当前生效的参数（想改就改这些）

**文件**：`launch/dm_dog_policy.launch.py`

| 参数 | 当前值 | 作用 |
| --- | --- | --- |
| `continuous_forward_max` | **0.4** | 前进上限 m/s |
| `continuous_backward_max` | **0.4** | 后退上限 m/s |
| `continuous_lateral_max` | 0.4 | 横移上限 m/s |
| `continuous_yaw_max` | 1.5 | 转向上限 rad/s |
| `forward_speed_mode` | `continuous` | 速度随摇杆连续变化 |
| `fixed_forward_speed` | 0.5 | （仅在 fixed 模式生效） |
| `deadzone` | 0.20 | 手柄死区 |
| `cmd_input_deadband` | 0.15 | 控制器侧再挡一次小指令 |
| `action_clip` | **2.0** | 策略动作裁剪 |

**策略侧参数**：`bs_hardware/config.py`
| 参数 | 当前值 |
| --- | --- |
| `RL_ACTION_SCALE` | **0.125** |
| `MODEL_JOINT_PERMUTATION` | `[0,1,2,6,7,8,3,4,5,9,10,11]`（go1dm 专用） |
| `DEFAULT_DOF_POS` | go1dm 实测站立角 |
| `TRANSLATION_POLICY_RL_KP` | **25.0** |
| `TRANSLATION_POLICY_RL_KD` | **2.0** |
| `TRANSLATION_POLICY_BM_CALF_KP` | **25.0** |
| `TRANSLATION_POLICY_BM_CALF_KD` | 2.0 |

### 改完必须重建

```bash
cd <ros2_ws>
source /opt/ros/humble/setup.bash
colcon build --packages-select dm_dog_controller --symlink-install
```

> `--symlink-install` 让 **Python 代码和 launch 文件改动免重建也能生效**，
> 但**新建文件、改动 `setup.py`、改动 C++ 必须重建**。

---

## 五、切回旧策略（go2 planA）

```bash
cd <repo>/dm_dog_controller
cp -a bs_hardware/config.py.bak-before-go1dm-deploy bs_hardware/config.py
cp -a launch/dm_dog_policy.launch.py.bak-before-go1dm-deploy launch/dm_dog_policy.launch.py
cd <ros2_ws> && source /opt/ros/humble/setup.bash
colcon build --packages-select dm_dog_controller --symlink-install
```

**切到别的 checkpoint**（比如扩张前的 19500）：
```bash
sed -i 's/go1dm_policy_iter23500.pt/go1dm_policy_iter19500.pt/' launch/dm_dog_policy.launch.py
```
（需要先把对应 checkpoint 导出并放到 `policies/`）

---

## 六、调试观察

### 看控制指令链路

```bash
# 原始手柄 → 整形后的指令
ros2 topic echo /joy_cmd       # 手柄输出
ros2 topic echo /cmd_vel       # 送给策略的最终指令
```

**关键验证**：把左摇杆**往前进方向推到底**，看 `/cmd_vel` 的 `linear.x`：
- 是 **正的** → ✅ 正确
- 是 **负的** → 前后反了，把 `continuous_forward_max` 和
  `continuous_backward_max` 对调，或改 `forward_sign`

### 看策略输出

控制器日志里会打印：
```
step=N rl=True active_policy=translation_policy cmd=(x, y, z) gravity=(...) policy_action[0:3]=[...] raw_action_units[0:3]=[...]
```

**关节顺序验证**（**上电前一定要做**）：
架空进入站立后，**用手只推一条腿**，看日志里 `policy_action`
是否**只有对应位置**变化：
- 推 **左前腿** → 索引 **0-2** 变化
- 推 **左后腿** → 索引 **3-5** 变化
- 推 **右前腿** → 索引 **6-8** 变化
- 推 **右后腿** → 索引 **9-11** 变化

**如果动错腿 → 立刻急停，检查 `MODEL_JOINT_PERMUTATION`。**

### 停车 / 恢复

```bash
# 软件停车（等价于按 A）
ros2 topic pub --once /dog_hold std_msgs/msg/Bool "{data: true}"
ros2 topic pub --once /dog_hold std_msgs/msg/Bool "{data: false}"
```

---

## 七、出问题时的排查顺序

| 现象 | 先查 |
| --- | --- |
| 狗不动、日志 `policy inference skipped: lying` | 没做站立序列 → 按 LB+A 或调服务 |
| 狗不动、日志 `rl_inactive` | 看是不是按了 A（停车保持） |
| 狗动错腿 / 抽搐 | **急停**，检查 `MODEL_JOINT_PERMUTATION` |
| 狗站起就前扑/后仰 | 检查 `DEFAULT_DOF_POS` 是否是 go1dm 那组 |
| 电机发烫 | 正常现象（策略用扭矩足），缩短测试时间 |
| 狗自己慢慢走 | 手柄漂移 → 加大 `deadzone`（当前 0.20） |

---

## 八、这台机器上这个工程的完整文件位置

| 项 | 路径 |
| --- | --- |
| 工程 | `<repo>/dm_dog_controller` |
| ROS 工作区 | `<ros2_ws>` |
| 策略 | `dm_dog_controller/policies/go1dm_policy_iter23500.pt` |
| 训练代码 | `<repo>/go2_rl_gym` |
| 训练记录 | `training_backup/TRAINING_RECORD.md` |
| 集成说明 | `training_backup/GO1DM_INTEGRATION.md` |
| 本手册 | `training_backup/CONTROL_GUIDE.md` |
