# four-legged-dog-rl

自研四足机器人（12 个达妙 DM 电机）的强化学习训练与 ROS 2 实机部署工程。
训练部分基于 [go2_rl_gym](https://github.com/wty-yy/go2_rl_gym)（Isaac Gym + PPO），
部署部分是一套独立的 ROS 2 控制器，直接驱动达妙电机。

工程包含两条训练线：

- **go2**：在 Unitree Go2 的 URDF 上训练，作为基线（**当前部署运行的是这条线**）。
- **go1dm**：在**按真狗实测尺寸改造的 Go1 URDF** 上训练，几何/质量/站立高度都贴合实机
  （**指标更好，但尚未做真机验证**）。

---

## 效果与验证状态

**目前没有策略行走的视频。** 下面这张是模型几何校验的实测输出，不是策略效果：

| 文件 | 内容 |
| --- | --- |
| [模型几何诊断](docs/assets/cad_urdf_diagnosis.png) | 零位下四脚位置、象限判定、高度分布 |
| [仿真对比](docs/assets/cad_urdf_sim_compare.png) | 自由落体 4 s 前后的姿态与质心轨迹 |

**两条训练线的验证状态：**

| | go2 基线 | go1dm |
| --- | --- | --- |
| 仿真训练 | ✅ 完成 65000 迭代 | ✅ 完成 30000 迭代 |
| Isaac Gym 回放 | ✅ | ✅ |
| **真机行走** | ⚠️ 能走，但停下来有多余动作 | ❌ **未验证** |
| MuJoCo sim2sim | ⚠️ 用的是 Go2 模型 | ❌ 未做 |

---

## 仓库结构

```
four-legged-dog-rl/
├── go2_rl_gym/                     # 训练框架（基于 wty-yy/go2_rl_gym）
│   ├── legged_gym/
│   │   ├── envs/go2/               # Go2 任务定义
│   │   ├── envs/go1dm/             # ★ 真狗尺寸 Go1 任务（本项目新增）
│   │   └── scripts/{train,play}.py # 训练 / 回放入口
│   ├── resources/robots/
│   │   ├── go2/                    # Go2 URDF + mesh
│   │   └── go1dm/                  # ★ 真狗尺寸 URDF（本项目新增）
│   ├── rsl_rl/                     # PPO 实现
│   └── deploy/                     # MuJoCo sim2sim
├── dm_dog_controller/              # ROS 2 部署控制器
│   ├── bs_hardware/                # 硬件层：CAN / 电机 / IMU / 启动序列
│   │   └── config.py               # ★ PD 增益、默认关节角、关节置换都在这里
│   ├── dm_dog_controller/          # 策略推理与动作管线
│   │   ├── deployment.py           # 观测构造 + 动作映射
│   │   └── cmd_vel_shaping.py      # 摇杆 → 速度指令
│   ├── launch/dm_dog_policy.launch.py
│   ├── policies/                   # 导出的 TorchScript 策略
│   └── third_party_libs/           # 达妙 SDK 需要的新版 libstdc++
└── docs/                           # 中文文档（踩坑记录、调参、控制、集成）
```

---

## 安装

### 训练环境（Isaac Gym）

需要 NVIDIA GPU + CUDA 11.7。**Isaac Gym 只能从 NVIDIA 官网下载**（本仓库不含）：

<https://developer.nvidia.com/isaac-gym>

```bash
# 1. Python 3.8 环境（Isaac Gym Preview 4 只支持到 3.8）
conda create -n isaacgym python=3.8
conda activate isaacgym

# 2. Isaac Gym
cd ~/isaacgym/python && pip install -e .
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

# 3. 本仓库
cd <repo>/go2_rl_gym
pip install -e .
```

CPU 环境（只能跑 MuJoCo sim2sim，不能训练）：

```bash
python3 -m venv .venv
.venv/bin/pip install -r go2_rl_gym/requirements-cpu.txt
```

### 部署环境（ROS 2 Humble）

```bash
sudo apt install ros-humble-joy ros-humble-sensor-msgs ros-humble-std-srvs

mkdir -p ~/ros2_ws/src && cd ~/ros2_ws/src
ln -s <repo>/dm_dog_controller .
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select dm_dog_controller --symlink-install
source install/setup.bash
```

内核需要 `can` 支持，用户需在 `dialout` 组。

---

## 资产准备

### 训练用的 URDF

| 机器人 | 路径 | 说明 |
| --- | --- | --- |
| Go2 | `go2_rl_gym/resources/robots/go2/urdf/go2.urdf` | 来自上游，含 dae mesh |
| **go1dm** | `go2_rl_gym/resources/robots/go1dm/go1dm.urdf` | 本项目新增，见下 |

`go1dm.urdf` 是把 Go1 模型按**真狗实测尺寸**改造后的版本：

| 参数 | 值 |
| --- | --- |
| 大腿长 / 小腿长 | 0.25 m / 0.23 m |
| 总质量 | 11.51 kg |
| 站立高度（base 原点到底球心） | 0.2906 m |
| 足端前后间距 | 0.4400 m（与实测报告一致） |
| 关节限幅 | 23.5 / 23.5 / 40 N·m（真机峰值） |

改造过程中做了三件事（详见 [docs/GO1DM_INTEGRATION.md](docs/GO1DM_INTEGRATION.md)）：

1. `visual` 的 mesh 引用（指向已不可用的外部路径）换成与 `collision` 相同的 primitive
2. 删除 Isaac Gym 不需要的 `<gazebo>` / `<transmission>` 块
3. 关节限幅从继承来的保守值 10/10/18 改成真机峰值 23.5/23.5/40 N·m

### 部署用的硬件

| 项 | 需要 |
| --- | --- |
| CAN 适配器 | **2 个** 达妙 USB2FDCAN（各驱动两条腿），序列号写在 `bs_hardware/config.py` 的 `CAN_SERIALS` |
| IMU | DM-IMU-L1（串口 921600） |
| 手柄 | shanwan "Android GamePad"（D-input 模式） |

---

## 训练

以下命令在 `go2_rl_gym/` 目录、激活 Isaac Gym 环境后运行。

### go1dm（真狗尺寸模型）

```bash
python legged_gym/scripts/train.py \
  --task=go1dm --headless --num_envs=4096 --max_iterations=30000
```

输出：`logs/go1dm_ppo/<时间戳>/model_<iter>.pt` + `config.yaml`。

**续训**（注意迭代计数器与课程会重置，见 [踩坑记录](docs/ALL_TRAINING_AND_PITFALLS.md) §6 坑 6）：

```bash
python legged_gym/scripts/train.py \
  --task=go1dm --headless --resume \
  --load_run=<run目录名> --checkpoint=<N> \
  --num_envs=4096 --max_iterations=<剩余迭代>
```

### go2 基线

```bash
python legged_gym/scripts/train.py \
  --task=go2 --headless --num_envs=4096 --max_iterations=65000
```

### 关键配置差异

`go1dm` 只覆盖机器人相关项，奖励权重/地形/课程全部继承 `go2`，保证是**单变量对比**：

| 项 | go2 | go1dm |
| --- | --- | --- |
| `asset.file` | `go2.urdf` | `go1dm.urdf` |
| `asset.foot_name` | `"foot"` | **`"FOOT"`**（body 名是 `LF_FOOT`，匹配大小写敏感） |
| `control.stiffness` | `{'joint': 20.0}` | **`{'HAA':40,'HFE':40,'KFE':40}`** |
| `control.damping` | `{'joint': 0.5}` | `{'HAA':1.0,'HFE':1.0,'KFE':1.0}` |
| `rewards.base_height_target` | 0.38 | **0.2906**（正运动学算出的 base 高度） |

> 增益 key 是**子串匹配**关节名（`'joint' in 'FL_hip_joint'`）。
> 新机器人关节名是 `LF_HAA` 这类，写 `'joint'` 匹配不到 → **全部增益为 0，机器人直接瘫倒**。

---

## 评估

```bash
python legged_gym/scripts/play.py --task=go1dm --headless --num_envs=16 \
  --load_run=<run目录名> --checkpoint=<N>
```

导出 TorchScript（`play.py` 会自动导出到 `logs/<任务>/exported/policies/`）：

```
policy.pt    # TorchScript，部署用
policy.onnx
policy.pkl
```

> `play.py` 中 `FIX_COMMAND = True` 会把指令固定为前进 1.0 m/s。
> 对比不同训练范围的 checkpoint 时，需要先改这个值。

---

## 部署

### 启动

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
ros2 launch dm_dog_controller dm_dog_policy.launch.py
```

`launch` 里 `startup: False` —— **启动时不会自动站起来**，需要手柄或服务触发：

```bash
ros2 service call /dog_posture std_srvs/srv/SetBool "{data: true}"    # 站
ros2 service call /dog_posture std_srvs/srv/SetBool "{data: false}"   # 趴
```

### 手柄操作

| 输入 | 动作 |
| --- | --- |
| 左摇杆 上下（axis 1） | 前进 / 后退 |
| 左摇杆 左右（axis 0） | 横移 |
| 右摇杆 左右（axis 2） | 原地转向 |
| **LB + A**（不按 RB） | 站立 / 趴下切换 |
| **A** | 停车保持 |
| **LB + RB + A**（三个同时） | **急停** |

限速、死区、`action_clip` 都在 `launch/dm_dog_policy.launch.py` 里；PD 增益和关节配置在
`bs_hardware/config.py` 里。完整说明见 [docs/CONTROL_GUIDE.md](docs/CONTROL_GUIDE.md)
和 [docs/DOG_TUNING.md](docs/DOG_TUNING.md)。

### 切换策略

改 `launch/dm_dog_policy.launch.py` 里的策略文件名即可。切换到 `go1dm_*` 时**必须同时**
修改 `bs_hardware/config.py` 的两处（否则腿命令会送错腿）：

```python
MODEL_JOINT_PERMUTATION = [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]
DEFAULT_DOF_POS = [
     0.119837, 0.994838, -1.780236,   # FL
    -0.088174, 0.994838, -1.780236,   # FR
     0.090845, 0.994838, -1.780236,   # RL
    -0.090082, 0.994838, -1.780236,   # RR
]
```

原因：go1dm 的关节顺序是 `LF, LH, RF, RH`，而 dm 控制器期望 `FL, FR, RL, RR`。
详见 [docs/GO1DM_INTEGRATION.md](docs/GO1DM_INTEGRATION.md)。

---

## 策略文件

`dm_dog_controller/policies/`：

| 文件 | 来源 | 训练范围 | 备注 |
| --- | --- | --- | --- |
| `go2_ppo_65000_planA.pt` | 本项目自训（Go2 URDF） | ±2.0 m/s | **当前部署使用** |
| `go2_ppo_50000_hipfix.pt` | 本项目自训 | ±1.0 m/s | |
| `go2_ppo_35000.pt` | 本项目自训 | ±1.0 m/s | |
| `go2_ppo_20000.pt` | 本项目自训 | ±0.5 m/s | |
| `go2_ppo_7000.pt` | 本项目自训 | ±0.5 m/s | |
| `go2_ppo_5000.pt` | 本项目自训 | ±0.5 m/s | |
| `go2_ppo_1000.pt` | 本项目自训 | ±0.5 m/s | |
| `go2_ppo_500.pt` | 本项目自训 | ±0.5 m/s | 最早的 checkpoint |
| `active.pt` | 本项目自训 | — | 与 `go2_ppo_7000.pt` 内容相同 |
| **`go1dm_policy_iter23500.pt`** | 本项目自训（go1dm） | ±1.0 m/s | **未做真机验证** |
| **`go1dm_policy_iter19500.pt`** | 本项目自训（go1dm） | ±0.5 m/s | 课程扩张前，低速范围最优 |
| `policy.pt` | 早期 legacy 策略（225 维输入） | — | 与上面不是同一套观测格式 |

**未包含**：四个来自其他开源项目的预训练策略（`go2_cts_150k`、`go2_him_21k`、
`go2_dwaq_119.5k`、`go2dog_policy_1.15M`）—— 分发许可不明确，因此不随本仓库发布。

---

## 结果

### go1dm vs go2（同一套指标口径）

| 指标 | go1dm | go2 (65000) |
| --- | --- | --- |
| 速度跟踪 `tracking_lin_vel` | **0.74** | 0.659 |
| 摔倒率 | **6.2 %** | ~12 % |
| 身体晃动 `ang_vel_xy` | **−0.058** | −0.109 |
| 内八 `hip_to_default` | **−0.061** | −0.140 |
| episode 长度 | 1196 | 1220 |
| 奖励 | 12.2 | 12.8 |
| **RMS 扭矩** | **12.9 N·m**（限幅 23.5，占用 55 %） | 5.9 N·m（占用 25 %） |

训练设备：单卡 RTX 4090（共享容器），go1dm 30000 迭代约 8 小时，约 1.0 s/迭代。

**go1dm 用扭矩更足**，而真机电机持续额定只有 6~8 N·m —— 实机运行时需要留意电机温度。

完整的分段曲线和诊断过程见 [docs/ALL_TRAINING_AND_PITFALLS.md](docs/ALL_TRAINING_AND_PITFALLS.md)。

---

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/ALL_TRAINING_AND_PITFALLS.md](docs/ALL_TRAINING_AND_PITFALLS.md) | 全部训练记录 + 踩坑清单 + 复现指引 |
| [docs/DOG_TUNING.md](docs/DOG_TUNING.md) | 调参手册：部署层 vs 策略层、速度/PD/姿态的影响 |
| [docs/PARAM_TUNING.md](docs/PARAM_TUNING.md) | 参数修改步骤与峰值为矩计算 |
| [docs/CONTROL_GUIDE.md](docs/CONTROL_GUIDE.md) | 控制指令、站立触发、急停、排查 |
| [docs/GO1DM_INTEGRATION.md](docs/GO1DM_INTEGRATION.md) | 切换 go1dm 策略的关节置换与默认角 |
| [docs/GO1DM_RETROSPECTIVE.md](docs/GO1DM_RETROSPECTIVE.md) | go1dm 训练复盘 |
| [docs/TRAINING_RECORD.md](docs/TRAINING_RECORD.md) | go2 首次训练的详细曲线 |

---

## 来源与许可

### 直接使用的框架

| 项目 | 用途 | 许可 |
| --- | --- | --- |
| [wty-yy/go2_rl_gym](https://github.com/wty-yy/go2_rl_gym) | 本仓库训练部分的基础 | MIT |
| [unitreerobotics/unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) | go2_rl_gym 的上游训练框架 | BSD-3-Clause |
| NVIDIA Isaac Gym | 物理仿真后端（**不随本仓库分发**） | NVIDIA 许可 |

`go2_rl_gym/` 目录下的代码保留原 MIT 许可，见 [go2_rl_gym/LICENSE](go2_rl_gym/LICENSE)。

### 机器人模型

| 模型 | 来源 | 说明 |
| --- | --- | --- |
| `resources/robots/go2/` | Unitree 官方 URDF，经 go2_rl_gym 分发 | |
| `resources/robots/go1dm/go1dm.urdf` | 由 Go1 模型按本机器人**实测尺寸**改造 | 改造内容见上文；mesh 已替换为 primitive |

### 本项目的改动

- `go2_rl_gym/legged_gym/envs/go1dm/` —— go1dm 任务配置与注册（新增）
- `go2_rl_gym/resources/robots/go1dm/` —— 训练版 URDF（新增）
- `dm_dog_controller/` —— ROS 2 部署控制器（本项目自有代码）
  - 新增 `MODEL_JOINT_PERMUTATION`（关节顺序置换）
  - `cmd_vel_shaping.py` 支持前进/后退分别限幅
- `docs/` —— 全部中文文档

### 许可

本仓库未附统一 LICENSE 文件。`go2_rl_gym/` 保留其原有 MIT 许可；
`dm_dog_controller/` 为项目自有代码，未声明开源许可。
如需再分发，请先确认 `dm_dog_controller/` 的授权范围。
