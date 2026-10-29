# 本机环境配置记录（CPU / 无 NVIDIA GPU）

本文件记录在此开发机上为 `go2_rl_gym` 配置的可用环境，
以及因显卡限制而**未安装**的部分。

## 1. 硬件与系统探测结论

| 项目 | 探测结果 | 影响 |
| --- | --- | --- |
| GPU | `00:02.0 Intel Corporation Device 7d51`（核显） | 无 NVIDIA GPU |
| `nvidia-smi` / `/dev/nvidia*` / `nvcc` | 均不存在 | 无驱动、无 CUDA |
| 操作系统 | Ubuntu 22.04.5 LTS x86_64 | — |
| 系统 Python | 3.10.12 | 与 isaacgym 要求的 3.8 不符 |
| Conda / MiniConda | 未安装 | 改用 `venv` |
| X 显示 | `DISPLAY=:1` 可达 | Mujoco 可视化可运行 |
| 磁盘 | `/` 剩余约 258 GB | 充足 |

**结论：本机无法安装/运行 Isaac Gym**（该框架强制要求 NVIDIA GPU + CUDA，且不发布
PyPI 包），因此训练与 `play.py` 在本机不可用；而 **Mujoco Sim2Sim 部署完全可用**。

## 2. 环境位置

虚拟环境创建在工程根目录下，已被 `.gitignore` 忽略（不会污染 git 仓库）：

```text
<repo>/go2_rl_gym/.venv
```

> 选择工程内 `.venv` 的原因：`.gitignore` 第 133-135 行已包含 `.venv` / `venv/`，
> 无需改动仓库配置。

激活方式：

```bash
source <repo>/go2_rl_gym/.venv/bin/activate
# 或不激活，直接用 .venv/bin/python 调用
```

> 系统缺少 `python3-venv`/`ensurepip`，且 `sudo` 需要密码，故采用
> `python3 -m venv --without-pip` + 官方 `get-pip.py` 引导 pip 的方式创建。

## 3. 已安装内容

| 组件 | 版本 | 用途 |
| --- | --- | --- |
| Python | 3.10.12 | 虚拟环境解释器 |
| torch | 2.3.1+cpu | 策略推理 / rsl_rl |
| numpy | 1.26.4 | 数值计算 |
| mujoco | 3.2.3 | Sim2Sim 物理仿真 |
| pygame | 2.6.1 | 手柄遥操作 |
| imageio / imageio-ffmpeg | 2.37.0 / 0.6.0 | 录制视频 |
| matplotlib | 3.9.4 | MoE 权重可视化 |
| PyYAML | 6.0.2 | 读取配置文件 |
| onnx | 1.17.0 | 模型导出 |
| tensorboard | 2.14.0 | 训练日志（为将来 GPU 机准备） |
| rsl_rl | 1.0.2（editable，`rsl_rl/`） | 强化学习算法库 |
| go2_rl_gym | 1.0.3（editable，仓库根） | 本工程 |

依赖清单见 [`requirements-cpu.txt`](../requirements-cpu.txt)。

两个本地包均以 `pip install -e . --no-deps` 安装，以跳过
`setup.py` 中钉死的 `isaacgym`（无法安装）和 `numpy==1.20`（无 Python 3.10 wheel）。

## 4. 验证结果

### 4.1 依赖导入

```
OK   torch            2.3.1+cpu      (torch.cuda.is_available() == False)
OK   numpy            1.26.4
OK   mujoco           3.2.3
OK   mujoco.viewer
OK   pygame           2.6.1
OK   imageio          2.37.0
OK   matplotlib       3.9.4
OK   yaml             6.0.2
OK   onnx             1.17.0
OK   tensorboard      2.14.0
OK   rsl_rl
OK   legged_gym
```

### 4.2 Sim2Sim 策略推理（无头）

加载 `deploy/pre_train/go2/go2_moe_cts_high_slope_thre_164k_0.6715.pt`，
在 `cross_stairs.xml` 地形上跑 5000 步物理仿真，策略正常输出动作、无报错。

### 4.3 可视化部署端到端

```bash
cd <repo>/go2_rl_gym
.venv/bin/python deploy/deploy_mujoco/deploy_go2.py
```

实测已成功启动 Mujoco viewer 并进入仿真循环，输出：

```
No Joystick detected. Using default commands from config.
```

未接手柄时使用配置中的默认前进指令 `cmd_init: [1.0, 0.0, 0.0]`。

## 5. 常用命令

```bash
cd <repo>/go2_rl_gym

# Sim2Sim 可视化（默认模型 + cross_stairs 地形）
.venv/bin/python deploy/deploy_mujoco/deploy_go2.py

# 录制视频 / 可视化 MoE 权重 / 保存 latent
.venv/bin/python deploy/deploy_mujoco/deploy_go2.py --save-video
.venv/bin/python deploy/deploy_mujoco/deploy_go2.py --visualize-moe-weights
.venv/bin/python deploy/deploy_mujoco/deploy_go2.py --save-moe-latent
```

切换模型与地形：编辑 `deploy/deploy_mujoco/configs/go2.yaml` 中的
`policy_path` 与 `xml_path`。可选地形：`flat.xml`、`stairs.xml`、
`race_track.xml`、`cross_slope.xml`、`cross_stairs.xml`。

## 6. 未安装部分及原因

| 组件 | 状态 | 原因 |
| --- | --- | --- |
| **isaacgym** | 未安装 | 仅 NVIDIA 官网提供，强制要求 NVIDIA GPU + CUDA；本机无独显 |
| **PyTorch CUDA 版** | 未安装 | 无 CUDA 运行时，已装 `+cpu` 版本 |
| **训练 / play.py** | 不可用 | 依赖 `import isaacgym`；实测报 `ModuleNotFoundError: No module named 'isaacgym'` |
| **numpy==1.20** | 未安装 | 该版本无 Python 3.10 wheel，改用 1.26.4 |
| **RoboGauge** | 未安装 | 需单独 clone 并启动独立 server，属可选评估工具 |
| **unitree_sdk2py** | 未安装 | 真机 Python 部署依赖，目标是机器人端 Jetson，不在本 x86 开发机构建 |

## 7. 将来在带 NVIDIA GPU 的机器上补训练环境

训练环境需要 Python 3.8 + isaacgym，建议用 Conda 单独建环境，与本 `.venv` 隔离：

```bash
conda create -n unitree-rl python=3.8
conda activate unitree-rl
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
# 从 NVIDIA 官网下载 Isaac Gym，解压后：
cd isaacgym/python && pip install -e .
# 回到本仓库
cd <repo>/go2_rl_gym
pip install -e ./rsl_rl
pip install -e .          # 此时 isaacgym 已可满足
```

若训练机 Python 为 3.8，`numpy==1.20` 可正常安装，无需改为 1.26.4。
