# 云端环境需求（训练 / 评估 / 部署）

本文件说明跑通 `go2_rl_gym` 各条链路分别需要什么环境。
按「是否需要 NVIDIA GPU」分成三类，可以按需只开对应机器。

---

## 0. 三条链路一览

| 链路 | 命令入口 | 需要 GPU？ | 能否在本地这台机器跑 |
| --- | --- | --- | --- |
| **A. 训练 / Play** | `legged_gym/scripts/train.py`、`play.py` | ✅ 必须 NVIDIA GPU + Isaac Gym | ❌ 已被显卡限制 |
| **B. RoboGauge 评估** | `train.py --robogauge` + 独立 server | ⚠️ 随训练附带；评估本身跑 CPU/Mujoco | ❌（依赖 A） |
| **C. Mujoco Sim2Sim 部署** | `deploy/deploy_mujoco/deploy_go2.py` | ❌ 纯 CPU 即可 | ✅ **已在本地 `.venv` 配好并验证** |
| **D. 真机部署** | `deploy/deploy_real/deploy_real_go2.py` | ❌ 需要 Jetson + 真实机器人 | ❌（无硬件） |

**结论**：只有链路 A/B 需要上云 GPU；链路 C 本地已经能跑，不需要云。

---

## 1. 链路 A/B 的云端环境规格（核心）

### 1.1 关键约束（顺序很重要）

1. **Python 必须是 3.8** —— 这是最容易踩的坑。
   Isaac Gym 官方 wheel 只提供 Python 3.7 / 3.8 的二进制绑定
   （`isaacgym/_bindings/linux-x86_64/py3.7`、`py3.8`），没有 3.10/3.11 版本。
   本仓库 `doc/setup_zh.md` 也明确规定 `conda create -n unitree-rl python=3.8`。
2. **必须有 NVIDIA GPU**，Isaac Gym 是 PhysX GPU 仿真，且不发布 PyPI 包，
   需从 NVIDIA 开发者官网手动下载（要注册账号）。
3. **numpy 钉死 1.20** —— Python 3.8 下正好可装；这也是必须用 3.8 的原因之一。

### 1.2 推荐配置表

| 项目 | 要求 | 推荐 |
| --- | --- | --- |
| 操作系统 | Ubuntu 18.04 / 20.04 / 22.04 | **Ubuntu 20.04**（Isaac Gym 兼容性最好） |
| GPU | NVIDIA，支持 CUDA 12.x | 见下方显存对照 |
| 驱动 | ≥ 525 | **≥ 535** |
| CUDA | 12.1（PyTorch 自带 runtime，无需系统装 CUDA Toolkit） | — |
| Python | **3.8** | 3.8（conda 环境） |
| PyTorch | 2.3.1 + cu121 | 照抄，不要换 |
| Isaac Gym | Preview 4（`isaacgym` 1.0rc4） | Preview 4 |
| 内存 | ≥ 16 GB | 32 GB |
| 磁盘 | ≥ 50 GB | **100 GB**（Isaac Gym + 日志 + 视频） |
| 显卡驱动/CUDA | 容器镜像需含 `nvidia-container-toolkit` | — |
| 显示 | 训练用 `--headless`，**无需 X server** | — |

### 1.3 显存 ↔ `--num_envs` 对照（**实测值**）

`legged_gym/envs/base/legged_robot_config.py` 默认 `num_envs = 4096`，
`cmd.md` 里的训练档位是 4096 与 8096。

以下为在 **RTX 4090 24G / 驱动 580 / CUDA 11.7 / torch 1.13.1 / isaacgym 1.0rc4**
上实测的峰值显存（1 秒粒度轮询 `nvidia-smi`）：

| 任务 | `--num_envs` | 峰值显存 | 吞吐 | 每 iteration |
| --- | --- | --- | --- | --- |
| `go2` | 4096 | **6.1 GB** | ~101k steps/s | 0.97 s |
| `go2` | 8096 | **7.7 GB** | ~126k steps/s | 1.55 s |
| `go2_moe_cts` | 8096 | **10.3 GB** | — | 2.23 s |

结论：**显存需求比直觉低得多**。MoE CTS 的 8096 档位也只要约 10 GB。

> 上述是短程（20–40 iteration）实测，长时间训练若有更多 rollout/日志开销会略有上浮，
> 但量级不会变。按此推算：
>
> | 卡型 | 可跑档位 |
> | --- | --- |
> | 12 GB（RTX 3060 / 4070） | 4096 档（go2），8096 需试 |
> | **16 GB（RTX 4080 / A4000）** | **8096 档全部任务，含 MoE CTS** |
> | **24 GB（RTX 4090 / A5000 / L40S）** | 8096 档绰绰有余，可尝试 16384 |
> | 40 GB+（A100 / L40S） | 非必需，除非要上 16384+ 或改大网络 |
>
> **选型建议修正：16 GB 卡就够跑本文档涉及的全部档位；24 GB 更稳但非硬性要求。**

### 1.4 云端实例类型参考

| 平台 | 规格 | 备注 |
| --- | --- | --- |
| 通用 | RTX 4090 24G / A5000 24G | 性价比最高，够跑 8096 |
| 高配 | A100 40G/80G、L40S 48G | 需要更大 `num_envs` 或用多卡时 |
| AWS | `g5.2xlarge`（A10G 24G）及以上 | — |
| 阿里云 | `gn7i`（A10）及以上 | — |
| AutoDL / 恒源云 | RTX 4090 / A100 镜像 | 国内方便，选 Ubuntu 20.04 + CUDA 12.1 镜像 |

**操作系统镜像选择**：Ubuntu 20.04 + 已装 NVIDIA 驱动（≥535）+ CUDA 12.1。
容器镜像需保证 `nvidia-smi` 可用。

---

## 2. 从零搭建（云端 GPU 机器）

```bash
# ---------- 1. 基础环境 ----------
# 系统需已装 NVIDIA 驱动（nvidia-smi 有输出）
sudo apt update && sudo apt install -y git wget build-essential

# ---------- 2. Miniconda + Python 3.8 ----------
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init --all
source ~/.bashrc

conda create -n unitree-rl python=3.8 -y
conda activate unitree-rl

# ---------- 3. PyTorch (CUDA 12.1) ----------
conda install -y pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    pytorch-cuda=12.1 -c pytorch -c nvidia

# ---------- 4. Isaac Gym Preview 4 ----------
# 从 https://developer.nvidia.com/isaac-gym 下载 IsaacGym_Preview_4_Package.tar.gz
tar -xzf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python && pip install -e . && cd -

# 验证（有窗口/无头均可，成功会打印 1080 个球的信息）
python isaacgym/examples/1080_balls_of_solitude.py

# ---------- 5. 本仓库 ----------
git clone <你的仓库地址> go2_rl_gym
cd go2_rl_gym
pip install -e ./rsl_rl
pip install -e .            # isaacgym 已就绪，此时可直接安装

# ---------- 6. 冒烟测试 ----------
python legged_gym/scripts/train.py --task=go2 --num_envs 8 --headless
```

启动正式训练（`cmd.md` 中的档位）：

```bash
# 4096 envs
python legged_gym/scripts/train.py --task=go2 --num_envs 4096 --headless
# 论文最终版 MoE CTS，8096 envs
python legged_gym/scripts/train.py --task=go2_moe_cts --num_envs 8096 --headless
```

---

## 3. RoboGauge 评估（可选，链路 B）

RoboGauge 是**独立进程**，在 CPU 上用 Mujoco 做 Sim2Sim 评估，
训练端通过 `--robogauge` 把策略推给它（默认端口 9973）。

```bash
git clone https://github.com/wty-yy/RoboGauge.git
cd RoboGauge
pip install -e .
```

单独开一个终端（或 tmux）先把 server 跑起来，再训练时加 `--robogauge`：

```bash
python legged_gym/scripts/train.py --task=go2_moe_cts --num_envs 8096 --headless --robogauge
```

结果写到 `logs/{exp_name}/{date}/robogauge_results/results_{it}.yaml`。
因为评估吃 CPU，云端机器建议**至少 8 核**，并在训练时留出 CPU 余量。

---

## 4. 链路 C（Mujoco Sim2Sim）—— 不需要云

纯 CPU、不需要 GPU，本地 `<repo>/go2_rl_gym/.venv` 已配好并验证通过。
如果云机器上也想跑：

```bash
pip install mujoco==3.2.3 pygame imageio imageio-ffmpeg matplotlib pyyaml \
    torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ./rsl_rl --no-deps && pip install -e . --no-deps
python deploy/deploy_mujoco/deploy_go2.py
```

详见 [`setup_local_cpu_zh.md`](setup_local_cpu_zh.md)。

---

## 5. 链路 D（真机部署）—— 在机器人端的 Jetson 上装

不在云端 x86 机器上做。按 JetPack 版本选 Python：

```bash
# JetPack 6 → Python 3.10；JetPack 5 → Python 3.8
conda create -n deploy python=3.10 -y && conda activate deploy
# 装对应 Jetson 的 PyTorch：https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python && pip install -e .
```

运行（假设低层控制网卡是 `eth0`）：

```bash
cd deploy/deploy_real && python deploy_real_go2.py eth0
```

---

## 6. 常见坑

| 现象 | 原因 / 解决 |
| --- | --- |
| `ModuleNotFoundError: isaacgym` | 没装 Isaac Gym，或 Python 版本不是 3.8 |
| `ImportError: libpython3.8.so.1.0` | conda 环境 `lib/` 不在 `LD_LIBRARY_PATH`，加上即可 |
| 无头机器上报 GLFW / 显示错误 | 训练务必加 `--headless`；Isaac Gym 无头模式不需要 X |
| 显存 OOM | 降 `--num_envs`，4096 → 2048 → 1024 逐档试 |
| 训练极慢 | 确认 `--sim_device` 与 `--rl_device` 都指向 GPU（默认 `cuda:0`） |
| 多卡 | 本仓库用 `--horovod` + horovod 做多 GPU，非必需 |
| 断点续训 | `--resume --load_run <run> --checkpoint <it>` |
| 磁盘爆掉 | `logs/` 会存 checkpoint 与视频，可用 `tools/logs_compress.py`、`tools/logs_merge.py` 清理 |

---

## 7. 一句话选型

> **Ubuntu 20.04 + NVIDIA 驱动 ≥535 + 16 GB 显存起（RTX 4080/A4000，24 GB 更稳）
> + Conda Python 3.8 + Isaac Gym Preview 4 + 50 GB 磁盘。**
> 训练加 `--headless`；Mujoco Sim2Sim 不需要 GPU，本地即可。

PyTorch 版本不是硬约束：仓库文档推荐 `2.3.1+cu121`，
而在 `1.13.1+cu117` 上实测训练、Play、ONNX 导出同样全部通过。
关键是 **Python 3.8 + Isaac Gym 1.0rc4**，torch 版本跟随 isaacgym 的 CUDA 绑定即可。
