# 训练监控与效果查看指令

> 适用机器：`gpufree-container`（RTX 4090 24G，无显示器）
> 工程路径：`/root/go2_rl_gym`

---

## 0. 前置：每次都要先激活正确环境

```bash
conda activate isaacgym
cd /root/go2_rl_gym
```

⚠️ **不激活会报错**。你容器的 `/root/.bashrc` 第 126 行默认 `conda activate isaaclab`（Python 3.11），
而 isaacgym 只装在 `isaacgym` 环境（Python 3.8）里。开新终端后务必手动切。

验证环境对不对：

```bash
python -c "import isaacgym, torch; print('isaacgym OK, cuda:', torch.cuda.is_available())"
```

---

## 1. 启动训练

### 方式 A：tmux（推荐，可随时断开重连）

```bash
conda activate isaacgym
cd /root/go2_rl_gym
tmux new -s train

# 在 tmux 里执行（日志同时写文件）
python legged_gym/scripts/train.py --task=go2_moe_cts --num_envs 8096 --headless \
    2>&1 | tee /root/train_$(date +%m%d_%H%M).log

# 按 Ctrl-B 然后按 D 脱离（训练继续跑）
# 回来：tmux attach -t train
# 看会话列表：tmux ls
```

### 方式 B：nohup

```bash
conda activate isaacgym
cd /root/go2_rl_gym
nohup python legged_gym/scripts/train.py --task=go2_moe_cts --num_envs 8096 --headless \
    > /root/train.log 2>&1 &
echo $! > /root/train.pid
```

### 常用 task 与档位

| `--task` | experiment_name（日志目录名） | 建议 `--num_envs` |
| --- | --- | --- |
| `go2` | `go2_ppo` | 4096 |
| `go2_cts` | `go2_cts` | 8096 |
| `go2_moe_cts` | `go2_moe_cts` | 8096（论文最终版） |
| `go2_moe_ng_cts` | `go2_moe_no_goal_cts` | 8096 |
| `go2_mcp_cts` | `go2_mcp_cts` | 8096 |
| `go2_ac_moe_cts` | `go2_ac_moe_cts` | 8096 |

续训：

```bash
python legged_gym/scripts/train.py --task=go2_moe_cts --num_envs 8096 --headless \
    --resume --load_run <run名> --checkpoint <迭代数>
```

---

## 2. 看训练曲线（TensorBoard）— 最直观

### 启动

```bash
conda activate isaacgym
tensorboard --logdir /root/go2_rl_gym/logs --port 6006 --bind_all
```

### 在 VSCode 里打开

VSCode Remote-SSH 通常会自动转发端口。若没有：

1. 底部面板切到 **PORTS（端口）** 标签
2. 点 **Forward a Port**，填 `6006`
3. 浏览器打开 `http://localhost:6006`

### 该看哪些曲线

| 曲线 tag | 含义 | 健康表现 |
| --- | --- | --- |
| `Episode/rew_tracking_lin_vel` | 跟踪线速度 | 持续上升并趋稳 |
| `Episode/rew_tracking_ang_vel` | 跟踪角速度 | 上升 |
| `Episode/rew_collision` | 碰撞惩罚 | 趋近 0 |
| `Episode/rew_feet_regulation` | 足端规整 | 趋近 0（不抖腿） |
| `Episode/rew_dof_acc`、`rew_action_rate` | 动作平滑度 | 趋近 0（动作不抖） |
| `Episode/terrain_level_*` | 地形课程进度 | 逐步上升=在爬更难的坡/台阶 |
| `Episode/episode_length` | 单回合存活步数 | 上升 |
| `Loss/value_function` | 价值损失 | 下降后平稳 |
| `Loss/surrogate` | 策略损失 | 小幅震荡 |
| `Policy/mean_noise_std` | 探索噪声 | 缓慢下降 |
| `Perf/total_fps` | 训练速度 | 4096 envs 约 10 万 steps/s |

---

## 3. 命令行看进度（不开浏览器时）

```bash
# 实时跟踪
tail -f /root/train_*.log

# 只提关键几行
grep -E "Learning iteration|Mean reward|Mean episode length|Computation" /root/train_*.log | tail -30

# 看是否还在跑
ps -ef | grep train.py | grep -v grep
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv

# 训练是否报错
grep -iE "error|traceback|out of memory" /root/train_*.log | head
```

---

## 4. 日志目录结构

```text
logs/<experiment_name>/<MonDD_HH-MM-SS>_<run_name>/
├── config.yaml                  # 本次训练全部超参，可事后查证
├── model_0.pt                   # checkpoint（每 50 迭代存一次）
├── model_50.pt
├── model_100.pt
└── events.out.tfevents.*        # TensorBoard 事件文件

logs/<experiment_name>/exported/policies/      # 由 play.py 生成
├── policy.pt                    # TorchScript → Mujoco Sim2Sim 用
├── policy.onnx                  # ONNX → Sim2Real 部署用
└── policy.pkl                   # 原始权重
```

查最新 checkpoint：

```bash
ls -lt logs/go2_moe_cts/*/model_*.pt | head -5
```

---

## 5. 看机器人实际走起来

云端桌面已开启（Xorg 在 **`DISPLAY=:20`**，OpenGL 直通 4090），`play.py` 可以直接弹窗。
如果你只想快速看效果而机器没桌面，也可以走 5.3 的「云端导出 → 本地 Mujoco」方案。

### 5.1 直接看（预训练权重，已就绪）

云端已下载好官方权重 `model_164000.pt`（22.6 MB，164k 迭代，RoboGauge 0.6715）
到 `logs/go2_moe_cts/0_pretrained/`。

```bash
conda activate isaacgym
cd /root/go2_rl_gym
python legged_gym/scripts/play.py --task=go2_moe_cts --num_envs 8 \
    --load_run 0_pretrained --checkpoint 164000
```

窗口会弹在云端桌面上。`play.py` 里 `FIX_COMMAND = True` 写死了前进指令，
所以机器人会以 `vx=1.0 m/s` 恒定直行。

> 权重来源：HuggingFace 的 `wty-yy/go2_rl_gym_data`。
> **云端访问不了 huggingface.co（连接超时）**，用镜像 `hf-mirror.com` 下载：
> ```bash
> curl -sL --fail -o model_164000.pt \
>   https://hf-mirror.com/wty-yy/go2_rl_gym_data/resolve/main/go2_moe_cts_high_slope_thre_164k_0.6715_20260419/model_164000.pt
> ```
> 注意：只有带 `model_<it>.pt` 的**完整训练权重**能给 `play.py` 用；
> 各 ckpt 目录里的 `policy.pt` / `policy.onnx` 是 TorchScript/ONNX 导出件，只能给 Mujoco 部署用。

### 5.2 看自己训练的模型

```bash
conda activate isaacgym && cd /root/go2_rl_gym

# 查有哪些 run 和 checkpoint
ls -lt logs/go2_moe_cts/*/model_*.pt | head

python legged_gym/scripts/play.py --task=go2_moe_cts --num_envs 8 \
    --load_run <run名> --checkpoint <迭代数>
```

> ⚠️ **不要省略 `--load_run`**。`get_load_path()` 是「按目录名字母序取最后一个」
> （`legged_gym/utils/helpers.py:74-97`）。预训练目录特意命名为 `0_pretrained`
> 让它排在所有 `Oct02_...` 训练目录之前，这样自动加载才会选中你训练的模型而不是预训练权重。

`play.py` 的评估设置（与训练时不同，属正常）：

- 关闭观测噪声、域随机化、推力扰动
- 地形固定为 7×7、关闭课程学习
- `num_envs` 上限 100，`play.py` 会自动截断
- 跑满 `10 × max_episode_length` 步后自动退出
- 结束后把策略导出到 `logs/<exp>/exported/policies/`

### 5.3 备选：云端导出 → 本地 Mujoco 看

本地机器已配好 CPU Sim2Sim 环境（`DISPLAY=:1` 可用）。

**云端导出：**

```bash
conda activate isaacgym && cd /root/go2_rl_gym
python legged_gym/scripts/play.py --task=go2_moe_cts --num_envs 8 --headless \
    --load_run <run名> --checkpoint <迭代数>
```

**本地拉取**（scp 链路已验证）：

```bash
scp 183.147.142.40:/root/go2_rl_gym/logs/go2_moe_cts/exported/policies/policy.pt \
    <repo>/go2_rl_gym/deploy/pre_train/go2/my_policy.pt
```

**本地播放**：把 `deploy/deploy_mujoco/configs/go2.yaml` 的 `policy_path` 改指向 `my_policy.pt`，然后

```bash
cd <repo>/go2_rl_gym
.venv/bin/python deploy/deploy_mujoco/deploy_go2.py
```

可选地形：`flat.xml`、`stairs.xml`、`race_track.xml`、`cross_slope.xml`、`cross_stairs.xml`（改 `xml_path`）。
接上 Xbox 手柄可遥操作；没有手柄则按 `cmd_init` 恒定前进。

---

## 6. 量化评估（RoboGauge，可选）

RoboGauge 是独立的 CPU 进程，用 Mujoco 做 Sim2Sim 打分，训练端异步把策略推给它。

```bash
# 终端 1：起评估 server
git clone https://github.com/wty-yy/RoboGauge.git && cd RoboGauge
pip install -e .
python -m robogauge.server   # 具体启动方式见其 README

# 终端 2：带 --robogauge 训练
cd /root/go2_rl_gym
python legged_gym/scripts/train.py --task=go2_moe_cts --num_envs 8096 --headless --robogauge
```

结果写在 `logs/<exp>/<date>/robogauge_results/results_<it>.yaml`，同时进 TensorBoard。
注意评估吃 CPU，会分走一部分算力，训练速度会下降。

---

## 7. 排错速查

| 现象 | 原因 / 解决 |
| --- | --- |
| `ModuleNotFoundError: isaacgym` | 忘了 `conda activate isaacgym`（多半被 `isaaclab` 顶掉了） |
| `ImportError: libpython3.8.so.1.0` | 同上；环境自带的激活钩子只在 `conda activate` 时生效 |
| `CUDA out of memory` | 降 `--num_envs`（实测 8096 的 MoE 只要 ~10 GB，正常不该 OOM） |
| play.py 看不到窗口 | 云端无显示器，属正常；按第 5 节在本地看 |
| TensorBoard 打不开 | 确认 `--bind_all` 已加，VSCode PORTS 面板手动转发 6006 |
| 训练中断 | `--resume --load_run <run> --checkpoint <it>` 续训 |
| 磁盘满 | `logs/` 累积 checkpoint，可用 `tools/logs_compress.py`、`tools/logs_merge.py` 清理 |
