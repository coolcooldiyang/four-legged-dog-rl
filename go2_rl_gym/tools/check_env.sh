#!/usr/bin/env bash
# go2_rl_gym 环境自检脚本
# 用法： bash check_env.sh
# 在装有 isaacgym 的云端机器上运行，把输出整段贴回来即可判断能否跑通。
set +e
hr(){ printf '\n=== %s ===\n' "$1"; }
# 自动选择解释器：优先 conda/venv 里的 python，否则退回 python3
if command -v python >/dev/null 2>&1; then PY=python; else PY=python3; fi
echo "使用解释器: $PY -> $(command -v $PY)"

hr "0. 主机"
hostname; whoami
grep -E '^(PRETTY_NAME|VERSION_ID)' /etc/os-release 2>/dev/null
echo "kernel: $(uname -r)"; echo "arch: $(uname -m)"

hr "1. GPU / 驱动"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv
else
  echo "!! 没有 nvidia-smi"
fi

hr "2. CUDA toolkit"
command -v nvcc >/dev/null 2>&1 && nvcc --version | tail -2 || echo "无 nvcc（isaacgym 不一定需要）"
ls -d /usr/local/cuda* 2>/dev/null

hr "3. Python / conda"
echo "which python : $(command -v python || echo '<无>')"
"$PY" -V 2>&1
echo "which python3: $(command -v python3)"; python3 -V 2>&1
if command -v conda >/dev/null 2>&1; then
  conda --version; echo "--- envs ---"; conda env list
else
  echo "无 conda"
fi
echo "CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<未激活>}"

hr "4. isaacgym"
"$PY" - <<'PY' 2>&1
try:
    import isaacgym, os
    print("isaacgym 路径:", os.path.dirname(isaacgym.__file__))
    from isaacgym import gymapi
    print("gymapi 导入成功")
    try:
        g = gymapi.acquire_gym(); print("acquire_gym OK")
    except Exception as e:
        print("acquire_gym 失败:", type(e).__name__, e)
    print("version:", getattr(isaacgym, "__version__", "n/a"))
except Exception as e:
    print("isaacgym 不可用:", type(e).__name__, e)
PY

hr "5. PyTorch / CUDA"
"$PY" - <<'PY' 2>&1
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda version:", torch.version.cuda)
    print("device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu{i}: {torch.cuda.get_device_name(i)}  {p.total_memory/1024**3:.1f} GB")
except Exception as e:
    print("torch 不可用:", type(e).__name__, e)
PY

hr "6. 关键依赖"
"$PY" - <<'PY' 2>&1
for m in ["numpy", "rsl_rl", "legged_gym", "mujoco", "tensorboard"]:
    try:
        mod = __import__(m)
        print(f"{m:12s}", getattr(mod, "__version__", "(无 __version__)"), getattr(mod, "__file__", ""))
    except Exception as e:
        print(f"{m:12s} 缺失 ({type(e).__name__})")
PY
pip show go2_rl_gym 2>/dev/null | head -3

hr "7. 工程目录"
for d in ~/go2_rl_gym /root/go2_rl_gym /root/autodl-tmp/go2_rl_gym /home/*/go2_rl_gym /root/*/go2_rl_gym; do
  [ -d "$d" ] && echo "找到: $d"
done
echo "(没找到的话请手动确认仓库路径)"

hr "8. 磁盘 / 内存 / 显示"
df -h / /root /root/autodl-tmp 2>/dev/null | sort -u
free -g | head -2
echo "DISPLAY=${DISPLAY:-<空，训练加 --headless 即可>}"

hr "9. isaacgym 真实建仿真环境（最关键一步）"
"$PY" - <<'PY' 2>&1
try:
    from isaacgym import gymapi
    g = gymapi.acquire_gym()
    sp = gymapi.SimParams(); sp.physx.use_gpu = True
    sim = g.create_sim("cuda:0", "cuda:0", gymapi.SIM_PHYSX, sp)
    print("PhysX 仿真环境创建成功。")
    g.destroy_sim(sim)
except Exception as e:
    print("建环境失败:", type(e).__name__, e)
PY

hr "自检结束"
