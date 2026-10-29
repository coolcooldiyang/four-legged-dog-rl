"""基于真狗实测尺寸的 Go1 URDF（go1dm）的 RL 训练配置。

## 为什么有这个机器人

原来的 go2 任务训练在 Unitree Go2 的 URDF 上，而实机是自研的四足狗，
两者几何/质量都不同 —— 这是策略迁移到实机效果差的根本原因之一。

本配置使用 `/home/yjp/legged_noetic/rootfs/tmp/legged_dm_candidate10/go1.urdf`，
它是 **Go1 模型按真狗实测尺寸改造**过的版本（见 `measured-model-report.json`）：

    大腿长          0.25 m      （Go2 是 0.213）
    小腿长          0.23 m      （Go2 是 0.213）
    髋前后偏置      0.22 m
    髋侧向偏置      0.05 m
    总质量          11.51 kg    （Go2 约 15 kg）
    站立高度        0.3106 m    （Go2 约 0.38）
    站立时足端前后间距  0.44 m
    站立时足端侧向间距  前 0.301 / 后 0.293 m

## URLF 与原版的差异（已处理）

| 项 | 原始 URDF | 训练版处理 |
| --- | --- | --- |
| visual mesh | 指向已删除的 legged_control（.dae） | 换成与 collision 相同的 primitive |
| `<gazebo>` ×24 | 存在 | 删除（Isaac Gym 用不到） |
| `<transmission>` ×12 | 存在 | 删除 |
| collision | 已经是 primitive（box/cylinder/sphere） | 保留原样 |

训练版 URDF: `resources/robots/go1dm/go1dm.urdf`（16975 字节）

## Isaac Gym 实测加载结果（2026-10-16）

    dof_count = 12，顺序 LF/LH/RF/RH 各 (HAA, HFE, KFE)
      LF_HAA  ±0.863 rad  effort=10 Nm  vel=30.1
      LF_HFE  -0.686~4.501  effort=10 Nm
      LF_KFE  -2.818~-0.888 effort=18 Nm
    body_count = 18
      [0] base, [1] LF_hip, [2] LF_thigh, [3] LF_calf, [4] LF_FOOT,
      [5..8] LH_*, [9..12] RF_*, [13..16] RH_*, [17] base_imu

**重要巧合**：HAA 关节索引是 `[0, 3, 6, 9]`，与 Go2 完全一致 ——
所以 `_reward_hip_to_default` / `_reward_x_command_hip_regular`
里硬编码的髋索引**不需要改**。

## 与 go2 配置的关系

本文件 `class GO1DMCfg(GO2Cfg)`，**只覆盖机器人相关的项**（URDF 路径、关节名、
默认角、站立高度、接触力阈值）。奖励权重、地形、课程、观测缩放等全部继承，
包括 2026-10-15 方案 A 的两项改动：
    termination  = -200.0
    ang_vel_xy   = -0.15
这样便于和 go2 的结果直接对比（单一变量：只换机器人模型）。
"""

import math

from legged_gym.envs.go2.go2_config import GO2Cfg, GO2CfgPPO

# 真狗实测的站立默认关节角（来自 reference.info / task.info 的 defaultJointState）
# 注意顺序是 (LF, LH, RF, RH)，且左右 HAA 异号（与 Go2 相同的展开模式）
DEFAULT_JOINT_ANGLES = {
    'LF_HAA': 0.11983700862503058,
    'LF_HFE': 0.9948376736367679,
    'LF_KFE': -1.7802358370342162,
    'LH_HAA': 0.09084457871246335,
    'LH_HFE': 0.9948376736367679,
    'LH_KFE': -1.7802358370342162,
    'RF_HAA': -0.08817437937545772,
    'RF_HFE': 0.9948376736367679,
    'RF_KFE': -1.7802358370342162,
    'RH_HAA': -0.09008201537322993,
    'RH_HFE': 0.9948376736367679,
    'RH_KFE': -1.7802358370342162,
}

#: URDF 里参考的 CoM 高度（reference.info 的 comHeight）
STANDING_HEIGHT = 0.31056241697284015

#: 由 URDF 正运动学算出的 base 原点高度（足端球心在 base 下方 0.2906 m）
#: 这个才是 `correct_base_height` 奖励该用的量。
#: 校验：足端前后间距算得 0.4400 m，与实测报告 0.44 完全一致。
FK_STANDING_HEIGHT = 0.2906


class GO1DMCfg(GO2Cfg):
    class asset(GO2Cfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1dm/go1dm.urdf'
        name = "go1dm"
        # ⚠️ link 名是 LF_FOOT / LH_FOOT 等，legged_gym 用 `name in s` 大小写敏感匹配，
        #    所以这里必须写大写 "FOOT"
        foot_name = "FOOT"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0

    class init_state(GO2Cfg.init_state):
        # 站立高度 0.3106 m，出生点稍高一点让它自然落下
        pos = [0.0, 0.0, 0.34]
        default_joint_angles = dict(DEFAULT_JOINT_ANGLES)

    class control(GO2Cfg.control):
        # ⚠️ 增益的 key 是【子串匹配】关节名（legged_robot.py:837-840）：
        #       for dof_name in stiffness.keys():
        #           if dof_name in name: p_gains[i] = stiffness[dof_name]
        #    go2 用的是 'joint'（能匹配 'FL_hip_joint'），但本机器人的关节名是
        #    'LF_HAA' / 'LF_HFE' / 'LF_KFE' —— 'joint' 不是子串，会导致
        #    【全部 12 个关节增益为 0，机器人直接瘫倒】（已实测踩到）。
        #    这里按关节类型分别给 key。
        control_type = 'P'
        # 2026-10-16 平地静态测试（16 env，零动作，400 步）扫描结果：
        #     kp=20/kd=0.5 → base z 0.2177（下沉 7 cm，太软）
        #     kp=30/kd=0.7 → 0.2608
        #     kp=40/kd=1.0 → 0.2784   ← 选用，最接近 FK 站立高度 0.2906
        #     kp=60/kd=1.5 → 0.2817（再高收益很小，且会撞 10 Nm 限幅）
        stiffness = {'HAA': 40.0, 'HFE': 40.0, 'KFE': 40.0}
        damping = {'HAA': 1.0, 'HFE': 1.0, 'KFE': 1.0}
        action_scale = 0.25
        decimation = 4

    class rewards(GO2Cfg.rewards):
        # 用 URDF 正运动学算出的「base 原点 → 足端球心」高度 = 0.2906 m
        # （URDF 自带的 comHeight 0.3106 量的是 CoM，不是 base 原点，两者差 2 cm）
        base_height_target = FK_STANDING_HEIGHT
        # Go2 是 147（对应 15 kg），本机器人 11.51 kg → 按重量等比缩放
        max_contact_force = 147.0 * (11.51 / 15.0)  # ≈ 112.8
        # 足端最小间距沿用 go2（两者足端尺寸接近）
        min_legs_distance = 0.1

    class env(GO2Cfg.env):
        # 观测维度不变（45），因为感知输入格式完全一样
        num_observations = 45
        num_privileged_obs = 45 + 3 + 4 + 12 + 12 + 187
        episode_length_s = 25


class GO1DMCfgPPO(GO2CfgPPO):
    class runner(GO2CfgPPO.runner):
        run_name = ''
        experiment_name = 'go1dm_ppo'
        max_iterations = 150000
        save_interval = 500
