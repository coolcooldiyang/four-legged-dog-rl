IMU_PORT = "/dev/serial/by-id/usb-DM-Tech_DM-IMU-L1_DMIMU20250212-if00"
IMU_BAUDRATE = 921600

CAN_SERIALS = [
    "14AA044B241402B10DDBDAFE448040BB",
    "A5E44EC9FE3B74CEFBF01A219CF688EE",
]
CAN_CHANNELS = [0, 1]
CAN_BITRATE = 1000000
CANFD_ENABLED = False
# Model/action leg order from the MIT URDF DOF order:
# leg0=FL, leg1=FR, leg2=RL, leg3=RR.
CAN_BUS_MAP = [
    (CAN_SERIALS[0], 0),
    (CAN_SERIALS[0], 1),
    (CAN_SERIALS[1], 0),
    (CAN_SERIALS[1], 1),
]

# Raw motor direction relative to the STM32 startup sequence convention.
# Keep this matched to the already-verified standing startup posture.
STARTUP_JOINT_DIRECTIONS = [
    [1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, 1.0],
]

# Raw motor direction relative to the trained URDF/model residual convention.
# Training computes model_target = default_dof_pos + action * action_scale.
# Deployment sends raw_target = raw_origin + direction * (model_target - default_dof_pos).
# -1.0 mirrors the residual interval around default_dof_pos/raw_origin; it does
# not negate absolute joint angles or mirror around URDF joint limits.
#
# 2026-10 实机试验记录：
#   曾把髋列翻转为 [+1.0, +1.0, -1.0, -1.0] 试机 → **实机表现更怪**，
#   说明髋方向不是问题所在，已改回原标定值。
#   结论：关节方向符号**不是** Go2 系列策略在实机上异常的原因。
#
# 髋列 = [-1.0, -1.0, +1.0, +1.0]  (FL, FR, RL, RR)  ← 原标定，已恢复
MODEL_JOINT_DIRECTIONS = [
    [-1.0, -1.0, -1.0],   # FL
    [-1.0,  1.0,  1.0],   # FR
    [ 1.0, -1.0, -1.0],   # RL
    [ 1.0,  1.0,  1.0],   # RR
]

OBS_DIM = 45
# ---------------------------------------------------------------------------
# Policy observation layout (MUST match the trained model file, not this file)
# ---------------------------------------------------------------------------
# This robot carries two generations of policies with different layouts:
#
#   legacy (HIMLoco / AMP_for_hardware era: policy.pt, policy_stair.pt)
#     [base_ang_vel(3), projected_gravity(3), command(3),
#      joint_pos_error(12), joint_vel(12), last_action(12)]  x 5 frames = 225
#
#   deployable (unitree_rl_gym era, G1-style "blind" layout)
#     [base_ang_vel(3), projected_gravity(3), command(3),
#      joint_pos_error(12), joint_vel(12), last_action(12),
#      sin_phase(1), cos_phase(1)]                            x 1 frame  = 47
#
# Both put the command at offset 6..8, so the only differences are the number of
# stacked frames and the two trailing gait-clock entries. The real history length
# is read from the model file; HISTORY_LENGTH below is only a fallback for input
# dimension inference (the previous value of 6 here was simply wrong: the shipped
# legacy models take 225 = 45 x 5).
OBS_DIM_DEPLOYABLE = 47
HISTORY_LENGTH = 5
# Gait-clock period for the deployable layout. Must equal the value used in
# training (dog4438_config.py / dog4438_env.py: PHASE_PERIOD = 0.8 s).
OBS_PHASE_PERIOD_S = 0.8
# ---------------------------------------------------------------------------
# 观测帧里"前三个 3 向量"的顺序（必须与训练时一致！）
# ---------------------------------------------------------------------------
# 可用取值（逗号分隔的排列）：ang_vel / gravity / cmd
#   "ang_vel,gravity,cmd"  —— 我们契约里的顺序；也是 unitree_rl_lab 官方 G1/Go2 的顺序
#   "gravity,cmd,ang_vel"  —— go2dog_policy_1.15M.pt（SB3 策略）在 MuJoCo 实测里
#                             唯一能让狗安静站住、力矩不打满的顺序；用契约顺序时
#                             它会把重力当成偏航指令，一直顶大腿力矩（实测打满 77%）
#   "gravity,ang_vel,cmd"  —— 该策略的次优候选（打满 10.7%）
# 改这里等于换策略的接口，改完必须重启节点；换回旧策略请改回 "ang_vel,gravity,cmd"。
OBS_HEAD_ORDER = "ang_vel,gravity,cmd"
ACTION_DIM = 12

# ---------------------------------------------------------------------------
# 策略关节顺序 → dm 电机顺序 的置换
#
# dm_dog_controller 的关节顺序（DEFAULT_DOF_POS 的顺序）是:
#     FL hip/thigh/calf, FR hip/thigh/calf, RL hip/thigh/calf, RR hip/thigh/calf
#
# 但训练用的 URDF 决定了策略输出的顺序。两者不一定一致：
#
#   go2.urdf  关节顺序 FL, FR, RL, RR  → 与 dm 一致，无需置换（恒等）
#   go1dm.urdf 关节顺序 LF, LH, RF, RH → 需要置换
#
# 置换的含义:  dm_action[i] = policy_action[PERM[i]]
#
# go1dm 的映射推导:
#     dm[0:3]  FL ← policy[0:3]  LF
#     dm[3:6]  FR ← policy[6:9]  RF
#     dm[6:9]  RL ← policy[3:6]  LH
#     dm[9:12] RR ← policy[9:12] RH
# 这个置换恰好是对合（自身即逆），所以观测侧用同一个数组反转即可。
# ---------------------------------------------------------------------------
MODEL_JOINT_PERMUTATION = list(range(ACTION_DIM))   # 默认恒等（go2 策略）

# 若加载 go1dm 策略，改成:
# MODEL_JOINT_PERMUTATION = [0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]


# Observation layout used by HIMLoco legged_gym:
# command(3), base_ang_vel(3), projected_gravity(3),
# joint_pos_error(12), joint_vel(12), last_action(12).
DEFAULT_COMMAND = [0.0, 0.0, 0.0]
COMMAND_SCALE = [2.0, 2.0, 0.25]
# Leg-major order matching the MIT URDF DOF order used by legged_gym:
# FL hip/thigh/calf, FR hip/thigh/calf, RL hip/thigh/calf, RR hip/thigh/calf.
#
# 2026-10 迁移到 go2_rl_gym 的 go2 PPO 策略：改为它训练时的 default_joint_angles
# （legged_gym/envs/go2/go2_config.py 的 FL/FR/RL/RR default_joint_angles）。
# 旧值（legacy htdw_4438 策略）：[0.0, 0.8, -1.5] x 4
#
# ⚠️ 重要：本值在 RL 路径中会被抵消，单独改它**不会**改变狗的实际站立位姿。
#   观测算的是 dof_pos - default_dof_pos，而
#     dof_pos = DEFAULT_DOF_POS + dirs * (raw_q - rl_origin)   (deployment.py:1443)
#   两者相减后 DEFAULT_DOF_POS 消失，输出目标 raw_target = rl_origin + dirs*action*scale
#   也完全不含本值。
#   真正决定「零动作时狗摆成什么姿势」的是 FIXED_RL_ORIGIN_POSE_RAW（见下方）。
#   本值仍需同步修改，用于保持模型空间自洽，以及让 tools/test_obs_layout.py 有意义。
# 模型坐标系下的「站立默认关节角」。它必须等于训练时 URDF 的
# default_joint_angles，否则观测里的 (dof_pos - default) 会有常值偏置，
# 动作目标 default + residual 的起点也会错。
#
# go2.urdf 的默认角（当前生效）：
DEFAULT_DOF_POS = [
     0.1, 0.8, -1.5,   # FL hip/thigh/calf
    -0.1, 0.8, -1.5,   # FR
     0.1, 1.0, -1.5,   # RL
    -0.1, 1.0, -1.5,   # RR
]

# go1dm.urdf（真狗尺寸 Go1）的默认角，按 dm 的 FL/FR/RL/RR 顺序重排。
# 加载 go1dm 策略时，连同 MODEL_JOINT_PERMUTATION 一起换成这一组：
# DEFAULT_DOF_POS = [
#      0.119837, 0.994838, -1.780236,   # FL (← go1dm LF)
#     -0.088174, 0.994838, -1.780236,   # FR (← go1dm RF)
#      0.090845, 0.994838, -1.780236,   # RL (← go1dm LH)
#     -0.090082, 0.994838, -1.780236,   # RR (← go1dm RH)
# ]
ANGULAR_VELOCITY_SCALE = 0.25
DOF_VELOCITY_SCALE = 0.05

STARTUP_KP = 30.0
STARTUP_KD = 1.0

# Seconds to wait after the motor power is switched on BEFORE sending anything that
# could produce torque. The DM motor encoders need roughly this long to initialise;
# a control command sent earlier is computed against a bogus position reading, which
# makes the dog jump. Enforced inside DogStartupController.run() BEFORE the initial
# position read, because the whole startup trajectory is built from that reading.
# Set to 0.0 to disable (not recommended).
MOTOR_INIT_SETTLE_SECONDS = 5.0

# --- Stall / overload guard (see bs_hardware/motor_guard.py) -----------------
# A jammed joint keeps pushing for its commanded position, so its current and the
# reported torque climb. On this robot that has already tripped the whole power
# supply: every motor went silent, the dog lost torque mid-motion and rolled over,
# and the policy node died with a motor feedback timeout.
#
# Phase 1 (default): MONITOR ONLY. Run the dog normally, then read the printed
#   "MOTOR GUARD SUMMARY" maxima to learn what normal looks like.
# Phase 2: set MOTOR_GUARD_ABORT = True (after confirming the normal maxima stay
#   well below the abort thresholds) so a stall disables the motors instead of
#   letting the supply trip.
MOTOR_GUARD_ABORT = True
# Thresholds calibrated from real runs (stand up + turn + stop, then forward walk,
# then forward + turn; ~5600 samples each):
#
#   condition        worst position error   worst |torque|
#   turn only              0.399 rad            5.877 Nm
#   forward                0.520 rad            8.101 Nm
#   forward + turn         0.609 rad           10.580 Nm
#
# Two consequences:
#   * the position error is the usable stall signal - normal peaks stay under
#     0.61 rad while a jammed leg lets it run past 1.3 rad;
#   * the torque is NOT usable as an abort trigger on this robot: normal walking
#     already reaches 10.6 Nm while the BM calf feedback range is +-18 Nm, so there
#     is no separating margin. It is kept as a very high last-resort net only.
#     (It is still worth reading in the summary: the calves work near their limit.)
MOTOR_GUARD_WARN_POS_ERROR_RAD = 0.8     # warning: |commanded - measured|
MOTOR_GUARD_WARN_TORQUE_NM = 12.0        # warning: |torque| from the feedback frame
MOTOR_GUARD_ABORT_POS_ERROR_RAD = 1.3    # primary abort criterion
MOTOR_GUARD_ABORT_TORQUE_NM = 15.5       # last resort only (BM calf scale is +-18)
MOTOR_GUARD_ABORT_SECONDS = 0.3          # must be exceeded continuously this long

BM_CALF_STARTUP_KP = 40.0
BM_CALF_STARTUP_KD = 20.0

# plane


# FIXED_RL_ORIGIN_POSE_RAW = [-0.870718, 3.101587, 0.467296, 0.936332, -1.398680, 2.475691, 0.752842, 0.472839, 1.270731, -0.849737, 0.823796, -0.169316]
# FIXED_RL_ORIGIN_POSE_RAW= [-0.876059, 3.134775, 0.504496, 0.931754, -1.406310, 2.476841, 0.768483, 0.483139, 1.315984, -0.845159, 0.834859, -0.185039]
FIXED_RL_ORIGIN_POSE_RAW = [-0.904669, 3.151560, 0.449655, 0.898947, -1.437972, 2.504454, 0.799382, 0.492294, 1.318285, -0.816548, 0.801289, -0.214569]
# Optional fixed RL origin/startup pose in raw motor radians.
# Paste the "copyable flat current_feedback_pose_raw" log here when you want to
# use a manually captured standing pose.
# Leg order: FL hip/thigh/calf, FR hip/thigh/calf, RL hip/thigh/calf, RR hip/thigh/calf.
# Leave empty to use the pose generated by the startup sequence.

# Gains used when the main translation policy (model parameter) is active.
# This policy is selected for forward motion, including forward + yaw.
#
# 2026-10 实机试机结论：
#   kp=120/kd=2.2 (legacy)  → 偏硬
#   kp=20 /kd=0.5 (训练值)  → **腿在自己"响"**（实机表现为"腿动得太快"）
#   kp=60 /kd=1.5 (折中)    → 狗身抖动严重（kp 抬高把闭环带宽抬上去了）
#   现在只提阻尼、不提刚度：kp=20 / kd=1.5
#
# 依据（大腿惯量 J≈0.01 kg*m^2）：
#   omega_n = sqrt(kp/J) ≈ 7 Hz（与步态 1.67 Hz 无关，是关节自身固有频率）
#   zeta = kd / (2*sqrt(kp*J))  
#     kd=0.5 → zeta≈0.56  欠阻尼，腿在 7 Hz 自激 → "动得太快"
#     kd=1.5 → zeta≈1.68  过阻尼，自激被压住
#   kp 不变 ⇒ omega_n 不变 ⇒ 不会把高频耦合进机身（这是 60/1.5 失败的原因）
#
# ⚠️ kp 与 launch 里的 action_clip 耦合：峰值需求力矩 ≈ kp × action_clip × 0.25
#   kp=20 + clip=4.0 → 20 Nm（不变）
TRANSLATION_POLICY_RL_KP = 25.0
TRANSLATION_POLICY_RL_KD = 2.0
TRANSLATION_POLICY_BM_CALF_KP = 25.0
TRANSLATION_POLICY_BM_CALF_KD = 2.0

# Gains used when the yaw policy (yaw_model parameter) is active.
# This policy is selected for backward, lateral, pure yaw, and lateral+yaw commands.
#
# 2026-10 与 TRANSLATION_* 保持一致（当前 use_dual_policy=False，这组不会被使用）。
# 旧值：RL 65.0/1.2，BM calf 75.0/20.0
YAW_POLICY_RL_KP = 20.0
YAW_POLICY_RL_KD = 1.5
YAW_POLICY_BM_CALF_KP = 20.0
YAW_POLICY_BM_CALF_KD = 1.5

# Backward-compatible aliases for modules that still read the old names.
RL_KP = TRANSLATION_POLICY_RL_KP
RL_KD = TRANSLATION_POLICY_RL_KD

# BM calf motors use a different driver/protocol from the other 8 MIT motors.
# These default gains only apply to calf joints when no per-policy override is active.
BM_CALF_KP = TRANSLATION_POLICY_BM_CALF_KP
BM_CALF_KD = TRANSLATION_POLICY_BM_CALF_KD

# Softer BM calf gains used only during stand-up / lie-down posture motion.
# Keep these lower than BM_CALF_KP to avoid calf snap when motors first engage
# or when the reverse startup sequence begins.


# RL action is interpreted as a joint-position residual around startup_pose.
RL_ACTION_SCALE = 0.125


# ---------------------------------------------------------------------------
# Observation layout resolver
# ---------------------------------------------------------------------------
# Given the flat input width a TorchScript policy expects, work out how that
# vector is laid out. Returns (per_frame, history, has_phase, layout_name) or
# None when the width matches no known layout.
#
# Rationale: the deployment code used to infer the layout from the traced code
# (looking for "command = torch.slice(...)"). Policies exported from rsl_rl's
# plain actor contain no such slice, and the old fallback then silently assumed
# "command first" - the exact opposite of the trained layout. Resolving from the
# input width is unambiguous here because 45 and 47 are coprime.
def resolve_observation_spec(total_input_dim: int):
    if total_input_dim is None or total_input_dim <= 0:
        return None
    if total_input_dim % OBS_DIM_DEPLOYABLE == 0:
        return OBS_DIM_DEPLOYABLE, total_input_dim // OBS_DIM_DEPLOYABLE, True, "deployable-47"
    if total_input_dim % OBS_DIM == 0:
        return OBS_DIM, total_input_dim // OBS_DIM, False, "legacy-45"
    return None
