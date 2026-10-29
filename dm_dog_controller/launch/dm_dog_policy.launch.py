from launch import LaunchDescription
from launch.substitutions import EnvironmentVariable, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node


def generate_launch_description():
    third_party_lib_dir = PathJoinSubstitution(
        [FindPackageShare("dm_dog_controller"), "third_party_libs"]
    )
    ld_library_path = [
        third_party_lib_dir,
        ":",
        EnvironmentVariable("LD_LIBRARY_PATH", default_value=""),
    ]

    return LaunchDescription(
        [
            Node(
                package="joy",
                executable="joy_node",
                name="joy_node",
                output="screen",
            ),
            Node(
                package="dm_dog_controller",
                executable="gamepad_command_node",
                name="dm_dog_gamepad_command_node",
                output="screen",
                parameters=[
                    {
                        "joy_topic": "joy",
                        "cmd_vel_topic": "joy_cmd",
                        "buttons_topic": "joy_buttons",
                        "linear_speed": 1.0,
                        "yaw_speed": 1.0,
                        # 摇杆死区：超过该幅度才输出，并重新线性映射到 0~1。
                        # 2026-10 由 0.12 提到 0.15：这只手柄回中后仍有漂移，
                        # 0.12 挡不住，配合 min_output 就会让狗自己走（见 mux 节点注释）。
                        "deadzone": 0.20,
                        "forward_axis": 1,
                        "lateral_axis": 0,
                        # Layout note: this robot's pad (shanwan "Android GamePad",
                        # i.e. D-input mode) reports the right stick as axis 2
                        # (left/right) and axis 3 (up/down), with LB=6 / RB=7.
                        # On a real Xbox pad in X-input mode the values would be
                        # yaw_axis=3, emergency_lb=4, emergency_rb=5.
                        "yaw_axis": 2,
                        "forward_sign": 1.0,
                        "lateral_sign": 1.0,
                        "yaw_sign": 1.0,
                        "deadman_button": -1,
                        "stop_button": 0,
                        "hold_topic": "/dog_hold",
                        "emergency_stop_topic": "/emergency_stop",
                        "emergency_lb_button": 6,
                        "emergency_rb_button": 7,
                        "emergency_a_button": 0,
                    }
                ],
            ),
            Node(
                package="dm_dog_controller",
                executable="cmd_vel_mux_node",
                name="dm_dog_cmd_vel_mux_node",
                output="screen",
                parameters=[
                    {
                        "joy_cmd_topic": "joy_cmd",
                        "aim_cmd_topic": "aim_cmd",
                        "cmd_vel_topic": "cmd_vel",
                        "publish_rate": 50.0,
                        "joy_active_timeout": 0.25,
                        "aim_timeout": 0.5,
                        "motion_epsilon": 0.0001,
                        # 最小输出塑形：小于该值的指令会被"吸"到该值（_apply_min_abs）。
                        # ⚠️ 2026-10 全部归零：原值 0.2/0.1/0.2 会把摇杆的微小漂移
                        #    （死区 0.12 之上的一点点）直接放大成 0.2 m/s 的持续前进指令，
                        #    表现为"手柄没动狗却一直往前走"。
                        #    归零后漂移只会产生 ~0.007 m/s，可忽略。
                        #    若之后发现轻推摇杆狗不动（策略自身有启动死区），再单独调大。
                        "min_output_angular_z": 0.0,
                        "min_output_linear_x": 0.0,
                        "min_output_linear_y": 0.0,
                        "rotation_blocks_translation": False,
                        "rotation_block_epsilon": 0.3,
                        "forward_blocks_lateral": True,
                        "forward_block_epsilon": 0.0001,
                        # Stick -> speed mapping.
                        #   "fixed" reproduces the old behaviour: any non-zero
                        #   forward stick means a constant +-0.5 cmd, which the
                        #   legacy policy turns into ~0.61 m/s.
                        #   "continuous" maps the stick magnitude linearly onto
                        #   0..continuous_forward_max m/s. 这是 go2_rl_gym 的
                        #   go2 PPO 速度跟踪策略（当前值）需要的模式。
                        "forward_speed_mode": "continuous",
                        "fixed_forward_speed": 0.5,
                        # 2026-10-15：前进/后退分别限幅
                        "continuous_forward_max": 0.5,     # 前进上限 m/s
                        "continuous_backward_max": 0.4,    # 后退上限 m/s（更保守）
                        "continuous_lateral_max": 0.4,
                        "continuous_yaw_max": 1.5,
                    }
                ],
            ),
            Node(
                package="dm_dog_controller",
                executable="gamepad_posture_toggle_node",
                name="dm_dog_gamepad_posture_toggle_node",
                output="screen",
                parameters=[
                    {
                        "buttons_topic": "joy_buttons",
                        "posture_service": "dog_posture",
                        "modifier_button": 6,
                        "action_button": 0,
                        "blocker_button": 7,
                        "initial_standing": False,
                    }
                ],
            ),
            # ------------------------------------------------------------------
            # 已迁移到 go2_rl_gym 的 go2 PPO 策略（单一网络，同时管前进/后退/横移/转向）。
            # 迁移前是双策略：model=policy_stair.pt + yaw_model=policy.pt。
            # 回滚方法见仓库根目录 MIGRATION_go2_rl_gym.md 第 6 节。
            #
            # 启动日志里必须看到：
            #   loaded translation_policy: ... input_dim=45 per_frame=45 history=1
            #     has_phase=False obs_layout=legacy-45 command_offset=6
            #     head_order=ang_vel,gravity,cmd history_order=newest_first
            #
            # ⚠️ 45 维 / has_phase=False 是**正常的**。
            #    launch 旧注释里的 "input_dim=47 has_phase=True" 是官方 unitree_rl_gym
            #    （带步态相位时钟）的契约；本仓库 go2_rl_gym 产出的是 45 维无相位。
            # 若显示 input_dim=225 history=5，说明 model 还指着旧文件；
            # 若报 "matches no known observation layout"，说明模型与部署侧契约对不上，先别上电。
            # ------------------------------------------------------------------
            Node(
                package="dm_dog_controller",
                executable="dm_dog_policy_node",
                name="dm_dog_policy_node",
                output="screen",
                additional_env={"LD_LIBRARY_PATH": ld_library_path},
                parameters=[
                    {
                        "startup": False,
                        # 策略轮测（保持其他参数不变，便于归因到策略本身）：
                        #   go2_cts_150k    → 已测：左右腿打架、停不下来
                        #   go2_him_21k     → 已测：更怪（髋方向翻转也试过，更差）
                        #   go2_dwaq_119.5k → 当前（最后一个可直接加载的）
                        # go2_rl_gym 能直接加载的策略到此测完。仍未试的：
                        #   go2_moe_cts_137k_0.6739 / high_slope_164k_0.6715
                        #   go2_ac_moe_cts_115k_0.6509   ← 三者都返回 tuple，
                        #   需要改 deployment.py 才能加载（见 MIGRATION 文档）
                        "model": PathJoinSubstitution(
                            [FindPackageShare("dm_dog_controller"), "policies", "go2_ppo_65000_planA.pt"]
                        ),
                        # 单一网络，不再需要 yaw_model
                        # （use_dual_policy=False 时节点不会去解析 yaw_model）。
                        "use_dual_policy": False,
                        "policy_gain_transition_duration": 0.4,
                        "policy_switch_pause_duration": 0.5,
                        "send_actions": True,
                        "gamepad_control": False,
                        "ros_command_control": True,
                        "cmd_vel_topic": "cmd_vel",
                        "buttons_topic": "joy_buttons",
                        # ⚠️ 2026-10 全部归零。原来 0.22/0.33/0.30 是"最小速度"：
                        #    _shape_axis 里 shaped = min_abs + (max_abs-min_abs)*magnitude，
                        #    所以只要 |cmd| 超过 cmd_input_deadband(0.0001)，
                        #    输出就**至少**是 min_abs。
                        #    表现为"摇杆没动狗也往前走"——摇杆漂移 0.16 经 gamepad 死区(0.15)
                        #    后剩 0.0118，经 mux 映射成 0.0071 m/s，到这里被抬到 0.2255 m/s。
                        #    注意 mux 侧还有一层 min_output_*，那里已归零；
                        #    这是第二层，必须一起改，否则堵不住。
                        # 恢复方式：想让轻推摇杆也能动，把 min_* 调到 0.1~0.2 之间，
                        #          但必须同时确认摇杆回中后 /cmd_vel 真的是 0。
                        "cmd_min_linear_x": 0.0,
                        "cmd_min_linear_y": 0.0,
                        "cmd_min_angular_z": 0.0,
                        "cmd_max_linear_x": 1.0,
                        "cmd_max_linear_y": 1.0,
                        "cmd_max_angular_z": 1.0,
                        # 保留一个极小的死区，滤掉浮点噪声；0.02 m/s 不影响正常操控
                        "cmd_input_deadband": 0.15,
                        # The node default (6) is the Xbox "Back" button, but on this
                        # pad 6 is LB - i.e. the node would exit as soon as LB was
                        # pressed for the LB+A posture toggle. Back is 10 here.
                        "quit_button": 10,
                        "posture_service": "dog_posture",
                        "emergency_stop_topic": "/emergency_stop",
                        "hold_topic": "/dog_hold",
                        "steps": 0,
                        "hz": 50.0,
                        "print_every": 0,
                        "startup_feedback_timeout": 2.0,
                        "use_startup_pose_as_rl_origin": False,
                        # ⚠️ 与 PD 增益耦合！峰值需求力矩 ≈ kp × action_clip × 0.25。
                        #    当前 kp=20 + clip=1.0 → ≈ 5 Nm（更保守）。
                        #    已试过的组合：
                        #      kp=120 + clip=4.0 → 120 Nm  实机"上电过猛"
                        #      kp=60  + clip=2.0 →  30 Nm  实机"狗身抖动严重"
                        #      kp=20  + clip=4.0 →  20 Nm  腿有自激（kd 低导致）
                        # 当前策略 go2_cts_150k 的实测动作幅值见下方注释，
                        # 2.0 基本只作为安全上限，不改变标称步态。
                        "action_clip": 2.0,
                        "hold_after_run": 5.0,
                        "shutdown_lie_down": True,
                        "shutdown_lie_down_hold": 0.25,
                    }
                ],
            )
        ]
    )
