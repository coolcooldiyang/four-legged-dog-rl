cmake_minimum_required(VERSION 3.8)
project(dm_dog_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(dm_dog_controller)
ament_python_install_package(bs_hardware)
ament_python_install_package(lib)

install(PROGRAMS
  dm_dog_controller/dog_policy_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME dm_dog_policy_node
)

install(PROGRAMS
  dm_dog_controller/gamepad_command_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME gamepad_command_node
)

install(PROGRAMS
  dm_dog_controller/cmd_vel_mux_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cmd_vel_mux_node
)

install(PROGRAMS
  dm_dog_controller/gamepad_posture_toggle_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME gamepad_posture_toggle_node
)

install(PROGRAMS
  dm_dog_controller/motor_feedback_check_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME motor_feedback_check_node
)

install(FILES
  policy.pt
  policy_m.pt
  policy_move.pt
  policy_old.pt
  policy_stair.pt
  DESTINATION share/${PROJECT_NAME}
)

# go2_rl_gym（go2 PPO）迁移过来的策略。
# 做成目录是为了方便放多个候选 checkpoint，launch 里用 policies/<文件名> 引用。
install(DIRECTORY
  policies
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  lib/libdm_device.so
  DESTINATION share/${PROJECT_NAME}/lib
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  third_party_libs
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
