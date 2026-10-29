# dm_dog_controller

ROS2 wrapper for the `bs_code` policy deployment loop.

Build:

```bash
cd test_ws
colcon build --packages-select dm_dog_controller
source install/setup.bash
```

Default ROS2 launch. This starts `joy_node`, converts `sensor_msgs/Joy` into
robot velocity commands, then runs the policy deployment node:

```bash
ros2 launch dm_dog_controller dm_dog_policy.launch.py
```

The launch file also prepends the packaged `third_party_libs` directory to
`LD_LIBRARY_PATH` for `dm_dog_policy_node`, so the DaMiao SDK can use the bundled
newer `libstdc++.so.6` without setting it manually in the shell.

`third_party_libs` is part of this package source tree and is installed into the
package share directory by `colcon build`.

The launch defaults are:

```text
startup=true
send_actions=true
gamepad_control=false
ros_command_control=true
steps=0
hz=50
print_every=10
action_clip=1.0
hold_after_run=5.0
shutdown_lie_down=true
shutdown_lie_down_hold=0.25
```

When `shutdown_lie_down` is enabled, the node replays the startup motor command
sequence in reverse before disabling motors, so the dog returns from standing to
the initial lying pose.

Topic flow:

```text
/joy -> gamepad_command_node -> /cmd_vel
/joy -> gamepad_command_node -> /joy_buttons
/cmd_vel + /joy_buttons -> dm_dog_policy_node
/joy_buttons -> gamepad_posture_toggle_node -> /dog_posture
```

`/cmd_vel` uses `geometry_msgs/Twist`:

```text
linear.x: forward/back
linear.y: left/right
angular.z: yaw rotation
```

`/dog_posture` uses `std_srvs/SetBool`:

```text
data=true: stand up
data=false: lie down
```

The default launch starts with `startup=false`, so the dog stays lying until a
posture request is received. Press `LB + A` once to stand up, and press `LB + A`
again to lie down. The toggle repeats on each new button press.

`/joy_buttons` uses `std_msgs/Int32MultiArray` and publishes the raw button
array from the joystick message.

Run the node directly:

```bash
ros2 run dm_dog_controller dm_dog_policy_node
```

Run only the joystick converter:

```bash
ros2 run dm_dog_controller gamepad_command_node
```

Run only the joystick posture toggle:

```bash
ros2 run dm_dog_controller gamepad_posture_toggle_node
```

Request posture manually:

```bash
ros2 service call /dog_posture std_srvs/srv/SetBool "{data: true}"
ros2 service call /dog_posture std_srvs/srv/SetBool "{data: false}"
```

Check motor feedback without running startup:

```bash
ros2 run dm_dog_controller motor_feedback_check_node
```

If this reports all 12 motors missing, do not bypass feedback for startup.
Check motor power, CAN wiring, adapter channel order, MIT mode, and motor IDs
first.

Override parameters with ROS arguments:

```bash
ros2 run dm_dog_controller dm_dog_policy_node --ros-args \
  -p action_clip:=0.5 \
  -p command_timeout:=0.5
```

Hardware-free smoke test:

```bash
ros2 run dm_dog_controller dm_dog_policy_node --ros-args \
  -p use_imu:=false \
  -p use_can:=false \
  -p startup:=false \
  -p send_actions:=false \
  -p gamepad_control:=false \
  -p steps:=1
```

The ROS2 Python environment must be able to import `torch` and `numpy`.
If PyTorch is only installed in the `leggym` conda environment, activate that
environment before sourcing the ROS2 workspace.
