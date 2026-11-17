# XYC hardware

This package is part of the complete project described in `../../../README.md`.
Run the PC client and ROS backend using the root README. There is one daily
teleoperation path; old per-joint keyboard clients and wrist-step tools have been removed.

## Runtime path

```text
PC tools/teleop_stream.py
  USB leaders + local keyboard
             |
       SSH TCP forward
             |
  teleop_nodes.LeaderStateBridge
       |                |
  LeaderState       chassis_bridge.PCChassisBridge
       |                |
  TeleopController   base/lift TwistStamped
       |                |
  ArmGroupCoordinator   |
       |                |
  left/right MotorStateNode (one owner per serial port)
       |
  arm_joint_control / chassis_control -> servo bus
```

`teleop.launch.py with_motors:=true` includes `motors.launch.py` internally.
Do not start a second motor launch on the same physical serial ports.
The motors launch also supports calibrated, read-only feedback for diagnostics.
Calibration remains a separate maintenance tool because it needs exclusive serial access.

`JointControl` has one workflow: explicitly hold the measured startup position,
accept absolute targets, and keep torque enabled after arrival. Relative jogs,
fixed-duration holds with automatic release, and wrist-only test modes are removed
from the implementation. Explicit release, shutdown and fault handling share the
same verified torque-off path.
Chassis preparation reports a distinct not-ready result while waiting for fresh,
stationary feedback. The PC bridge retries that result within its existing deadline;
hardware enable failures are reported without repeating the enable writes.

## Source modules

| Module | Responsibility |
| --- | --- |
| `feetech_reader.py` | Packet parsing, model checks and raw feedback |
| `arm_calibration.py` | Follower calibration file recording and validation |
| `arm_state.py` | Calibrated per-arm state and validity |
| `arm_joint_control.py` | Shared motor transport, persistent targets, ownership and protection |
| `motor_state_node.py` | Exclusive bus owner, state publication, grouped arm transactions |
| `arm_group_coordinator.py` | 14-target validation and coordination of the two bus owners |
| `teleop_transport.py` | PC/Pi wire format, frame validation and timing constants |
| `teleop_control.py` | Following target calculation and source validation |
| `teleop_nodes.py` | Leader receiver and ROS following controller |
| `chassis_bridge.py` | PC keyboard intent to ROS topics; no local keyboard UI |
| `base_kinematics.py` | Three-wheel velocity conversion |
| `lift_tracking.py` | Multi-turn encoder tracking and lift height conversion |
| `chassis_control.py` | Base/lift speed-mode control, homing and watchdogs |
| `command_contract.py`, `arm_batch_planner.py` | Reusable command validation for later integrations |

The base and lift driver runs inside the existing left bus owner. It does not
open a second serial connection. Arm joints use position targets; wheels use
velocity mode; lift jogging uses velocity mode and homing establishes a height reference.

## Interfaces

- `/motors/left/feedback`, `/motors/right/feedback`: raw motor feedback.
- `/motors/left/state`, `/motors/right/state`: seven calibrated positions each.
- `/teleop/left/leader_state`, `/teleop/right/leader_state`: PC leader feedback.
- `/robot/arms/start`: validate and hold all fourteen current arm positions.
- `/robot/arms/set_targets`: one ordered fourteen-position `SetArmTargets` request.
- `/motors/{left,right}/group/transaction`: internal prepare/commit endpoint.
- `/motors/{left,right}/joint/status`: read-only ownership and fault diagnostics.
- `/motors/{left,right}/joint/torque_status`: read actual torque registers.
- `/robot/base/cmd_vel`: base-frame x/y/yaw `TwistStamped`.
- `/robot/lift/cmd_vel`: base-frame vertical speed `TwistStamped`.
- `/robot/chassis/enable`, `/robot/chassis/stop`, `/robot/chassis/home_lift`: chassis services.
- `/robot/chassis/status`: base/lift readiness, homing, height and fault status.
- `/robot/chassis/state`: timestamped measured lift height and base `vx/vy/wz` for the unified 18-D observation; this read-only publisher does not enable motors.

Arms are ordered left seven then right seven. Each arm is shoulder_pan,
shoulder_lift, elbow_flex, wrist_flex, wrist_yaw, wrist_roll, gripper.
Positions use the recorded calibration scale: arm joints [-100, 100], grippers [0, 100].
These are not degrees. Group requests include source identity, sequence,
lifetime and both calibration IDs. Transactions coordinate independent serial buses;
they do not guarantee physically simultaneous movement.

## Hardware configuration

`config/motors.yaml` maps USB serial identities, IDs, model numbers and read rates.
Left bus IDs 1..7 are the arm, 8..10 are the wheels and 11 is the lift.
Right bus IDs 1..7 are the other arm. Model numbers 2825 and 2569 map to
STS3250 and STS3095. Serial bus settings are 1000000 baud, 8N1.

The main launch has no `pilot_joint`, `keyboard_mode` or extended wrist test mode.
It does not expose the old jog/set-position single-joint services.
Low-level joint control remains a shared implementation required by grouped control.
Normal targets retain torque; current, temperature, lost feedback and shutdown protections remain.
One isolated invalid arm-feedback frame pauses new targets but keeps the existing goal.  A joint
releases only when valid feedback remains missing for 0.20 s; teleoperation waits up to 0.50 s
before stopping its controller.  Out-of-range numeric feedback still stops immediately; a servo
alarm that makes feedback invalid must remain present through the same 0.20 s feedback-loss window.
Restarting a driver can release supported arm joints. Confirm physical support before switching.

See the root README for current startup commands, calibration assets, environment
files and the isolated fake-hardware regression command. Passing these tests does
not validate every load, joint range, lift mechanical stop or base motion direction.
