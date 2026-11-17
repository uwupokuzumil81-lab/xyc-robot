# Command contract v3. This message alone does not control hardware.
# Values describe one complete desired command, in fixed order.
std_msgs/Header header
string source_id
uint64 sequence
string schema_id

# The producer's left/right calibration IDs, matching the current robot assets.
string[2] arm_calibration_ids
# Identity of the verified lift reference for this session.
string lift_reference_id

# Left six joints + left gripper, then right six joints + right gripper.
# The 12 arm joints use the current project's [-100,100] calibration coordinates;
# the two grippers use [0,100]. This is not radians and is not a LeRobot conversion.
float64[14] arm_position_normalized

# Strict direction commands: each value must be exactly -1, 0 or +1.
# +x is forward, +y is left and +yaw is counterclockwise.
float64 lift_command
float64 base_x_command
float64 base_y_command
float64 base_yaw_command

# Maximum allowed command age, evaluated against a common ROS clock.
builtin_interfaces/Duration valid_for
