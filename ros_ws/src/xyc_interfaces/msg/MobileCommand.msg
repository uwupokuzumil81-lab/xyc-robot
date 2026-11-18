# One atomic lift/base intent. All timestamps use the robot ROS clock.
std_msgs/Header header
builtin_interfaces/Time expires_at
# lift, base-x, base-y, base-yaw; each value must be -1, 0 or +1.
int8[4] commands
