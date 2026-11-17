# Host completion time of this bus read. Individual receipt times are below.
std_msgs/Header header
uint32 sequence
float32 read_duration_s
# Configured order, including explicit invalid entries for missing replies.
MotorFeedback[] motors
