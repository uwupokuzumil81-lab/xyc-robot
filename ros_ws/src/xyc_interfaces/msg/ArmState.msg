# Calibrated model coordinates for ONE arm, including its gripper.
# Six arm joints: [-100, 100]. Gripper: [0, 100]. These are NOT radians/metres.
# All arrays have seven entries, ordered by names; invalid values are NaN.
# Header stamp is the raw batch completion time, not a synchronized sample time.
std_msgs/Header header
string arm
string calibration_id
string[] names
float64[] position_normalized
bool[] valid
string[] errors
# Per-motor host receipt times; zero when no valid sample is available.
builtin_interfaces/Time[] sample_stamps
bool complete
