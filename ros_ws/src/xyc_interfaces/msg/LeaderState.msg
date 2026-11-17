# One leader arm, in the same joint order and calibration units as ArmState.
# The Pi stamps the conservative sample time using the transmitted sample age.
std_msgs/Header header
string arm
string session_id
uint64 sample_id
string calibration_id
string[7] names
float64[7] position_normalized
