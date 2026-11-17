# Host receipt time for this reply. This is not a hardware sampling timestamp.
std_msgs/Header header
string name
string bus
uint8 motor_id
uint16 model_number

# Numeric feedback is usable only when valid is true.
# Invalid samples carry zero integer fields and NaN floating-point fields.
bool valid
uint8 servo_error
string error

# STS encoder counts and signed raw speed/load; not radians, rad/s, or Nm.
int32 position_ticks
int32 velocity_raw
int16 load_raw
float32 voltage_v
float32 temperature_c
float32 current_a
bool moving
