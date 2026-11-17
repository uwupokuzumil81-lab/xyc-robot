# Timestamped measured state for the lift and three-degree-of-freedom base.
# This message is read-only feedback; it never represents a command.
std_msgs/Header header

# Fixed canonical order:
# lift.height_m, base.vx_m_s, base.vy_m_s, base.wz_rad_s
string[] names
float64[] state
string[] units

# valid=false means consumers must reject this sample rather than reuse values.
bool valid
bool enabled
bool homing
string error
