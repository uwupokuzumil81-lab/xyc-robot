# One validated LIVE observation. Images are preserved, not resized or compressed.
# header.stamp is the forward-camera timestamp, NOT publication time.
# Camera sampling is approximate; these fields do not imply hardware sync.
std_msgs/Header header
builtin_interfaces/Time published_at
uint32 sequence
uint32 schema_version
string schema_id
# images[i] comes from camera_topics[i] and is labelled by camera_names[i].
string[] camera_names
string[] camera_topics
sensor_msgs/Image[] images
# Fixed 18-D order: left arm seven, right arm seven, lift height, base vx/vy/wz.
string[] state_names
float64[] state
string[] state_units
builtin_interfaces/Time[] state_stamps
# Left/right source batch completion timestamps, mobile batch timestamp,
# then the two arm calibration IDs.
builtin_interfaces/Time[] arm_batch_stamps
builtin_interfaces/Time mobile_batch_stamp
string[] calibration_ids
float32 max_time_skew_s
float32 oldest_source_age_s
# Producer publishes only complete validated frames. Consumers must ALSO check age.
bool valid
