cmake_minimum_required(VERSION 3.16)
project(xyc_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MotorFeedback.msg"
  "msg/MotorFeedbackArray.msg"
  "msg/ArmState.msg"
  "msg/LeaderState.msg"
  "msg/MobileState.msg"
  "msg/MobileCommand.msg"
  "msg/Observation.msg"
  "srv/ArmBusTransaction.srv"
  "srv/SetArmTargets.srv"
  "srv/RecordingControl.srv"
  DEPENDENCIES std_msgs builtin_interfaces sensor_msgs
)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
