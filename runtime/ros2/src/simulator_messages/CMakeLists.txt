cmake_minimum_required(VERSION 3.5)
project(simulator_messages)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SimulatorCommand.msg"
  DEPENDENCIES std_msgs
)

ament_package()
