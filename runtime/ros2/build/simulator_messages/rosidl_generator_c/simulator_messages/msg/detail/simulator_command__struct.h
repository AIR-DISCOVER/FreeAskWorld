// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from simulator_messages:msg/SimulatorCommand.idl
// generated code does not contain a copyright notice

#ifndef SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__STRUCT_H_
#define SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>


// Constants defined in the message

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.h"
// Member 'method'
// Member 'method_params'
#include "rosidl_runtime_c/string.h"

/// Struct defined in msg/SimulatorCommand in the package simulator_messages.
/**
  * Header
 */
typedef struct simulator_messages__msg__SimulatorCommand
{
  std_msgs__msg__Header header;
  /// 调用接收端的函数方法名
  rosidl_runtime_c__String method;
  rosidl_runtime_c__String method_params;
} simulator_messages__msg__SimulatorCommand;

// Struct for a sequence of simulator_messages__msg__SimulatorCommand.
typedef struct simulator_messages__msg__SimulatorCommand__Sequence
{
  simulator_messages__msg__SimulatorCommand * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} simulator_messages__msg__SimulatorCommand__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__STRUCT_H_
