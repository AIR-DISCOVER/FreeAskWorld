// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from simulator_messages:msg/SimulatorCommand.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "simulator_messages/msg/detail/simulator_command__rosidl_typesupport_introspection_c.h"
#include "simulator_messages/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "simulator_messages/msg/detail/simulator_command__functions.h"
#include "simulator_messages/msg/detail/simulator_command__struct.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/header.h"
// Member `header`
#include "std_msgs/msg/detail/header__rosidl_typesupport_introspection_c.h"
// Member `method`
// Member `method_params`
#include "rosidl_runtime_c/string_functions.h"

#ifdef __cplusplus
extern "C"
{
#endif

void simulator_messages__msg__SimulatorCommand__rosidl_typesupport_introspection_c__SimulatorCommand_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  simulator_messages__msg__SimulatorCommand__init(message_memory);
}

void simulator_messages__msg__SimulatorCommand__rosidl_typesupport_introspection_c__SimulatorCommand_fini_function(void * message_memory)
{
  simulator_messages__msg__SimulatorCommand__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember simulator_messages__msg__SimulatorCommand__rosidl_typesupport_introspection_c__SimulatorCommand_message_member_array[3] = {
  {
    "header",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(simulator_messages__msg__SimulatorCommand, header),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "method",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(simulator_messages__msg__SimulatorCommand, method),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "method_params",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_STRING,  // type
    0,  // upper bound of string
    NULL,  // members of sub message
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(simulator_messages__msg__SimulatorCommand, method_params),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL,  // fetch(index, &value) function pointer
    NULL,  // assign(index, value) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers simulator_messages__msg__SimulatorCommand__rosidl_typesupport_introspection_c__SimulatorCommand_message_members = {
  "simulator_messages__msg",  // message namespace
  "SimulatorCommand",  // message name
  3,  // number of fields
  sizeof(simulator_messages__msg__SimulatorCommand),
  simulator_messages__msg__SimulatorCommand__rosidl_typesupport_introspection_c__SimulatorCommand_message_member_array,  // message members
  simulator_messages__msg__SimulatorCommand__rosidl_typesupport_introspection_c__SimulatorCommand_init_function,  // function to initialize message memory (memory has to be allocated)
  simulator_messages__msg__SimulatorCommand__rosidl_typesupport_introspection_c__SimulatorCommand_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t simulator_messages__msg__SimulatorCommand__rosidl_typesupport_introspection_c__SimulatorCommand_message_type_support_handle = {
  0,
  &simulator_messages__msg__SimulatorCommand__rosidl_typesupport_introspection_c__SimulatorCommand_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_simulator_messages
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, simulator_messages, msg, SimulatorCommand)() {
  simulator_messages__msg__SimulatorCommand__rosidl_typesupport_introspection_c__SimulatorCommand_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, std_msgs, msg, Header)();
  if (!simulator_messages__msg__SimulatorCommand__rosidl_typesupport_introspection_c__SimulatorCommand_message_type_support_handle.typesupport_identifier) {
    simulator_messages__msg__SimulatorCommand__rosidl_typesupport_introspection_c__SimulatorCommand_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &simulator_messages__msg__SimulatorCommand__rosidl_typesupport_introspection_c__SimulatorCommand_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
