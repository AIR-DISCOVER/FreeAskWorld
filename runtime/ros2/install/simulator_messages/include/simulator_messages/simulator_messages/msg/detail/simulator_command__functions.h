// generated from rosidl_generator_c/resource/idl__functions.h.em
// with input from simulator_messages:msg/SimulatorCommand.idl
// generated code does not contain a copyright notice

#ifndef SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__FUNCTIONS_H_
#define SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__FUNCTIONS_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stdlib.h>

#include "rosidl_runtime_c/visibility_control.h"
#include "simulator_messages/msg/rosidl_generator_c__visibility_control.h"

#include "simulator_messages/msg/detail/simulator_command__struct.h"

/// Initialize msg/SimulatorCommand message.
/**
 * If the init function is called twice for the same message without
 * calling fini inbetween previously allocated memory will be leaked.
 * \param[in,out] msg The previously allocated message pointer.
 * Fields without a default value will not be initialized by this function.
 * You might want to call memset(msg, 0, sizeof(
 * simulator_messages__msg__SimulatorCommand
 * )) before or use
 * simulator_messages__msg__SimulatorCommand__create()
 * to allocate and initialize the message.
 * \return true if initialization was successful, otherwise false
 */
ROSIDL_GENERATOR_C_PUBLIC_simulator_messages
bool
simulator_messages__msg__SimulatorCommand__init(simulator_messages__msg__SimulatorCommand * msg);

/// Finalize msg/SimulatorCommand message.
/**
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_simulator_messages
void
simulator_messages__msg__SimulatorCommand__fini(simulator_messages__msg__SimulatorCommand * msg);

/// Create msg/SimulatorCommand message.
/**
 * It allocates the memory for the message, sets the memory to zero, and
 * calls
 * simulator_messages__msg__SimulatorCommand__init().
 * \return The pointer to the initialized message if successful,
 * otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_simulator_messages
simulator_messages__msg__SimulatorCommand *
simulator_messages__msg__SimulatorCommand__create();

/// Destroy msg/SimulatorCommand message.
/**
 * It calls
 * simulator_messages__msg__SimulatorCommand__fini()
 * and frees the memory of the message.
 * \param[in,out] msg The allocated message pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_simulator_messages
void
simulator_messages__msg__SimulatorCommand__destroy(simulator_messages__msg__SimulatorCommand * msg);

/// Check for msg/SimulatorCommand message equality.
/**
 * \param[in] lhs The message on the left hand size of the equality operator.
 * \param[in] rhs The message on the right hand size of the equality operator.
 * \return true if messages are equal, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_simulator_messages
bool
simulator_messages__msg__SimulatorCommand__are_equal(const simulator_messages__msg__SimulatorCommand * lhs, const simulator_messages__msg__SimulatorCommand * rhs);

/// Copy a msg/SimulatorCommand message.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source message pointer.
 * \param[out] output The target message pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer is null
 *   or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_simulator_messages
bool
simulator_messages__msg__SimulatorCommand__copy(
  const simulator_messages__msg__SimulatorCommand * input,
  simulator_messages__msg__SimulatorCommand * output);

/// Initialize array of msg/SimulatorCommand messages.
/**
 * It allocates the memory for the number of elements and calls
 * simulator_messages__msg__SimulatorCommand__init()
 * for each element of the array.
 * \param[in,out] array The allocated array pointer.
 * \param[in] size The size / capacity of the array.
 * \return true if initialization was successful, otherwise false
 * If the array pointer is valid and the size is zero it is guaranteed
 # to return true.
 */
ROSIDL_GENERATOR_C_PUBLIC_simulator_messages
bool
simulator_messages__msg__SimulatorCommand__Sequence__init(simulator_messages__msg__SimulatorCommand__Sequence * array, size_t size);

/// Finalize array of msg/SimulatorCommand messages.
/**
 * It calls
 * simulator_messages__msg__SimulatorCommand__fini()
 * for each element of the array and frees the memory for the number of
 * elements.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_simulator_messages
void
simulator_messages__msg__SimulatorCommand__Sequence__fini(simulator_messages__msg__SimulatorCommand__Sequence * array);

/// Create array of msg/SimulatorCommand messages.
/**
 * It allocates the memory for the array and calls
 * simulator_messages__msg__SimulatorCommand__Sequence__init().
 * \param[in] size The size / capacity of the array.
 * \return The pointer to the initialized array if successful, otherwise NULL
 */
ROSIDL_GENERATOR_C_PUBLIC_simulator_messages
simulator_messages__msg__SimulatorCommand__Sequence *
simulator_messages__msg__SimulatorCommand__Sequence__create(size_t size);

/// Destroy array of msg/SimulatorCommand messages.
/**
 * It calls
 * simulator_messages__msg__SimulatorCommand__Sequence__fini()
 * on the array,
 * and frees the memory of the array.
 * \param[in,out] array The initialized array pointer.
 */
ROSIDL_GENERATOR_C_PUBLIC_simulator_messages
void
simulator_messages__msg__SimulatorCommand__Sequence__destroy(simulator_messages__msg__SimulatorCommand__Sequence * array);

/// Check for msg/SimulatorCommand message array equality.
/**
 * \param[in] lhs The message array on the left hand size of the equality operator.
 * \param[in] rhs The message array on the right hand size of the equality operator.
 * \return true if message arrays are equal in size and content, otherwise false.
 */
ROSIDL_GENERATOR_C_PUBLIC_simulator_messages
bool
simulator_messages__msg__SimulatorCommand__Sequence__are_equal(const simulator_messages__msg__SimulatorCommand__Sequence * lhs, const simulator_messages__msg__SimulatorCommand__Sequence * rhs);

/// Copy an array of msg/SimulatorCommand messages.
/**
 * This functions performs a deep copy, as opposed to the shallow copy that
 * plain assignment yields.
 *
 * \param[in] input The source array pointer.
 * \param[out] output The target array pointer, which must
 *   have been initialized before calling this function.
 * \return true if successful, or false if either pointer
 *   is null or memory allocation fails.
 */
ROSIDL_GENERATOR_C_PUBLIC_simulator_messages
bool
simulator_messages__msg__SimulatorCommand__Sequence__copy(
  const simulator_messages__msg__SimulatorCommand__Sequence * input,
  simulator_messages__msg__SimulatorCommand__Sequence * output);

#ifdef __cplusplus
}
#endif

#endif  // SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__FUNCTIONS_H_
