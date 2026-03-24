// generated from rosidl_generator_c/resource/idl__functions.c.em
// with input from simulator_messages:msg/SimulatorCommand.idl
// generated code does not contain a copyright notice
#include "simulator_messages/msg/detail/simulator_command__functions.h"

#include <assert.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "rcutils/allocator.h"


// Include directives for member types
// Member `header`
#include "std_msgs/msg/detail/header__functions.h"
// Member `method`
// Member `method_params`
#include "rosidl_runtime_c/string_functions.h"

bool
simulator_messages__msg__SimulatorCommand__init(simulator_messages__msg__SimulatorCommand * msg)
{
  if (!msg) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__init(&msg->header)) {
    simulator_messages__msg__SimulatorCommand__fini(msg);
    return false;
  }
  // method
  if (!rosidl_runtime_c__String__init(&msg->method)) {
    simulator_messages__msg__SimulatorCommand__fini(msg);
    return false;
  }
  // method_params
  if (!rosidl_runtime_c__String__init(&msg->method_params)) {
    simulator_messages__msg__SimulatorCommand__fini(msg);
    return false;
  }
  return true;
}

void
simulator_messages__msg__SimulatorCommand__fini(simulator_messages__msg__SimulatorCommand * msg)
{
  if (!msg) {
    return;
  }
  // header
  std_msgs__msg__Header__fini(&msg->header);
  // method
  rosidl_runtime_c__String__fini(&msg->method);
  // method_params
  rosidl_runtime_c__String__fini(&msg->method_params);
}

bool
simulator_messages__msg__SimulatorCommand__are_equal(const simulator_messages__msg__SimulatorCommand * lhs, const simulator_messages__msg__SimulatorCommand * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__are_equal(
      &(lhs->header), &(rhs->header)))
  {
    return false;
  }
  // method
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->method), &(rhs->method)))
  {
    return false;
  }
  // method_params
  if (!rosidl_runtime_c__String__are_equal(
      &(lhs->method_params), &(rhs->method_params)))
  {
    return false;
  }
  return true;
}

bool
simulator_messages__msg__SimulatorCommand__copy(
  const simulator_messages__msg__SimulatorCommand * input,
  simulator_messages__msg__SimulatorCommand * output)
{
  if (!input || !output) {
    return false;
  }
  // header
  if (!std_msgs__msg__Header__copy(
      &(input->header), &(output->header)))
  {
    return false;
  }
  // method
  if (!rosidl_runtime_c__String__copy(
      &(input->method), &(output->method)))
  {
    return false;
  }
  // method_params
  if (!rosidl_runtime_c__String__copy(
      &(input->method_params), &(output->method_params)))
  {
    return false;
  }
  return true;
}

simulator_messages__msg__SimulatorCommand *
simulator_messages__msg__SimulatorCommand__create()
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  simulator_messages__msg__SimulatorCommand * msg = (simulator_messages__msg__SimulatorCommand *)allocator.allocate(sizeof(simulator_messages__msg__SimulatorCommand), allocator.state);
  if (!msg) {
    return NULL;
  }
  memset(msg, 0, sizeof(simulator_messages__msg__SimulatorCommand));
  bool success = simulator_messages__msg__SimulatorCommand__init(msg);
  if (!success) {
    allocator.deallocate(msg, allocator.state);
    return NULL;
  }
  return msg;
}

void
simulator_messages__msg__SimulatorCommand__destroy(simulator_messages__msg__SimulatorCommand * msg)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (msg) {
    simulator_messages__msg__SimulatorCommand__fini(msg);
  }
  allocator.deallocate(msg, allocator.state);
}


bool
simulator_messages__msg__SimulatorCommand__Sequence__init(simulator_messages__msg__SimulatorCommand__Sequence * array, size_t size)
{
  if (!array) {
    return false;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  simulator_messages__msg__SimulatorCommand * data = NULL;

  if (size) {
    data = (simulator_messages__msg__SimulatorCommand *)allocator.zero_allocate(size, sizeof(simulator_messages__msg__SimulatorCommand), allocator.state);
    if (!data) {
      return false;
    }
    // initialize all array elements
    size_t i;
    for (i = 0; i < size; ++i) {
      bool success = simulator_messages__msg__SimulatorCommand__init(&data[i]);
      if (!success) {
        break;
      }
    }
    if (i < size) {
      // if initialization failed finalize the already initialized array elements
      for (; i > 0; --i) {
        simulator_messages__msg__SimulatorCommand__fini(&data[i - 1]);
      }
      allocator.deallocate(data, allocator.state);
      return false;
    }
  }
  array->data = data;
  array->size = size;
  array->capacity = size;
  return true;
}

void
simulator_messages__msg__SimulatorCommand__Sequence__fini(simulator_messages__msg__SimulatorCommand__Sequence * array)
{
  if (!array) {
    return;
  }
  rcutils_allocator_t allocator = rcutils_get_default_allocator();

  if (array->data) {
    // ensure that data and capacity values are consistent
    assert(array->capacity > 0);
    // finalize all array elements
    for (size_t i = 0; i < array->capacity; ++i) {
      simulator_messages__msg__SimulatorCommand__fini(&array->data[i]);
    }
    allocator.deallocate(array->data, allocator.state);
    array->data = NULL;
    array->size = 0;
    array->capacity = 0;
  } else {
    // ensure that data, size, and capacity values are consistent
    assert(0 == array->size);
    assert(0 == array->capacity);
  }
}

simulator_messages__msg__SimulatorCommand__Sequence *
simulator_messages__msg__SimulatorCommand__Sequence__create(size_t size)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  simulator_messages__msg__SimulatorCommand__Sequence * array = (simulator_messages__msg__SimulatorCommand__Sequence *)allocator.allocate(sizeof(simulator_messages__msg__SimulatorCommand__Sequence), allocator.state);
  if (!array) {
    return NULL;
  }
  bool success = simulator_messages__msg__SimulatorCommand__Sequence__init(array, size);
  if (!success) {
    allocator.deallocate(array, allocator.state);
    return NULL;
  }
  return array;
}

void
simulator_messages__msg__SimulatorCommand__Sequence__destroy(simulator_messages__msg__SimulatorCommand__Sequence * array)
{
  rcutils_allocator_t allocator = rcutils_get_default_allocator();
  if (array) {
    simulator_messages__msg__SimulatorCommand__Sequence__fini(array);
  }
  allocator.deallocate(array, allocator.state);
}

bool
simulator_messages__msg__SimulatorCommand__Sequence__are_equal(const simulator_messages__msg__SimulatorCommand__Sequence * lhs, const simulator_messages__msg__SimulatorCommand__Sequence * rhs)
{
  if (!lhs || !rhs) {
    return false;
  }
  if (lhs->size != rhs->size) {
    return false;
  }
  for (size_t i = 0; i < lhs->size; ++i) {
    if (!simulator_messages__msg__SimulatorCommand__are_equal(&(lhs->data[i]), &(rhs->data[i]))) {
      return false;
    }
  }
  return true;
}

bool
simulator_messages__msg__SimulatorCommand__Sequence__copy(
  const simulator_messages__msg__SimulatorCommand__Sequence * input,
  simulator_messages__msg__SimulatorCommand__Sequence * output)
{
  if (!input || !output) {
    return false;
  }
  if (output->capacity < input->size) {
    const size_t allocation_size =
      input->size * sizeof(simulator_messages__msg__SimulatorCommand);
    rcutils_allocator_t allocator = rcutils_get_default_allocator();
    simulator_messages__msg__SimulatorCommand * data =
      (simulator_messages__msg__SimulatorCommand *)allocator.reallocate(
      output->data, allocation_size, allocator.state);
    if (!data) {
      return false;
    }
    // If reallocation succeeded, memory may or may not have been moved
    // to fulfill the allocation request, invalidating output->data.
    output->data = data;
    for (size_t i = output->capacity; i < input->size; ++i) {
      if (!simulator_messages__msg__SimulatorCommand__init(&output->data[i])) {
        // If initialization of any new item fails, roll back
        // all previously initialized items. Existing items
        // in output are to be left unmodified.
        for (; i-- > output->capacity; ) {
          simulator_messages__msg__SimulatorCommand__fini(&output->data[i]);
        }
        return false;
      }
    }
    output->capacity = input->size;
  }
  output->size = input->size;
  for (size_t i = 0; i < input->size; ++i) {
    if (!simulator_messages__msg__SimulatorCommand__copy(
        &(input->data[i]), &(output->data[i])))
    {
      return false;
    }
  }
  return true;
}
