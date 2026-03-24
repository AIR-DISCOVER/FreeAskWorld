// generated from rosidl_generator_cpp/resource/idl__struct.hpp.em
// with input from simulator_messages:msg/SimulatorCommand.idl
// generated code does not contain a copyright notice

#ifndef SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__STRUCT_HPP_
#define SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__STRUCT_HPP_

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rosidl_runtime_cpp/bounded_vector.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__struct.hpp"

#ifndef _WIN32
# define DEPRECATED__simulator_messages__msg__SimulatorCommand __attribute__((deprecated))
#else
# define DEPRECATED__simulator_messages__msg__SimulatorCommand __declspec(deprecated)
#endif

namespace simulator_messages
{

namespace msg
{

// message struct
template<class ContainerAllocator>
struct SimulatorCommand_
{
  using Type = SimulatorCommand_<ContainerAllocator>;

  explicit SimulatorCommand_(rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_init)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->method = "";
      this->method_params = "";
    }
  }

  explicit SimulatorCommand_(const ContainerAllocator & _alloc, rosidl_runtime_cpp::MessageInitialization _init = rosidl_runtime_cpp::MessageInitialization::ALL)
  : header(_alloc, _init),
    method(_alloc),
    method_params(_alloc)
  {
    if (rosidl_runtime_cpp::MessageInitialization::ALL == _init ||
      rosidl_runtime_cpp::MessageInitialization::ZERO == _init)
    {
      this->method = "";
      this->method_params = "";
    }
  }

  // field types and members
  using _header_type =
    std_msgs::msg::Header_<ContainerAllocator>;
  _header_type header;
  using _method_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _method_type method;
  using _method_params_type =
    std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>>;
  _method_params_type method_params;

  // setters for named parameter idiom
  Type & set__header(
    const std_msgs::msg::Header_<ContainerAllocator> & _arg)
  {
    this->header = _arg;
    return *this;
  }
  Type & set__method(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->method = _arg;
    return *this;
  }
  Type & set__method_params(
    const std::basic_string<char, std::char_traits<char>, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<char>> & _arg)
  {
    this->method_params = _arg;
    return *this;
  }

  // constant declarations

  // pointer types
  using RawPtr =
    simulator_messages::msg::SimulatorCommand_<ContainerAllocator> *;
  using ConstRawPtr =
    const simulator_messages::msg::SimulatorCommand_<ContainerAllocator> *;
  using SharedPtr =
    std::shared_ptr<simulator_messages::msg::SimulatorCommand_<ContainerAllocator>>;
  using ConstSharedPtr =
    std::shared_ptr<simulator_messages::msg::SimulatorCommand_<ContainerAllocator> const>;

  template<typename Deleter = std::default_delete<
      simulator_messages::msg::SimulatorCommand_<ContainerAllocator>>>
  using UniquePtrWithDeleter =
    std::unique_ptr<simulator_messages::msg::SimulatorCommand_<ContainerAllocator>, Deleter>;

  using UniquePtr = UniquePtrWithDeleter<>;

  template<typename Deleter = std::default_delete<
      simulator_messages::msg::SimulatorCommand_<ContainerAllocator>>>
  using ConstUniquePtrWithDeleter =
    std::unique_ptr<simulator_messages::msg::SimulatorCommand_<ContainerAllocator> const, Deleter>;
  using ConstUniquePtr = ConstUniquePtrWithDeleter<>;

  using WeakPtr =
    std::weak_ptr<simulator_messages::msg::SimulatorCommand_<ContainerAllocator>>;
  using ConstWeakPtr =
    std::weak_ptr<simulator_messages::msg::SimulatorCommand_<ContainerAllocator> const>;

  // pointer types similar to ROS 1, use SharedPtr / ConstSharedPtr instead
  // NOTE: Can't use 'using' here because GNU C++ can't parse attributes properly
  typedef DEPRECATED__simulator_messages__msg__SimulatorCommand
    std::shared_ptr<simulator_messages::msg::SimulatorCommand_<ContainerAllocator>>
    Ptr;
  typedef DEPRECATED__simulator_messages__msg__SimulatorCommand
    std::shared_ptr<simulator_messages::msg::SimulatorCommand_<ContainerAllocator> const>
    ConstPtr;

  // comparison operators
  bool operator==(const SimulatorCommand_ & other) const
  {
    if (this->header != other.header) {
      return false;
    }
    if (this->method != other.method) {
      return false;
    }
    if (this->method_params != other.method_params) {
      return false;
    }
    return true;
  }
  bool operator!=(const SimulatorCommand_ & other) const
  {
    return !this->operator==(other);
  }
};  // struct SimulatorCommand_

// alias to use template instance with default allocator
using SimulatorCommand =
  simulator_messages::msg::SimulatorCommand_<std::allocator<void>>;

// constant definitions

}  // namespace msg

}  // namespace simulator_messages

#endif  // SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__STRUCT_HPP_
