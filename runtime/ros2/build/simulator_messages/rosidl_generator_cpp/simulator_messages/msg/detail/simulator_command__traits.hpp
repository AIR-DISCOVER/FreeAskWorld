// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from simulator_messages:msg/SimulatorCommand.idl
// generated code does not contain a copyright notice

#ifndef SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__TRAITS_HPP_
#define SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "simulator_messages/msg/detail/simulator_command__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'header'
#include "std_msgs/msg/detail/header__traits.hpp"

namespace simulator_messages
{

namespace msg
{

inline void to_flow_style_yaml(
  const SimulatorCommand & msg,
  std::ostream & out)
{
  out << "{";
  // member: header
  {
    out << "header: ";
    to_flow_style_yaml(msg.header, out);
    out << ", ";
  }

  // member: method
  {
    out << "method: ";
    rosidl_generator_traits::value_to_yaml(msg.method, out);
    out << ", ";
  }

  // member: method_params
  {
    out << "method_params: ";
    rosidl_generator_traits::value_to_yaml(msg.method_params, out);
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const SimulatorCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: header
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "header:\n";
    to_block_style_yaml(msg.header, out, indentation + 2);
  }

  // member: method
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "method: ";
    rosidl_generator_traits::value_to_yaml(msg.method, out);
    out << "\n";
  }

  // member: method_params
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    out << "method_params: ";
    rosidl_generator_traits::value_to_yaml(msg.method_params, out);
    out << "\n";
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const SimulatorCommand & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace simulator_messages

namespace rosidl_generator_traits
{

[[deprecated("use simulator_messages::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const simulator_messages::msg::SimulatorCommand & msg,
  std::ostream & out, size_t indentation = 0)
{
  simulator_messages::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use simulator_messages::msg::to_yaml() instead")]]
inline std::string to_yaml(const simulator_messages::msg::SimulatorCommand & msg)
{
  return simulator_messages::msg::to_yaml(msg);
}

template<>
inline const char * data_type<simulator_messages::msg::SimulatorCommand>()
{
  return "simulator_messages::msg::SimulatorCommand";
}

template<>
inline const char * name<simulator_messages::msg::SimulatorCommand>()
{
  return "simulator_messages/msg/SimulatorCommand";
}

template<>
struct has_fixed_size<simulator_messages::msg::SimulatorCommand>
  : std::integral_constant<bool, false> {};

template<>
struct has_bounded_size<simulator_messages::msg::SimulatorCommand>
  : std::integral_constant<bool, false> {};

template<>
struct is_message<simulator_messages::msg::SimulatorCommand>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__TRAITS_HPP_
