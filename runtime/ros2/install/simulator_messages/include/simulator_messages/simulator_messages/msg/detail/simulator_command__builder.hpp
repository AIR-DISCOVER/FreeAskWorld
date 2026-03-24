// generated from rosidl_generator_cpp/resource/idl__builder.hpp.em
// with input from simulator_messages:msg/SimulatorCommand.idl
// generated code does not contain a copyright notice

#ifndef SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__BUILDER_HPP_
#define SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__BUILDER_HPP_

#include <algorithm>
#include <utility>

#include "simulator_messages/msg/detail/simulator_command__struct.hpp"
#include "rosidl_runtime_cpp/message_initialization.hpp"


namespace simulator_messages
{

namespace msg
{

namespace builder
{

class Init_SimulatorCommand_method_params
{
public:
  explicit Init_SimulatorCommand_method_params(::simulator_messages::msg::SimulatorCommand & msg)
  : msg_(msg)
  {}
  ::simulator_messages::msg::SimulatorCommand method_params(::simulator_messages::msg::SimulatorCommand::_method_params_type arg)
  {
    msg_.method_params = std::move(arg);
    return std::move(msg_);
  }

private:
  ::simulator_messages::msg::SimulatorCommand msg_;
};

class Init_SimulatorCommand_method
{
public:
  explicit Init_SimulatorCommand_method(::simulator_messages::msg::SimulatorCommand & msg)
  : msg_(msg)
  {}
  Init_SimulatorCommand_method_params method(::simulator_messages::msg::SimulatorCommand::_method_type arg)
  {
    msg_.method = std::move(arg);
    return Init_SimulatorCommand_method_params(msg_);
  }

private:
  ::simulator_messages::msg::SimulatorCommand msg_;
};

class Init_SimulatorCommand_header
{
public:
  Init_SimulatorCommand_header()
  : msg_(::rosidl_runtime_cpp::MessageInitialization::SKIP)
  {}
  Init_SimulatorCommand_method header(::simulator_messages::msg::SimulatorCommand::_header_type arg)
  {
    msg_.header = std::move(arg);
    return Init_SimulatorCommand_method(msg_);
  }

private:
  ::simulator_messages::msg::SimulatorCommand msg_;
};

}  // namespace builder

}  // namespace msg

template<typename MessageType>
auto build();

template<>
inline
auto build<::simulator_messages::msg::SimulatorCommand>()
{
  return simulator_messages::msg::builder::Init_SimulatorCommand_header();
}

}  // namespace simulator_messages

#endif  // SIMULATOR_MESSAGES__MSG__DETAIL__SIMULATOR_COMMAND__BUILDER_HPP_
