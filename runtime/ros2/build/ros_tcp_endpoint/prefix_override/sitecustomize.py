import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/wyabz/research/FreeAskWorld/runtime/ros2/install/ros_tcp_endpoint'
