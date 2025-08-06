from typing import Optional
import numpy as np
from messages import *  

# 可共享的变量（全局缓存）
rgb_array: Optional[np.ndarray] = None
depth_array: Optional[np.ndarray] = None
transform_data: Optional[TransformData] = None
instruction: Optional[str] = None

# 标志位
Init: Optional[bool] = False

def clear_shared_state():
    global rgb_array, depth_array, transform_data, instruction
    rgb_array = None
    depth_array = None
    transform_data = None
    instruction = None
