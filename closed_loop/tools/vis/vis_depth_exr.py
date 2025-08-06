import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import os

def read_exr_depth_linear(filepath, near=0.1, far=1000.0):
    exr = OpenEXR.InputFile(filepath)
    print("Channels:", exr.header()['channels'].keys())  # 打印所有通道名，确认用哪个通道

    dw = exr.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    # 常用深度通道是 'R' 或 'Z'，你可以根据打印结果换
    channel_name = 'Y'
    if 'Z' in exr.header()['channels']:
        channel_name = 'Z'

    depth_str = exr.channel(channel_name, FLOAT)
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape((height, width))

    print("Depth min:", np.min(depth), "max:", np.max(depth), "mean:", np.mean(depth))

    # 线性化
    z_b = depth  # 原始深度缓冲值，范围[0,1]
    z_b = np.clip(z_b, 0, 1)
    z_n = 2.0 * z_b - 1.0
    linear_depth = (2.0 * near * far) / (far + near - z_n * (far - near))

    return linear_depth

def visualize_depth(depth):
    plt.imshow(depth, cmap='gray')
    plt.colorbar(label='Linear Depth')
    plt.title('Linearized Depth Map')
    plt.axis('off')
    os.makedirs("vis", exist_ok=True)
    plt.savefig("vis/depth_linear.png")
    plt.show()

if __name__ == "__main__":
    filepath = 'received/depth_20250710_152226.exr'  # 你自己的路径
    linear_depth = read_exr_depth_linear(filepath, near=0.1, far=1000)
    visualize_depth(linear_depth)
