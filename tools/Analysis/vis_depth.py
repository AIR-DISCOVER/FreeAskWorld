import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取 EXR 深度图
exr_path = r"E:\softwares_document\VS_Code_Projects\python_Project\FreeWorldSimulator\data\solo_test\sequence.0\step0.Perception Test Camera.Depth.exr"
img = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)

# 使用某一通道（例如 Z 在 B 通道）
depth = img[:, :, 2]  # 2 表示 B 通道

# # 打印结构信息
# print("=== 图像结构信息 ===")
# print(f"类型：{type(img)}")
# print(f"形状（H, W, C）：{img.shape}")
# print(f"数据类型：{img.dtype}")
# print(f"像素最小值：{np.min(img)}")
# print(f"像素最大值：{np.max(img)}")

# for i, name in enumerate(["R", "G", "B", "A"]):
#     channel = img[:, :, i]
#     print(f"\n--- 通道 {name} ---")
#     print(f"最小值：{np.min(channel)}")
#     print(f"最大值：{np.max(channel)}")


# 打印 min/max 以调试
print("Depth min/max:", depth.min(), depth.max())

# 可视化归一化
norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

# 显示图像
plt.imshow(norm, cmap="plasma")
plt.title("Depth Visualization (B channel)")
plt.colorbar()
plt.axis("off")
plt.show()
