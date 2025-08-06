import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def read_exr(path, channels=("R", "G", "B")):
    file = OpenEXR.InputFile(path)
    header = file.header()
    dw = header["dataWindow"]
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    data = []
    for c in channels:
        channel_data = file.channel(c, pt)
        arr = np.frombuffer(channel_data, dtype=np.float32)
        arr = arr.reshape((size[1], size[0]))  # (H, W)
        data.append(arr)

    result = np.stack(data, axis=-1)  # (H, W, C)
    return result


def visualize(normal_path, depth_path):
    normal = read_exr(normal_path, channels=("R", "G", "B"))  # (H, W, 3)
    depth = read_exr(depth_path, channels=("R",))  # (H, W, 1)
    depth = depth[..., 0]  # squeeze to (H, W)

    # 归一化处理
    normal = (normal + 1.0) / 2.0  # [-1,1] -> [0,1]
    normal = np.clip(normal, 0, 1)

    depth_vis = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)

    # 保存图像
    output_dir = "visualized_outputs"
    os.makedirs(output_dir, exist_ok=True)

    normal_save_path = os.path.join(output_dir, "normal_visualization.png")
    depth_save_path = os.path.join(output_dir, "depth_visualization.png")

    plt.imsave(normal_save_path, normal)
    plt.imsave(depth_save_path, depth_vis, cmap='plasma')

    print(f"Normal saved to: {normal_save_path}")
    print(f"Depth saved to: {depth_save_path}")

    # 可视化显示
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(normal)
    axes[0].set_title("Normal")
    axes[0].axis('off')

    axes[1].imshow(depth_vis, cmap='plasma')
    axes[1].set_title("Depth")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    normal_path = r"E:\softwares_document\VS_Code_Projects\python_Project\FreeAskWorldSimulator\data\Selected\step2.PanoCamera_front_Perception.Normal_3.exr"
    depth_path = r"E:\softwares_document\VS_Code_Projects\python_Project\FreeAskWorldSimulator\data\Selected\step2.PanoCamera_front_Perception.Depth_3.exr"

    visualize(normal_path, depth_path)
