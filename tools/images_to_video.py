import os
import cv2
import glob
import numpy as np

def images_to_video(image_folder, output_path, fps=24):
    image_files = sorted(
        glob.glob(os.path.join(image_folder, "*.png"))
    )

    if not image_files:
        raise ValueError("No PNG images found in the folder.")

    # 使用 Unicode 路径读取第一张图片以获取宽高
    first_img = cv2.imdecode(
        np.fromfile(image_files[0], dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if first_img is None:
        raise ValueError(f"Failed to read the first image: {image_files[0]}")
    
    height, width, _ = first_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for file in image_files:
        img = cv2.imdecode(np.fromfile(file, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Skipping unreadable image {file}")
            continue
        out.write(img)

    out.release()
    print(f"✅ Video saved to {output_path}")


# 示例使用方式：
if __name__ == "__main__":
    image_folder_path = r"E:\A_Work\学术\AIR\服务机器人人机交互Benchmark\技术方案设计\实验\ETPNav FreeAskWorld2\20250728_084921\Video\BenchmarkVideo"
    output_video_path = "output_video.mp4"
    images_to_video(image_folder_path, output_video_path, fps=12)
