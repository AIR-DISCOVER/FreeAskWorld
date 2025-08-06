import numpy as np
import json
import os
from tqdm import tqdm

def convert_npy_to_json(npy_path, json_path):
    data = np.load(npy_path, allow_pickle=True)
    data_list = data.tolist()
    with open(json_path, 'w') as f:
        json.dump(data_list, f, indent=2)
    # print(f"Converted:\n  {npy_path}\n-> {json_path}")

def batch_convert_and_copy_structure(src_root, dst_root):
    # 先收集所有 npy 文件路径
    npy_files = []
    for dirpath, _, filenames in os.walk(src_root):
        for filename in filenames:
            if filename.endswith('.npy'):
                full_path = os.path.join(dirpath, filename)
                npy_files.append(full_path)
    total = len(npy_files)
    print(f"找到 {total} 个 .npy 文件，开始转换...")

    for npy_file in tqdm(npy_files, desc="转换进度"):
        # 计算相对路径
        rel_path = os.path.relpath(os.path.dirname(npy_file), src_root)
        # 目标文件夹路径
        dst_dir = os.path.join(dst_root, rel_path)
        os.makedirs(dst_dir, exist_ok=True)

        json_file = os.path.join(dst_dir, os.path.splitext(os.path.basename(npy_file))[0] + '.json')
        try:
            convert_npy_to_json(npy_file, json_file)
        except Exception as e:
            print(f"转换失败: {npy_file}\n错误信息: {e}")

if __name__ == "__main__":
    src_folder = r"E:\A-Dataset\MotionX\motion_data"  # 源目录
    dst_folder = r"E:\A-Dataset\MotionX\motion_data_unity"      # 目标目录
    batch_convert_and_copy_structure(src_folder, dst_folder)
