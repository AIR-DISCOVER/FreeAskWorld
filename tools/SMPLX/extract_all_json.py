import os
import shutil

def copy_all_json_files(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(".json"):
                source_file = os.path.join(root, file)
                target_file = os.path.join(target_dir, file)

                # 防止同名文件覆盖：可在文件名前加上父目录名
                if os.path.exists(target_file):
                    parent_folder = os.path.basename(root)
                    new_name = f"{parent_folder}_{file}"
                    target_file = os.path.join(target_dir, new_name)

                shutil.copy2(source_file, target_file)
                print(f"Copied: {source_file} -> {target_file}")

# 示例用法：
source_folder = r"E:\A-Dataset\MotionX\motion_data_unity\smplx_322"
target_folder = r"E:\softwares_document\unity_document\HCI Simulator\MotionData\MotionX\smplx_322"
copy_all_json_files(source_folder, target_folder)
