import numpy as np
import json
import os
from tqdm import tqdm

def merge_all_json_filenames(dst_root, merged_json_path):
    json_files = []
    for dirpath, _, filenames in os.walk(dst_root):
        for filename in filenames:
            if filename.endswith('.json') and filename != os.path.basename(merged_json_path):
                json_path = os.path.join(dirpath, filename)
                rel_path = os.path.relpath(json_path, dst_root).replace("\\", "/")
                json_files.append(rel_path)

    with open(merged_json_path, 'w') as f:
        json.dump(json_files, f, indent=2)
    print(f"已写入所有文件名到：{merged_json_path}")
    
    
    
if __name__ == "__main__":
    dst_folder = r"E:\A-Dataset\MotionX\motion_data_unity"

    # 仅收集json文件名
    merged_output_path = os.path.join(dst_folder, "DataBase.json")
    merge_all_json_filenames(dst_folder, merged_output_path)
