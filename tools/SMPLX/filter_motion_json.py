import os
import json

def clean_motion_folder(motion_folder_path: str, keep_list_json_path: str):
    # 1. 读取 JSON 文件（结构如你提供）
    with open(keep_list_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. 递归提取所有动作名
    def extract_all_motion_names(d):
        motions = []
        if isinstance(d, dict):
            for v in d.values():
                motions.extend(extract_all_motion_names(v))
        elif isinstance(d, list):
            for item in d:
                motions.append(item)
        return motions

    motion_names = extract_all_motion_names(data)

    # 3. 构造应保留的文件名（加 .json 后缀）
    keep_filenames = set(name + ".json" for name in motion_names)

    # 4. 遍历 motion 文件夹，删除不在 keep_filenames 中的 .json 文件
    deleted = 0
    kept = 0
    for filename in os.listdir(motion_folder_path):
        file_path = os.path.join(motion_folder_path, filename)
        if filename.endswith('.json'):
            if filename not in keep_filenames:
                os.remove(file_path)
                print(f"Deleted: {filename}")
                deleted += 1
            else:
                kept += 1

    print(f"✅ Done. Kept: {kept} files, Deleted: {deleted} files.")


# ✅ 示例调用
if __name__ == "__main__":
    a = r"E:\softwares_document\unity_document\HCI Simulator\MotionData\MotionX\smplx_322"        # ← 替换为你的 motion json 文件夹路径
    b = r"E:\softwares_document\VS_Code_Projects\python_Project\FreeAskWorldSimulator\final_motion(2).json"      # ← 替换为你的 json 文件路径
    clean_motion_folder(a, b)
