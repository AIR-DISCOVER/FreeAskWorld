import os
import json
from datetime import datetime

def get_first_level_subdirectories(root_path):
    return [
        name for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]

def generate_json(root_path, dataset_type, output_file='folder_list.json'):
    folder_list = get_first_level_subdirectories(root_path)
    result = {
        "Type": dataset_type,
        "Time": datetime.now().isoformat(),
        "FolderList": folder_list
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"JSON saved to {output_file}")


if __name__ == "__main__":
    root = r"E:\softwares_document\unity_document\HCI Simulator\Assets\Resources\Datas\Test"
    generate_json(root, 'test')
