import os
import json

def check_files_in_folder(folder_path):
    for root, dir, files in os.walk(folder_path):
        for file in files:
            if file == 'VLNData.json':
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    for item in data.get("Dialogs", []):
                        humanlike_instruction = item.get("HumanLikeInstruction", None)

                        if "#" in humanlike_instruction or "□" in humanlike_instruction:
                            print(f"Wrong symbols in：{file_path}")
                            print(f"Content：{humanlike_instruction}\n")
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错：{e}")


folder_path = './test'
check_files_in_folder(folder_path)
