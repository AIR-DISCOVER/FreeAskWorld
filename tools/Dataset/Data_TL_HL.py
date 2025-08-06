import os
import json
from collections import Counter


def find_json_files(root_dir, target_folder):
    """递归查找目标文件夹下的 JSON 文件"""
    json_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == target_folder:
            for filename in filenames:
                if filename.endswith('.json'):
                    json_files.append(os.path.join(dirpath, filename))
    return json_files


def process_metrics_files(metrics_files):
    """提取 Metrics 文件夹的 PathLength"""
    data = []
    for file in metrics_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                item = {"file": file}
                if "PathLength" in content:
                    item["PathLength"] = content["PathLength"]
                if len(item) > 1:
                    data.append(item)
        except Exception as e:
            print(f"读取 {file} 失败: {e}")
    return data


def process_vlndata_files(vlndata_files):
    """提取 VLNData 文件夹的 human-like instruction"""
    data = []
    for file in vlndata_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                item = {"file": file}
                # 提取dialog中的human-like instruction
                if "dialog" in content:
                    dialog = content["dialog"]
                    # 假设human-like instruction在dialog的"instruction"字段中
                    instructions = [turn.get("instruction", "") for turn in dialog if isinstance(turn, dict)]
                    item["human_like_instructions"] = instructions
                    item["total_instruction_length"] = sum(len(inst) for inst in instructions)
                if len(item) > 1:
                    data.append(item)
        except Exception as e:
            print(f"读取 {file} 失败: {e}")
    return data


def calculate_tl_average(metrics_data):
    """计算TL（PathLength）的平均值"""
    path_lengths = [item["PathLength"] for item in metrics_data if "PathLength" in item]
    if not path_lengths:
        return None
    return sum(path_lengths) / len(path_lengths)


def calculate_hl_average(vlndata_data):
    """计算每个dialog中human-like instruction长度的平均值"""
    total_lengths = []
    for item in vlndata_data:
        if "total_instruction_length" in item and "human_like_instructions" in item:
            instructions = item["human_like_instructions"]
            if instructions:  # 确保有指令
                total_lengths.append(item["total_instruction_length"])
    
    if not total_lengths:
        return None
    return sum(total_lengths) / len(total_lengths)


def main():
    root_dir = r"D:\Unity Hub\下载位置\HCI_Simulator\Datas\TrainDataSet"
    if not os.path.exists(root_dir):
        print(f"错误：目录 {root_dir} 不存在！")
        return

    # 查找 JSON 文件
    metrics_files = find_json_files(root_dir, "Metrics")
    vlndata_files = find_json_files(root_dir, "VLNData")

    print(f"找到 {len(metrics_files)} 个 Metrics JSON 文件")
    print(f"找到 {len(vlndata_files)} 个 VLNData JSON 文件")

    # 处理数据
    metrics_data = process_metrics_files(metrics_files)
    vlndata_data = process_vlndata_files(vlndata_files)

    # 计算TL平均值（PathLength的平均值）
    tl_average = calculate_tl_average(metrics_data)
    if tl_average is not None:
        print(f"TL平均值: {tl_average:.2f}")
    else:
        print("没有足够的数据计算TL平均值")

    # 计算HL平均值（human-like instruction的平均总长度）
    hl_average = calculate_hl_average(vlndata_data)
    if hl_average is not None:
        print(f"HL平均值: {hl_average:.2f}")
    else:
        print("没有足够的数据计算HL平均值")


if __name__ == "__main__":
    main()
    