import os
import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

plt.rcParams["font.sans-serif"] = ["SimHei"]  
plt.rcParams["axes.unicode_minus"] = False  


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
    """提取 Metrics 文件夹的 PathLength、StartTimeOfDay"""
    data = []
    for file in metrics_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                item = {"file": file}
                if "PathLength" in content:
                    item["PathLength"] = content["PathLength"]
                if "StartTimeOfDay" in content:
                    item["StartTimeOfDay"] = content["StartTimeOfDay"]
                if len(item) > 1:
                    data.append(item)
        except Exception as e:
            print(f"读取 {file} 失败: {e}")
    return data


def process_vlndata_files(vlndata_files):
    """提取 VLNData 文件夹的 WeatherTypeName、StartTimeOfDay"""
    data = []
    for file in vlndata_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = json.load(f)
                item = {"file": file}
                if "WeatherTypeName" in content:
                    item["WeatherTypeName"] = content["WeatherTypeName"]
                if "StartTimeOfDay" in content:
                    item["StartTimeOfDay"] = content["StartTimeOfDay"]
                if len(item) > 1:
                    data.append(item)
        except Exception as e:
            print(f"读取 {file} 失败: {e}")
    return data


def analyze_time_periods(start_times):
    """划分时间段：0-6是凌晨，6-18是早上，18-24是晚上"""
    periods = {
        "凌晨 (0-6)": 0,
        "早上 (6-18)": 0,
        "晚上 (18-24)": 0
    }
    for t in start_times:
        if 0 <= t < 6:
            periods["凌晨 (0-6)"] += 1
        elif 6 <= t < 18:
            periods["早上 (6-18)"] += 1
        else:
            periods["晚上 (18-24)"] += 1
    return {k: v for k, v in periods.items() if v > 0}


def create_pie_with_legend(data, title):
    """绘制扇形图（单独窗口），饼块标百分比，右侧列名称+占比"""
    labels = list(data.keys())
    sizes = list(data.values())
    total = sum(sizes)
    
    # 计算占比文本
    legend_texts = [f"{label} ({(s / total)*100:.1f}%)" for label, s in data.items()]
    
    # 新建独立窗口
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 绘制饼图
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct="%1.1f%%", 
        startangle=90, 
        colors=plt.cm.tab20.colors[:len(labels)]  
    )
    
    # 在右侧创建图例
    ax.legend(
        wedges, 
        legend_texts, 
        title=title, 
        loc="center left", 
        bbox_to_anchor=(1, 0.5), 
        fontsize=10, 
        title_fontsize=12
    )
    ax.set_title(title, fontsize=14, y=1.05)
    ax.axis("equal")
    
    plt.tight_layout()
    plt.show()


def create_pathlength_interval_bar(data):
    """绘制PathLength区间柱状图（横轴：区间，纵轴：数量）"""
    # 提取所有PathLength值
    path_lengths = [item["PathLength"] for item in data]
    if not path_lengths:
        print("没有PathLength数据可绘制")
        return
    
    # 自动计算合理区间（根据数据范围动态调整）
    min_pl = min(path_lengths)
    max_pl = max(path_lengths)
    # 计算区间间隔（确保5-10个区间）
    interval = max(1, round((max_pl - min_pl) / 8, 1)) 
    # 生成区间边界（从0开始，到超过最大值的区间结束）
    bins = []
    current = 0
    while current <= max_pl + interval:
        bins.append(round(current, 1))
        current += interval
    # 生成区间标签（如"0-5"、"5-10"）
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    
    # 统计每个区间的数量
    interval_counts = Counter()
    for pl in path_lengths:
        for i in range(len(bins)-1):
            if bins[i] <= pl < bins[i+1]:
                interval_counts[bin_labels[i]] += 1
                break
        else:  # 处理超出最大区间的情况
            interval_counts[bin_labels[-1]] += 1
    
    # 按区间顺序排列数据
    labels = bin_labels
    counts = [interval_counts[label] for label in labels]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制柱状图
    x_pos = np.arange(len(labels))
    bars = ax.bar(x_pos, counts, color="#FD9B2B") 
    
    # 设置坐标轴
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("PathLength 区间", fontsize=12)  
    ax.set_ylabel("数量", fontsize=12)  
    ax.set_title("PathLength 区间分布", fontsize=14, y=1.05)
    
    # 在柱子上方添加数量标签
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.1,
            f"{height}",
            ha="center", 
            va="bottom", 
            fontsize=9
        )
    
    plt.tight_layout()
    plt.show()


def main():
    root_dir = r"D:\Unity Hub\下载位置\HCI_Simulator\Datas\TrainDataSet"
    if not os.path.exists(root_dir):
        print(f"错误：目录 {root_dir} 不存在！")
        return

    # 查找 JSON 文件
    metrics_files = find_json_files(root_dir, "Metrics")
    vlndata_files = find_json_files(root_dir, "VLNData")

    # 处理数据
    metrics_data = process_metrics_files(metrics_files)
    vlndata_data = process_vlndata_files(vlndata_files)

    # 提取各类数据
    weather_types = [item["WeatherTypeName"] for item in vlndata_data if "WeatherTypeName" in item]
    weather_counter = Counter(weather_types)
    
    all_start_times = [
        item["StartTimeOfDay"] 
        for item in metrics_data + vlndata_data 
        if "StartTimeOfDay" in item
    ]
    time_periods = analyze_time_periods(all_start_times)
    
    pathlength_data = [item for item in metrics_data if "PathLength" in item]

    # 生成图表
    if weather_counter:
        create_pie_with_legend(weather_counter, "Weather 分布")
    
    if time_periods:
        create_pie_with_legend(time_periods, "Time分布")
    
    if pathlength_data:
        create_pathlength_interval_bar(pathlength_data)  


if __name__ == "__main__":
    main()