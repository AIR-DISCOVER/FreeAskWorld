import os
import json

directory = r"D:\Unity Hub\下载位置\HCI_Simulator\Datas\TestDataSet"

try:
    success_folders = 0  # 包含IsSuccess:true的文件夹数
    total_folders = 0    # 包含Metrics文件夹的总文件夹数
    trajectory_lengths = []  # 存储所有TrajectoryLength值
    
    for root, dirs, files in os.walk(directory):
        # 跳过根目录，只处理子文件夹
        if root == directory:
            continue
            
        # 检查是否存在Metrics文件夹
        metrics_path = os.path.join(root, 'Metrics')
        if not os.path.isdir(metrics_path):
            continue
            
        # 该文件夹包含Metrics，计数+1
        total_folders += 1
        
        # 查找Metrics文件夹中的JSON文件
        json_files = [f for f in os.listdir(metrics_path) if f.endswith('.json')]
        
        # 如果没有JSON文件，算失败
        if not json_files:
            continue
            
        # 检查是否至少有一个JSON文件包含IsSuccess:true
        has_success = False
        for json_file in json_files:
            json_path = os.path.join(metrics_path, json_file)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 提取TrajectoryLength值
                    if 'TrajectoryLength' in data:
                        trajectory_lengths.append(data['TrajectoryLength'])
                    # 检查成功标志
                    if data.get('IsSuccess') is True:
                        has_success = True
            except (json.JSONDecodeError, PermissionError):
                continue  # 解析失败则继续检查其他文件
        
        # 如果有成功标记，计数+1
        if has_success:
            success_folders += 1
    
    # 计算成功率
    success_rate = (success_folders / total_folders) * 100 if total_folders > 0 else 0
    
    # 计算轨迹长度统计数据
    tl_sum = sum(trajectory_lengths) if trajectory_lengths else 0
    tl_avg = tl_sum / len(trajectory_lengths) if trajectory_lengths else 0
    
    print(f"目录 '{directory}' 中共有 {total_folders} 个包含Metrics的文件夹")
    print(f"成功: {success_folders}, 失败: {total_folders - success_folders}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"TL (Trajectory Length) - 总和: {tl_sum:.2f}, 平均: {tl_avg:.2f}")
    
except FileNotFoundError:
    print(f"错误：找不到目录 '{directory}'")
except PermissionError:
    print(f"错误：没有权限访问目录 '{directory}'")
except Exception as e:
    print(f"发生未知错误：{e}")    