import os
import json
from typing import Dict, Any, List, Tuple

def calculate_success_rate(root_dir: str) -> Tuple[float, float, float, float, float, float, float]:
    """
    遍历根目录下所有包含TestMetrics的路径，计算多种导航评估指标
    
    Args:
        root_dir: 根目录路径
    
    Returns:
        成功率, 平均轨迹长度, SPL, 平均导航误差, 平均Oracle导航误差, Oracle成功率, 平均询问次数
    """
    success_count = 0
    oracle_success_count = 0
    total_count = 0
    total_traj_length = 0
    total_nav_error = 0
    total_oracle_nav_error = 0
    total_askway_num = 0
    spl_sum = 0
    
    # 遍历根目录下的所有内容
    for root, dirs, _ in os.walk(root_dir):
        # 检查当前目录是否包含TestMetrics子目录
        if 'TestMetrics' in dirs:
            test_metrics_dir = os.path.join(root, 'TestMetrics')
            
            # 遍历TestMetrics目录下的所有JSON文件
            for json_root, _, files in os.walk(test_metrics_dir):
                for file in files:
                    if file.lower().endswith('.json'):
                        file_path = os.path.join(json_root, file)
                        try:
                            # 读取JSON文件
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            
                            # 检查必要字段是否存在
                            if not isinstance(data, dict) or "IsSuccess" not in data or "TrajectoryLength" not in data or "PathLength" not in data:
                                continue
                            
                            total_count += 1
                            
                            # 计算成功率
                            s = 1 if data["IsSuccess"] is True else 0
                            if s == 1:
                                success_count += 1
                            
                            # 计算OracleSuccess
                            if "OracleSuccess" in data and data["OracleSuccess"] is True:
                                oracle_success_count += 1
                            
                            # 获取轨迹长度P和理想路径长度L
                            p = float(data["TrajectoryLength"])
                            l = float(data["PathLength"])
                            
                            # 累加轨迹长度
                            total_traj_length += p
                            
                            # 累加导航误差
                            if "NavigationError" in data:
                                total_nav_error += float(data["NavigationError"])
                            
                            # 累加Oracle导航误差
                            if "OracleNavigationError" in data:
                                total_oracle_nav_error += float(data["OracleNavigationError"])
                            
                            # 累加询问次数
                            if "AskwayNum" in data:
                                total_askway_num += int(data["AskwayNum"])
                            
                            # 计算并累加SPL (Success weighted by Path Length)
                            max_lp = max(l, p)
                            if max_lp > 0:
                                spl_sum += s * (l / max_lp)
                                
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错: {e}")
    
    # 计算各项指标
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    avg_traj_length = total_traj_length / total_count if total_count > 0 else 0
    spl = spl_sum / total_count if total_count > 0 else 0
    avg_nav_error = total_nav_error / total_count if total_count > 0 else 0
    avg_oracle_nav_error = total_oracle_nav_error / total_count if total_count > 0 else 0
    oracle_success_rate = (oracle_success_count / total_count) * 100 if total_count > 0 else 0
    avg_askway_num = total_askway_num / total_count if total_count > 0 else 0
    
    return success_rate, avg_traj_length, spl, avg_nav_error, avg_oracle_nav_error, oracle_success_rate, avg_askway_num

if __name__ == "__main__":
    # 指定根目录
    root_directory = r"D:\Unity Hub\下载位置\HCI_Simulator\Datas\2"
    
    # 验证目录是否存在
    if not os.path.exists(root_directory) or not os.path.isdir(root_directory):
        print(f"错误: 指定的目录 '{root_directory}' 不存在或不是一个目录。")
    else:
        # 计算各项指标
        sr, tl, spl, ne, one, osr, awn = calculate_success_rate(root_directory)
        
        # 输出结果（保留两位小数，SPL保留四位小数）
        print(f"SR={sr:.2f}%")     # Success Rate
        print(f"TL={tl:.2f}")      # Average Trajectory Length
        print(f"SPL={spl:.4f}")    # Success weighted by Path Length
        print(f"NE={ne:.2f}")      # Average Navigation Error
        print(f"ONE={one:.2f}")    # Average Oracle Navigation Error
        print(f"OSR={osr:.2f}%")   # Oracle Success Rate
        print(f"AWN={awn:.2f}")    # Average Ask Way Number