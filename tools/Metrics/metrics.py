import os
import json
from pathlib import Path

def calculate_success_rate(train_dir):
    
    success_count = 0
    oracle_success_count = 0
    total_count = 0
    total_traj_length = 0
    total_nav_error = 0
    total_oracle_nav_error = 0
    total_askway_num = 0
    spl_sum = 0 
    
    # 遍历Train目录下的所有文件夹
    for sub_dir in os.listdir(train_dir):
        sub_dir_path = os.path.join(train_dir, sub_dir)
        
        # 检查是否是文件夹
        if not os.path.isdir(sub_dir_path):
            continue
            
        # 构建TestMetrics文件夹路径
        test_metrics_dir = os.path.join(sub_dir_path, "TestMetrics")
        
        # 检查TestMetrics文件夹是否存在
        if not os.path.exists(test_metrics_dir) or not os.path.isdir(test_metrics_dir):
            continue
            
        # 遍历TestMetrics文件夹中的所有json文件
        for file_name in os.listdir(test_metrics_dir):
            if file_name.endswith(".json"):
                file_path = os.path.join(test_metrics_dir, file_name)
                
                try:
                    # 读取JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    if not isinstance(data, dict) or "IsSuccess" not in data or "TrajectoryLength" not in data or "PathLength" not in data:
                        continue
                    
                    total_count += 1
                    
                    s = 1 if data["IsSuccess"] is True else 0
                    if s == 1:
                        success_count += 1
                    
                    # 计算OracleSuccess
                    if "OracleSuccess" in data:
                        if data["OracleSuccess"] is True:
                            oracle_success_count += 1
                    
                    # 获取轨迹长度P和理想路径长度L
                    p = float(data["TrajectoryLength"])
                    l = float(data["PathLength"])
                    
                    # 累加TrajectoryLength
                    total_traj_length += p
                    
                    # 累加NavigationError
                    if "NavigationError" in data:
                        total_nav_error += float(data["NavigationError"])
                        
                    # 累加OracleNavigationError
                    if "OracleNavigationError" in data:
                        #print(float(data["OracleNavigationError"]))
                        total_oracle_nav_error += float(data["OracleNavigationError"])
                        
                    # 累加AskwayNum
                    if "AskwayNum" in data:
                        total_askway_num += int(data["AskwayNum"])
                    
                    # 计算并累加SPL分量
                    max_lp = max(l, p)
                    if max_lp > 0:  
                        spl_sum += s * (l / max_lp)
                        
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
    
    # 计算metrics指标
    success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
    avg_traj_length = total_traj_length / total_count if total_count > 0 else 0
    spl = spl_sum / total_count if total_count > 0 else 0 
    avg_nav_error = total_nav_error / total_count if total_count > 0 else 0
    avg_oracle_nav_error = total_oracle_nav_error / total_count if total_count > 0 else 0
    oracle_success_rate = (oracle_success_count / total_count) * 100 if total_count > 0 else 0   
    avg_askway_num = total_askway_num / total_count if total_count > 0 else 0
     
    
    return success_rate, avg_traj_length, spl, avg_nav_error, avg_oracle_nav_error, oracle_success_rate, avg_askway_num

if __name__ == "__main__":
    # 训练数据目录路径
    train_directory = r"E:\A_Work\学术\AIR\服务机器人人机交互Benchmark\技术方案设计\实验\human baseline xinyu\Benchmarking"
    
    # 验证目录是否存在
    if not os.path.exists(train_directory) or not os.path.isdir(train_directory):
        print(f"错误: 指定的目录 '{train_directory}' 不存在或不是一个目录。")
    else:
        sr, tl, spl, ne, one, osr, awn = calculate_success_rate(train_directory)
        print(f"SR={sr:.2f}%")
        print(f"TL={tl:.2f}")
        print(f"SPL={spl:.4f}")
        print(f"NE={ne:.2f}")
        print(f"ONE={one:.2f}")
        print(f"OSR={osr:.2f}%")
        print(f"NDI={awn:.2f}")  # Number of Direction Inquiries (NDI)（Ask Way Num）
