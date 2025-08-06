#!/usr/bin/env python3
"""
修复UTF-8 BOM问题并重新转换数据
"""

import json
import os
from pathlib import Path
import codecs

def fix_utf8_bom_in_file(file_path):
    """修复单个文件的UTF-8 BOM问题"""
    try:
        # 尝试用utf-8-sig读取（会自动去除BOM）
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
        
        # 重新写入，不带BOM
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return True
    except Exception as e:
        print(f"修复文件 {file_path} 失败: {e}")
        return False

def fix_all_json_files():
    """修复所有JSON文件的BOM问题"""
    source_dir = Path('/data/yinxy/etpnav_training_data/datasets')
    fixed_count = 0
    
    print("🔧 开始修复UTF-8 BOM问题...")
    
    # 查找所有JSON文件
    json_files = list(source_dir.rglob('*.json'))
    print(f"找到 {len(json_files)} 个JSON文件")
    
    for json_file in json_files:
        if fix_utf8_bom_in_file(json_file):
            fixed_count += 1
    
    print(f"✅ 成功修复 {fixed_count} 个文件")
    return fixed_count

def main():
    print("🚀 开始修复BOM问题...")
    
    # 修复BOM问题
    fix_all_json_files()
    
    print("\n🔄 重新运行转换...")
    
    # 重新导入并运行转换器
    import sys
    sys.path.append('/data/yinxy/etpnav_training_data')
    
    from convert_unity_data import UnityToVLNCEConverter
    
    # 删除之前的输出（如果存在）
    import shutil
    output_dir = Path('/data/yinxy/etpnav_training_data/converted_vlnce')
    if output_dir.exists():
        print("删除之前的转换结果...")
        shutil.rmtree(output_dir)
    
    # 重新转换
    converter = UnityToVLNCEConverter()
    converter.run_conversion()

if __name__ == "__main__":
    main()
