import os
import re
import json
import shutil
from collections import defaultdict


def main(train1_path):
    # 1. 获取所有子文件夹并按时间排序
    subfolders = [f.path for f in os.scandir(train1_path) if f.is_dir()]
    sorted_folders = sorted(subfolders, key=lambda x: os.path.basename(x))

    if not sorted_folders:
        print("错误：未找到子文件夹")
        return

    # 2. 找到最新（时间序号最大）的子文件夹
    latest_folder = sorted_folders[-1]
    print(f"最新文件夹: {latest_folder}")

    # 3. 获取源文件路径
    source_seq0 = os.path.join(latest_folder, "PerceptionData", "solo", "sequence.0")
    if not os.path.exists(source_seq0):
        print(f"错误：未找到源文件夹 {source_seq0}")
        return

    # 4. 收集所有step文件并分组
    file_groups = defaultdict(list)
    pattern = re.compile(r'step(\d+)\.')

    for filename in os.listdir(source_seq0):
        match = pattern.match(filename)
        if match:
            step_idx = int(match.group(1))
            file_groups[step_idx].append(filename)

    if not file_groups:
        print("错误：未找到step文件")
        return

    # 5. 找出连续的块
    all_indices = sorted(file_groups.keys())
    blocks = []
    current_block = []

    for i in range(len(all_indices)):
        if not current_block:
            current_block.append(all_indices[i])
        else:
            if all_indices[i] == current_block[-1] + 1:
                current_block.append(all_indices[i])
            else:
                blocks.append(current_block)
                current_block = [all_indices[i]]

    if current_block:
        blocks.append(current_block)

    print(f"找到 {len(blocks)} 个连续块: {[f'{b[0]}-{b[-1]}' for b in blocks]}")
    print(f"共有 {len(sorted_folders)} 个子文件夹")

    # 6. 检查块数量是否匹配
    if len(blocks) != len(sorted_folders):
        print(f"警告：连续块数量({len(blocks)})与子文件夹数量({len(sorted_folders)})不匹配")
        # 尝试调整：使用所有块，最后一个文件夹保留最后一个块
        if len(blocks) > len(sorted_folders):
            print("警告：块数量多于文件夹数量，将尝试合并")
            return
        if len(blocks) < len(sorted_folders):
            print("警告：块数量少于文件夹数量，将使用所有块")

    # 7. 将块分配到其他子文件夹
    for i, folder in enumerate(sorted_folders[:-1]):  # 排除最新文件夹
        if i >= len(blocks):
            print(f"警告：没有足够的块分配给 {folder}")
            continue

        block = blocks[i]
        target_seq0 = os.path.join(folder, "PerceptionData", "solo", "sequence.0")

        # 创建目标目录
        os.makedirs(target_seq0, exist_ok=True)

        # 移动文件
        moved_count = 0
        for step_idx in block:
            for filename in file_groups[step_idx]:
                src = os.path.join(source_seq0, filename)
                if os.path.exists(src):
                    dst = os.path.join(target_seq0, filename)
                    shutil.move(src, dst)
                    moved_count += 1

        print(f"已移动块 {block[0]}-{block[-1]} ({moved_count}个文件) 到 {folder}")

    # 8. 复制其他文件（除sequence.0外）
    source_solo = os.path.join(latest_folder, "PerceptionData", "solo")
    for folder in sorted_folders[:-1]:
        target_solo = os.path.join(folder, "PerceptionData", "solo")

        if not os.path.exists(target_solo):
            os.makedirs(target_solo)

        copied_count = 0
        for item in os.listdir(source_solo):
            if item == "sequence.0":
                continue

            src = os.path.join(source_solo, item)
            dst = os.path.join(target_solo, item)

            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
                copied_count += 1
            else:
                shutil.copy2(src, dst)
                copied_count += 1

        print(f"已复制 {copied_count} 个其他文件到 {folder}")

    # 9. 处理所有子文件夹的sequence.0文件
    for folder in sorted_folders:
        seq0_path = os.path.join(folder, "PerceptionData", "solo", "sequence.0")
        json_path = os.path.join(folder, "PerceptionData", "step_transform.json")

        if not os.path.exists(seq0_path):
            print(f"跳过 {folder}，缺少 sequence.0 文件夹")
            continue
        if not os.path.exists(json_path):
            print(f"跳过 {folder}，缺少 step_transform.json 文件")
            continue

        try:
            # 使用 utf-8-sig 编码读取JSON文件
            with open(json_path, 'r', encoding='utf-8-sig') as f:
                transform_data = json.load(f)
        except Exception as e:
            print(f"处理 {json_path} 时出错: {str(e)}")
            continue

        json_indices = sorted([int(k) for k in transform_data.keys()])
        total_frames = len(json_indices)

        # 收集当前sequence.0中的文件
        current_files = []
        for filename in os.listdir(seq0_path):
            match = pattern.match(filename)
            if match:
                step_idx = int(match.group(1))
                current_files.append((step_idx, filename))

        if not current_files:
            print(f"跳过 {folder}，无step文件")
            continue

        # 获取唯一索引并排序
        unique_indices = sorted(set(idx for idx, _ in current_files))
        actual_frames = len(unique_indices)
        missing_frames = total_frames - actual_frames

        # 确定重命名范围
        if missing_frames < 0:
            print(f"警告：{folder} JSON帧数({total_frames})少于文件帧数({actual_frames})")
            # 使用所有JSON索引
            rename_indices = json_indices
        elif missing_frames > 0:
            rename_indices = json_indices[missing_frames:]
        else:
            rename_indices = json_indices

        # 按原始索引分组文件
        file_groups = defaultdict(list)
        for idx, filename in current_files:
            file_groups[idx].append(filename)

        sorted_indices = sorted(file_groups.keys())

        # 重命名文件 - 修复源和目标相同的问题
        renamed_count = 0
        skipped_count = 0

        # 创建重命名映射
        rename_map = {}
        for orig_idx, new_idx in zip(sorted_indices, rename_indices):
            for filename in file_groups[orig_idx]:
                # 获取文件扩展名
                ext = filename.split('.', 1)[1] if '.' in filename else ''

                # 创建新文件名
                new_name = f"step{new_idx}.{ext}"

                # 如果新文件名与旧文件名相同，则跳过
                if new_name == filename:
                    skipped_count += 1
                    continue

                rename_map[filename] = new_name

        # 执行重命名
        for old_name, new_name in rename_map.items():
            src = os.path.join(seq0_path, old_name)
            dst = os.path.join(seq0_path, new_name)

            if not os.path.exists(src):
                print(f"警告：文件不存在 {src}")
                continue

            # 如果目标文件已存在，先删除
            if os.path.exists(dst):
                os.remove(dst)

            # 重命名文件
            os.rename(src, dst)
            renamed_count += 1

        # 打印结果
        folder_name = os.path.basename(folder)
        if missing_frames == 0:
            print(
                f"{folder_name} 无丢帧，共{actual_frames}帧，命名为{rename_indices[0]}-{rename_indices[-1]}，已重命名{renamed_count}个文件，跳过{skipped_count}个文件")
        elif missing_frames > 0:
            print(
                f"{folder_name} 丢失{missing_frames}帧，共{actual_frames}帧，命名为{rename_indices[0]}-{rename_indices[-1]}，已重命名{renamed_count}个文件，跳过{skipped_count}个文件")
        else:
            print(
                f"{folder_name} 文件帧数({actual_frames})多于JSON帧数({total_frames})，使用JSON索引{json_indices[0]}-{json_indices[-1]}，已重命名{renamed_count}个文件，跳过{skipped_count}个文件")


if __name__ == "__main__":
    # import sys
    #
    # if len(sys.argv) != 2:
    #     print("用法: python script.py <Train_1路径>")
    #     sys.exit(1)
    #
    # main(sys.argv[1])
    main("E:\A-Dataset\FreeAskWorld\Train\TrainDataSet_epoch_2025719_7")
