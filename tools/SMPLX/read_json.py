import numpy as np
import json
import os

def convert_npy_to_json(npy_path, save_path):
    data = np.load(npy_path)  # shape: (frame_num, 322)
    motion_185 = []

    for frame in data:
        vec185 = []

        # 1. betas (只用前10维)
        vec185 += frame[209:209+10].tolist()

        # 2. expression (前10维表情)
        vec185 += frame[159:159+10].tolist()

        # 3. global_orient
        vec185 += frame[0:3].tolist()

        # 4. body_pose
        vec185 += frame[3:66].tolist()

        # 5. face_pose: jaw + 2眼（有的模型没有眼）
        jaw = frame[156:159]
        eye_L = [0, 0, 0]  # 占位
        eye_R = [0, 0, 0]  # 占位
        vec185 += jaw.tolist() + eye_L + eye_R

        # 6. right_hand_pose
        vec185 += frame[66:66+45].tolist()

        # 7. left_hand_pose
        vec185 += frame[66+45:66+90].tolist()

        motion_185.append(vec185)

    with open(save_path, 'w') as f:
        json.dump(motion_185, f)
    print(f"Saved {len(motion_185)} frames to {save_path}")

# 示例用法
convert_npy_to_json(
    'E:\softwares_document\VS_Code_Projects\python_Project\FreeWorldSimulator\Answer_Phone.npy',
    'smplx_185_000001.json'
)
