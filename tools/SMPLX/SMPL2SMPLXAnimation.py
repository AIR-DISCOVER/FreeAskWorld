import numpy as np
import os

def convert_amass_to_smplx(amass_npz_path, output_dir):
    """
    将 AMASS 格式的 npz 文件转换为 SMPL-X 格式的 npy 文件
    （结构与 Motion-X 一致，每帧322维）
    """
    os.makedirs(output_dir, exist_ok=True)
    data = np.load(amass_npz_path)

    # ---- 读取原始参数 ----
    poses = data["poses"]          # [N, 156] for SMPL-H (body + hands)
    betas = data["betas"]          # [10]
    trans = data["trans"]          # [N, 3]

    # ---- 拆分 body / hand ----
    root_orient = poses[:, :3]      # 全局旋转
    pose_body = poses[:, 3:66]      # 身体姿态 (21 joints * 3)
    
    # 若有手部参数
    if poses.shape[1] >= 156:  # SMPL-H 格式
        pose_hand = poses[:, 66:66+90]   # 左右手各15×3维
    else:
        pose_hand = np.zeros((poses.shape[0], 90))
    
    # SMPL 没有脸/表情，补零
    pose_jaw   = np.zeros((poses.shape[0], 3))
    face_expr  = np.zeros((poses.shape[0], 50))
    face_shape = np.zeros((poses.shape[0], 100))

    # betas 拓展到 10维
    betas_full = np.tile(betas[:10], (poses.shape[0], 1))

    # ---- 按 SMPL-X 参数顺序拼接 ----
    motion = np.concatenate([
        root_orient,      # 3
        pose_body,        # 63
        pose_hand,        # 90
        pose_jaw,         # 3
        face_expr,        # 50
        face_shape,       # 100
        trans,            # 3
        betas_full        # 10
    ], axis=1)

    print(f"Final motion shape: {motion.shape}")  # [N, 322]

    # ---- 保存 npy 文件 ----
    base = os.path.splitext(os.path.basename(amass_npz_path))[0]
    out_path = os.path.join(output_dir, f"{base}_smplx.npy")
    np.save(out_path, motion)
    print(f"✅ Saved: {out_path}")

# ===== 批处理示例 =====
if __name__ == "__main__":
    input_folder = r"D:\gendered_ground_truth\gendered_ground_truth\male_34_us_1371\moving_body_para\0005"    # 你的AMASS npz文件夹
    output_folder = r"D:\data\npy" # 输出npy文件夹
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".npz"):
            path = os.path.join(input_folder, file)
            convert_amass_to_smplx(path, output_folder)

