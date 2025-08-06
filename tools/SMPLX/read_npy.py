import numpy as np
import torch

motion_data_path = "E:\softwares_document\VS_Code_Projects\python_Project\FreeWorldSimulator\Answer_Phone.npy"

# read motion and save as smplx representation
motion = np.load(motion_data_path)
motion = torch.tensor(motion).float()
# [帧数][每帧的参数]
motion_parms = {
            'root_orient': motion[:, :3],  # controls the global root orientation
            'pose_body': motion[:, 3:3+63],  # controls the body
            'pose_hand': motion[:, 66:66+90],  # controls the finger articulation
            'pose_jaw': motion[:, 66+90:66+93],  # controls the yaw pose
            'face_expr': motion[:, 159:159+50],  # controls the face expression
            'face_shape': motion[:, 209:209+100],  # controls the face shape
            'trans': motion[:, 309:309+3],  # controls the global body position
            'betas': motion[:, 312:],  # controls the body shape. Body shape is static
        }
print(motion_parms)
# read text labels
#semantic_text = np.loadtxt('semantic_labels/000001.npy')     # semantic labels 