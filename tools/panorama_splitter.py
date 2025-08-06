'''
This code transfer panorama to 6 perspective images
'''

import numpy as np
import cv2
import os

def dir_to_equirectangular_uv(dir):
    x, y, z = dir
    theta = np.arctan2(x, z)
    phi = np.arcsin(y)
    u = (theta + np.pi) / (2 * np.pi)
    v = (np.pi/2 - phi) / np.pi
    return u, v

def generate_face(face_size, face_orientation, panorama_img):
    height, width, _ = panorama_img.shape
    face_img = np.zeros((face_size, face_size, 3), dtype=np.uint8)
    for y in range(face_size):
        for x in range(face_size):
            nx = (2 * (x + 0.5) / face_size) - 1
            ny = (2 * (y + 0.5) / face_size) - 1
            if face_orientation == '+X': dir = np.array([1, -ny, -nx])
            elif face_orientation == '-X': dir = np.array([-1, -ny, nx])
            elif face_orientation == '+Y': dir = np.array([nx, 1, ny])
            elif face_orientation == '-Y': dir = np.array([nx, -1, -ny])
            elif face_orientation == '+Z': dir = np.array([nx, -ny, 1])
            elif face_orientation == '-Z': dir = np.array([-nx, -ny, -1])
            else: raise ValueError("Invalid face orientation")
            dir = dir / np.linalg.norm(dir)
            u, v = dir_to_equirectangular_uv(dir)
            px = int(u * width) % width
            py = int(v * height) % height
            face_img[y, x] = panorama_img[py, px]
    return face_img

def extract_cubemap_faces(panorama_path, face_size=512, output_folder="cubemap_faces"):
    panorama_img = cv2.imread(panorama_path)
    if panorama_img is None:
        print(f"无法加载图像: {panorama_path}")
        return

    os.makedirs(output_folder, exist_ok=True)

    # 6个面，key为你的命名，value对应face_orientation
    face_map = {
        'right': '+X',
        'left': '-X',
        'up': '+Y',
        'down': '-Y',
        'front': '+Z',
        'back': '-Z'
    }

    face_imgs = {}
    for name, orientation in face_map.items():
        face_imgs[name] = generate_face(face_size, orientation, panorama_img)
        cv2.imwrite(os.path.join(output_folder, f"{name}.png"), face_imgs[name])
        print(f"保存 {name}.png")

    # 创建展开图，布局3行4列，尺寸如下：
    #  3 行，4 列
    #  但实际上有6个面，我们按如下布局拼：
    #      第1行:      [  空  ,  up  ,  空  ,  空  ]
    #      第2行:      [ left , front, right, back ]
    #      第3行:      [  空  , down ,  空  ,  空  ]

    W, H = face_size, face_size
    combined_width = W * 4
    combined_height = H * 3
    combined_img = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

    # 放置up
    combined_img[0:H, W:2*W] = face_imgs['up']
    # 放置left, front, right, back
    combined_img[H:2*H, 0:W] = face_imgs['left']
    combined_img[H:2*H, W:2*W] = face_imgs['front']
    combined_img[H:2*H, 2*W:3*W] = face_imgs['right']
    combined_img[H:2*H, 3*W:4*W] = face_imgs['back']
    # 放置down
    combined_img[2*H:3*H, W:2*W] = face_imgs['down']

    combined_path = os.path.join(output_folder, "cubemap_unfold.png")
    cv2.imwrite(combined_path, combined_img)
    print(f"保存展开图 {combined_path}")

# 示例调用
extract_cubemap_faces(r"E:\softwares_document\VS_Code_Projects\python_Project\FreeWorldSimulator\\2.png", face_size=2048)
