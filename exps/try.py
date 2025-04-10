import os
import numpy as np

# 指定包含.npy文件的文件夹路径
folder_path = "/home/xz/Dev/baseline-exp-playground/DATASET/vivid-close/test"  # 请替换为你的实际文件夹路径

# 获取文件夹中所有的.npy文件
npy_files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]


file_count, point_count = 0, 0
# 遍历所有.npy文件并读取
for npy_file in npy_files:
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, npy_file)

    # 读取.npy文件
    data = np.load(file_path)

    file_count += 1
    point_count += data.shape[0]

# Calculate and print total files, total points, and average points per file
print(f"Total files: {file_count}")
print(f"Total points: {point_count}")
print(f"Average points per file: {point_count / file_count:.2f}")
