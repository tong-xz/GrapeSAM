from PIL import Image
import numpy as np
import os
import cv2
import argparse
from glob import glob
import matplotlib.pyplot as plt
import random


def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w * ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w * ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h * ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h * ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    square = np.sum(point * point, axis=1)
    dis = np.sqrt(
        np.maximum(
            square[:, None] - 2 * np.matmul(point, point.T) + square[None, :], 0.0
        )
    )
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis


def generate_data(im_path, points_path):
    im = Image.open(im_path)
    im_w, im_h = im.size
    points = np.load(points_path).astype(np.float32)
    idx_mask = (
        (points[:, 0] >= 0)
        * (points[:, 0] <= im_w)
        * (points[:, 1] >= 0)
        * (points[:, 1] <= im_h)
    )
    points = points[idx_mask]

    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    if rr != 1.0:
        im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
        points = points * rr
    return Image.fromarray(im), points


def process_folder(input_dir, output_dir, is_training=False):
    """处理指定文件夹中的所有图片和对应的npy文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取所有npy文件（用作参照）
    npy_files = glob(os.path.join(input_dir, "*.npy"))

    for npy_path in npy_files:
        base_name = os.path.splitext(os.path.basename(npy_path))[0]

        # 查找对应的图片文件
        img_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
        img_path = None
        for ext in img_extensions:
            potential_path = os.path.join(input_dir, base_name + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break

        if img_path is None:
            print(f"Skipping {base_name} - no corresponding image file found")
            continue

        print(f"Processing {base_name}")

        try:
            im, points = generate_data(img_path, npy_path)

            if is_training:
                dis = find_dis(points)
                points = np.concatenate((points, dis), axis=1)

            # 保存处理后的图片和点数据
            im_save_path = os.path.join(output_dir, f"{base_name}.jpg")
            gd_save_path = os.path.join(output_dir, f"{base_name}.npy")

            im.save(im_save_path)
            np.save(gd_save_path, points)

        except Exception as e:
            print(f"Error processing {base_name}: {str(e)}")
            continue


def visualize_random_sample(input_dir):
    """随机选择一个样本并可视化"""
    # 获取所有npy文件
    npy_files = glob(os.path.join(input_dir, "*.npy"))
    if not npy_files:
        print("No .npy files found!")
        return

    # 随机选择一个npy文件
    npy_path = random.choice(npy_files)
    base_name = os.path.splitext(os.path.basename(npy_path))[0]

    # 查找对应的图片文件
    img_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    img_path = None
    for ext in img_extensions:
        potential_path = os.path.join(input_dir, base_name + ext)
        if os.path.exists(potential_path):
            img_path = potential_path
            break

    if img_path is None:
        print(f"No image file found for {base_name}")
        return

    # 读取数据
    img = Image.open(img_path)
    points = np.load(npy_path)

    # 创建可视化
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1], c="red", s=1)
    plt.title("Original Image with Points")

    plt.subplot(122)
    plt.hist2d(points[:, 0], points[:, 1], bins=100, cmap="jet")
    plt.colorbar()
    plt.title("Points Density Map")

    plt.suptitle(f"Visualization for {base_name}")
    plt.tight_layout()

    # 直接显示而不保存
    plt.show()
    print(f"Total points in image: {len(points)}")
    return img_path, npy_path


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Dataset")
    parser.add_argument(
        "--origin-dir",
        default="/home/xz/Dev/Bayesian-Crowd-Counting/vivid",
        help="original data directory",
    )
    parser.add_argument(
        "--data-dir", default="./processed_data", help="processed data directory"
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["train", "test", "val"],
        help="folders to process",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="visualize a random sample"
    )
    parser.add_argument("--visualize-dir", type=str, help="directory for visualization")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    min_size = 512
    max_size = 2048

    # 如果需要可视化
    if args.visualize:
        visualize_dir = args.visualize_dir or os.path.join(args.origin_dir, "train")
        img_path, npy_path = visualize_random_sample(visualize_dir)
        print(f"Visualized files:\nImage: {img_path}\nPoints: {npy_path}")
        exit(0)

    # 处理指定的文件夹
    for folder in args.folders:
        print(f"\nProcessing {folder} folder...")
        input_dir = os.path.join(args.origin_dir, folder)
        output_dir = os.path.join(args.data_dir, folder)

        if not os.path.exists(input_dir):
            print(f"Skipping {folder} - directory not found")
            continue

        # 只在训练集上计算距离
        is_training = folder == "train"
        process_folder(input_dir, output_dir, is_training)
