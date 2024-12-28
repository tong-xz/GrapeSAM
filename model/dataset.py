import random
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch

# import albumentations as A
import cv2
from torch.utils.data import DataLoader
from .utils import visualize_img_and_heatmap, visualize_quadrants

# from .util import restore_image_from_quadrants, visualize_restored_image


def _split_phases(root_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Define filenames for Train; Test; Validation phases and store in three respective .txt files
    @param folder: root directory of the dataset
    @return names in list without suffix
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "ratio sum must be 1"
    print(f"---Split dataset: train-{train_ratio}; val-{val_ratio}; test-{test_ratio}")

    img_dir = os.path.join(root_dir, "imgs")
    all_files = os.listdir(img_dir)
    all_files = [
        os.path.splitext(file)[0]
        for file in all_files
        if os.path.isfile(os.path.join(img_dir, file))
    ]
    random.shuffle(all_files)

    total_files = len(all_files)
    train_split_index = int(total_files * train_ratio)
    val_split_index = train_split_index + int(total_files * val_ratio)

    train_files = all_files[:train_split_index]
    val_files = all_files[train_split_index:val_split_index]
    test_files = all_files[val_split_index:]

    # create and write list in .txt files
    txt_file_lists = {
        "train.txt": train_files,
        "val.txt": val_files,
        "test.txt": test_files,
    }
    for k, v in txt_file_lists.items():
        txt_path = os.path.join(root_dir, k)
        with open(txt_path, "w") as f:
            for item in v:
                f.write(f"{item}\n")

    return train_files, val_files, test_files


def _read_phases(root_dir):
    """
    Reads train.txt, val.txt, and test.txt from the root directory and returns three lists.

    @param root_dir: The root directory where train.txt, val.txt, and test.txt are located.
    @return: Three lists representing the contents of train.txt, val.txt, and test.txt, respectively.
    """

    def read_txt_to_list(file_path):
        try:
            with open(file_path, "r") as f:
                return [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            print(f"Error: The file {file_path} does not exist.")
        except Exception as e:
            print(f"An error occurred while reading the file {file_path}: {e}")
        return []

    split_files = ["train.txt", "val.txt", "test.txt"]
    return [read_txt_to_list(os.path.join(root_dir, file)) for file in split_files]


def _convert(img, keypoints, target_size=(2048, 2048)):
    """
    Resize the image tensor while maintaining aspect ratio, pad to target size, and adjust keypoints accordingly.

    :param img: Tensor image of shape (C, H, W)
    :param keypoints: Numpy array of shape (N, 2), where N is the number of keypoints and each keypoint is (x, y)
    :param target_size: Desired output size as a tuple (target_height, target_width)
    :return: Resized and padded image tensor, adjusted keypoints numpy array
    """

    _, original_height, original_width = img.shape

    # Calculate the scale factor while maintaining aspect ratio
    scale = min(target_size[0] / original_height, target_size[1] / original_width)

    # Calculate new size
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)

    img = F.interpolate(
        img.unsqueeze(0),
        size=(new_height, new_width),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    keypoints = keypoints * scale

    # Calculate padding
    pad_height = (target_size[0] - new_height) // 2
    pad_width = (target_size[1] - new_width) // 2

    # Apply padding to the image
    img = F.pad(
        img,
        (
            pad_width,
            target_size[1] - new_width - pad_width,
            pad_height,
            target_size[0] - new_height - pad_height,
        ),
    )

    keypoints[:, 0] += pad_width  # Adjust x coordinates
    keypoints[:, 1] += pad_height  # Adjust y coordinates

    return img, keypoints


def random_crop(img, keypoints, crop_size=(1024, 1024)):
    """
    Randomly crop the image and keypoints to a specified size.

    :param img: Tensor image of shape (C, H, W)
    :param keypoints: Numpy array of shape (N, 2), where N is the number of keypoints and each keypoint is (x, y)
    :param crop_size: Tuple (height, width) specifying the size of the random crop
    :return: Cropped image tensor, adjusted keypoints numpy array
    """

    _, original_height, original_width = img.shape
    crop_height, crop_width = crop_size

    # Ensure crop size is not larger than the original image size
    assert (
        crop_height <= original_height and crop_width <= original_width
    ), "Crop size must be smaller than image size"

    # Randomly choose top-left corner for the crop
    top = random.randint(0, original_height - crop_height)
    left = random.randint(0, original_width - crop_width)

    # Crop the image
    img = img[:, top : top + crop_height, left : left + crop_width]

    # Adjust keypoints based on the crop
    keypoints[:, 0] -= left  # Adjust x coordinates
    keypoints[:, 1] -= top  # Adjust y coordinates

    # Remove keypoints that are outside the crop
    valid_indices = (
        (keypoints[:, 0] >= 0)
        & (keypoints[:, 0] <= crop_width)
        & (keypoints[:, 1] >= 0)
        & (keypoints[:, 1] <= crop_height)
    )
    keypoints = keypoints[valid_indices]

    return img, keypoints


def quad_crop(img, crop_size=(1024, 1024)):
    """
    Split the image into four non-overlapping 1024x1024 crops.

    :param img: Tensor image of shape (C, H, W)
    :param crop_size: Tuple (height, width) specifying the size of each crop (default is 1024x1024)
    :return: A dictionary of 4 cropped images
    """
    _, original_height, original_width = img.shape
    crop_height, crop_width = crop_size

    # Ensure the image is 2048x2048 as expected
    assert (
        original_height == 2048 and original_width == 2048
    ), "Source image must be 2048x2048"

    # Split the image into 4 crops: top-left, top-right, bottom-left, bottom-right
    crops = {
        "1": img[:, :crop_height, :crop_width],  # Top-left
        "2": img[:, :crop_height, crop_width:],  # Top-right
        "3": img[:, crop_height:, :crop_width],  # Bottom-left
        "4": img[:, crop_height:, crop_width:],  # Bottom-right
    }
    return crops


# TODO  Sigma value is proper?
def _create_heatmap(
    points, img_size, heatmap_size=(256, 256), sigma=1.0, normalize=True
):
    """
    Generate a heatmap for crowd counting tasks.

    :param points: Array of points (N, 2)
    :param img_size: Size of the original image (height, width)
    :param heatmap_size: Size of the heatmap (height, width)
    :param sigma: Standard deviation for Gaussian kernel
    :param normalize: Whether to normalize the heatmap
    :return: Heatmap tensor
    """
    assert (
        isinstance(img_size, tuple) and img_size[0] == img_size[1]
    ), "img_size type should be tuple and square shape"
    scale = img_size[0] / heatmap_size[0]  # 2048/256 = 8

    # Create an empty heatmap
    heatmap = torch.zeros(1, 1, heatmap_size[0], heatmap_size[1])

    # If there are no points, return the empty heatmap
    if len(points) == 0:
        return heatmap.squeeze(1).float()

    if not isinstance(sigma, torch.Tensor):
        sigma = torch.ones(len(points)) * sigma

    points = points / scale
    points = torch.tensor(points, dtype=torch.float32)

    x = torch.arange(0, heatmap_size[0], 1)
    y = torch.arange(0, heatmap_size[1], 1)
    x, y = torch.meshgrid(x, y, indexing="xy")
    x, y = x.unsqueeze(0), y.unsqueeze(0)

    for i in range(len(points)):
        mu_x, mu_y = points[i, 0].view(-1, 1, 1), points[i, 1].view(-1, 1, 1)
        heatmap_ = torch.exp(
            -((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma[i].view(-1, 1, 1) ** 2)
        )
        heatmap_ = heatmap_.reshape(1, 1, heatmap_size[0], heatmap_size[1])
        heatmap += heatmap_

    if normalize and heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap = heatmap.squeeze(1)

    return heatmap.float()


def _create_heatmap_dsigma(points, img_size, heatmap_size=(256, 256), normalize=True):
    """
    Generate a heatmap for crowd counting tasks with dynamic sigma based on nearest neighbor distances.

    :param points: Array of points (N, 2)
    :param img_size: Size of the original image (height, width)
    :param heatmap_size: Size of the heatmap (height, width)
    :param normalize: Whether to normalize the heatmap
    :return: Heatmap tensor
    """
    assert (
        isinstance(img_size, tuple) and img_size[0] == img_size[1]
    ), "img_size type should be tuple and square shape"

    scale = img_size[0] / heatmap_size[0]  # 例如 2048/256 = 8，用于缩放点坐标

    # 创建一个空的 heatmap
    heatmap = torch.zeros(1, 1, heatmap_size[0], heatmap_size[1])

    # 如果没有任何点，返回空的 heatmap
    if len(points) == 0:
        return heatmap.squeeze(1).float()

    # 缩放点坐标
    points = points / scale
    points = torch.tensor(points, dtype=torch.float32)

    # 使用 torch 生成网格
    x = torch.arange(0, heatmap_size[0], 1)
    y = torch.arange(0, heatmap_size[1], 1)
    x, y = torch.meshgrid(x, y, indexing="xy")
    x, y = x.unsqueeze(0), y.unsqueeze(0)

    # 计算动态 sigma，基于最近邻距离
    distances = torch.cdist(points, points, p=2)  # 欧几里得距离矩阵，形状 (N, N)
    distances.fill_diagonal_(float("inf"))  # 排除点自身
    nearest_distances, _ = torch.min(distances, dim=1)  # 每个点的最近邻距离
    sigma = nearest_distances / 3.0  # 将最近邻距离作为 sigma 值

    # 对每个点生成高斯分布并叠加到 heatmap
    for i in range(len(points)):
        mu_x, mu_y = points[i, 0].view(-1, 1, 1), points[i, 1].view(-1, 1, 1)
        heatmap_ = torch.exp(
            -((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma[i].view(-1, 1, 1) ** 2)
        )
        heatmap_ = heatmap_.reshape(1, 1, heatmap_size[0], heatmap_size[1])
        heatmap += heatmap_

    # 如果要求 normalize 且 heatmap 的最大值大于 0，进行归一化处理
    if normalize and heatmap.max() > 0:
        heatmap /= heatmap.max()

    # 移除多余的维度并返回 heatmap
    heatmap = heatmap.squeeze(1)

    return heatmap.float()


class VividDataset(Dataset):
    def __init__(self, data_root, file_list, mode="train") -> None:
        super(VividDataset, self).__init__()
        self.data_root = data_root
        self.img_path = os.path.join(data_root, "imgs")
        self.ann_path = os.path.join(data_root, "anns")
        self.file_list = file_list
        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # read img and npy
        item_name = self.file_list[index]

        img_file_path, dot_ann_path = os.path.join(
            self.img_path, item_name
        ), os.path.join(self.ann_path, item_name.split(".")[0] + ".npy")
        img = Image.open(img_file_path).convert("RGB")
        if self.img_transform:
            img = self.img_transform(img)
        keypoints = np.load(dot_ann_path)  # np.ndarray: (n, 2)

        target_img_size = (1024, 1024)  # ViT can only take (1024, 1024) image
        img, keypoints = _convert(
            img, keypoints, target_img_size
        )  # resize to target size
        heatmap = _create_heatmap(keypoints, img_size=target_img_size, sigma=1)
        # heatmap = _create_heatmap_dsigma(
        #     keypoints, img_size=target_img_size
        # )  # (1, 256, 256) make heatmap based on the img and points

        if self.mode == "train":
            del keypoints
            return img, heatmap

        else:
            # reverse normalize
            inv_transform = transforms.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            )
            img = inv_transform(img)

            # Convert point count to float32
            point_num = torch.tensor(len(keypoints), dtype=torch.float32)
            return img, point_num


def build_loader(root_dir, batch_size, phase="train"):
    """
    the only function exposed to the outer class to build dataloaders

    @param ROOT_DIR root dir of the entire dataset
    @param BATCH_SIZE
    @return dict: three respective dataloaders
    """

    # only initiate once
    if not os.path.exists(os.path.join(root_dir, "train.txt")):
        print
        train_files, val_files, test_files = _split_phases(root_dir)
    else:
        train_files, val_files, test_files = _read_phases(root_dir)

    train_dataset, val_dataset, test_dataset = (
        VividDataset(root_dir, train_files, mode="train"),  # loss
        VividDataset(root_dir, val_files, mode="val"),  # loss
        VividDataset(root_dir, test_files, mode="test"),  # metric mse/msn
    )

    # loader
    train_loader = DataLoader(
        train_dataset, batch_size, num_workers=31, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size, num_workers=31)
    test_loader = DataLoader(test_dataset, batch_size, num_workers=31)
    return {"train": train_loader, "val": val_loader, "test": test_loader}
