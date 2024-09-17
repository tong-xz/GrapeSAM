import random
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch
import albumentations as A
import cv2
from torch.utils.data import DataLoader
from .util import visualize_img_and_heatmap, visualize_quadrants

# from .util import restore_image_from_quadrants, visualize_restored_image


def _split_phases(root_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Define filenames for Train; Test; Validation phases and store in three respective .txt files
    @param folder: root directory of the dataset
    @return names in list without suffix
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "ratio sum must be 1"
    print(f"---Split dataset: train-{train_ratio}; val-{val_ratio}; test-{test_ratio}")

    img_dir = os.path.join(root_dir, "images")
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


class VividDataset(Dataset):
    def __init__(
        self, data_root, file_list, mode="train", use_random_crop=False
    ) -> None:
        super(VividDataset, self).__init__()
        self.data_root = data_root
        self.img_path = os.path.join(data_root, "images")
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
        self.use_random_crop = use_random_crop

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # read img and npy
        item_name = self.file_list[index]
        img_file_path, dot_ann_path = os.path.join(
            self.img_path, item_name + ".png"
        ), os.path.join(self.ann_path, item_name + ".npy")
        img = Image.open(img_file_path).convert("RGB")
        if self.img_transform:
            img = self.img_transform(img)
        keypoints = np.load(dot_ann_path)  # np.ndarray: (n, 2)

        # random crop images
        if self.use_random_crop:
            img, keypoints = _convert(
                img, keypoints, target_size=(2048, 2048)
            )  # maybe larger img size
            
            if self.mode == "train":
                img, keypoints = random_crop(img, keypoints, crop_size=(1024, 1024))
                heatmap = _create_heatmap(keypoints, img_size=(1024, 1024), sigma=1)
                
                return img, heatmap

            elif self.mode == "test":
                img_dict = quad_crop(img)
                return img_dict, keypoints

            else:
                raise NotImplementedError("Please use right mode code")

        # use original whole image
        else:
            target_img_size = (1024, 1024)  # ViT can only take (1024, 1024) image
            img, keypoints = _convert(
                img, keypoints, target_img_size
            )  # resize to target size
            heatmap = _create_heatmap(
                keypoints, img_size=target_img_size, sigma=1
            )  # (1, 256, 256) make heatmap based on the img and points
            if self.mode == "train":
                del keypoints
                return img, heatmap

            elif self.mode == "test":
                #reverse normalize
                inv_transform = transforms.Normalize(
                    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
                )
                img = inv_transform(img)
                # there should be the number of points
                point_num = len(keypoints)
                return img, point_num

            else:
                raise NotImplementedError("Please use right mode code")


def build_loader(root_dir, batch_size, use_rcrop):
    """
    the only function exposed to the outer class to build dataloaders

    @param ROOT_DIR root dir of the entire dataset
    @param BATCH_SIZE
    @return dict: three respective dataloaders
    """
    train_files, val_files, test_files = _split_phases(root_dir)
    train_dataset, val_dataset, test_dataset = (
        VividDataset(root_dir, train_files, mode="train", use_random_crop=use_rcrop),  # loss
        VividDataset(root_dir, val_files, mode="train", use_random_crop=use_rcrop),  # loss
        VividDataset(root_dir, test_files, mode="test", use_random_crop=use_rcrop),  # metric mse/msn
    )

    # loader
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    return {"train": train_loader, "val": val_loader, "test": test_loader}



class WgisdDataset(Dataset):
    def __init__(self, data_path, img_transform=None) -> None:
        super(WgisdDataset, self).__init__()
        self.data_path = data_path
        self.img_transform = img_transform
        self.img_path = os.path.join(data_path, "images")
        self.ann_path = os.path.join(data_path, "annotations")
        self.img_list = os.listdir(self.img_path)
        self.transform = A.Compose(
            [
                A.LongestMaxSize(1024),
                A.PadIfNeeded(
                    1024,
                    1024,
                    border_mode=0,
                    value=(0, 0, 0),
                    position=A.PadIfNeeded.PositionType.TOP_LEFT,
                ),
            ]
        )

    def __len__(self):
        return len(self.img_list)

    # TODO changeable sigma
    # def _create_heatmap(self, points, heatmap_size=(256, 256)):
    #     sigma = 1
    #     scale = 8  # 2048 / 8 = 256
    #     # 检查 sigma 是否是 torch.Tensor 类型
    #     if not isinstance(sigma, torch.Tensor):
    #         sigma = torch.ones(len(points)) * sigma

    #     # 缩放点坐标
    #     points = points / scale
    #     points = torch.tensor(points, dtype=torch.float32)

    #     # 生成网格坐标
    #     x = torch.arange(0, heatmap_size[0], 1)
    #     y = torch.arange(0, heatmap_size[1], 1)
    #     x, y = torch.meshgrid(x, y, indexing="xy")
    #     x, y = x.unsqueeze(0), y.unsqueeze(0)

    #     heatmaps = torch.zeros(1, 1, heatmap_size[0], heatmap_size[1])

    #     # 计算每个点的高斯热力图并合并
    #     for indices in torch.arange(len(points)):
    #         mu_x, mu_y = points[indices, 0].view(-1, 1, 1), points[indices, 1].view(
    #             -1, 1, 1
    #         )
    #         heatmaps_ = torch.exp(
    #             -((x - mu_x) ** 2 + (y - mu_y) ** 2)
    #             / (2 * sigma[indices].view(-1, 1, 1) ** 2)
    #         )
    #         heatmaps_ = torch.max(heatmaps_, dim=0).values
    #         heatmaps_ = heatmaps_.reshape(1, 1, heatmap_size[0], heatmap_size[1])
    #         heatmaps = torch.maximum(heatmaps, heatmaps_)
    #     # 删除不必要的维度
    #     heatmaps = heatmaps.squeeze(0)
    #     return heatmaps.float()

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.img_list[idx])
        img = Image.open(img_path)

        img = Image.fromarray(transform(image=np.array(img))["image"])  # PIL image
        img = self.img_transform(img)  # (3, 256, 256)

        ann_path = os.path.join(
            self.ann_path, self.img_list[idx].split(".")[0] + "-berries.txt"
        )
        dot_ann = np.loadtxt(ann_path)  # np: (n, 2)
        heatmap = self._create_heatmap(dot_ann)  # [1, 256, 256]
        return img, heatmap

    def visualize(self, idx, mode="ann"):
        """_summary_
        visualize data: idx; ann - show annotations; heatmap - show heatmap
        Args:
            idx (_type_): _description_
            mode (str, optional): _description_. Defaults to 'ann'.
        """
        img_path = os.path.join(self.img_path, self.img_list[idx])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        if mode == "ann":
            ann_path = os.path.join(
                self.ann_path, self.img_list[idx].split(".")[0] + "-berries.txt"
            )
            dot_ann = np.loadtxt(ann_path)
            plt.imshow(image)
            plt.scatter(dot_ann[:, 0], dot_ann[:, 1], color="r", s=1)
        elif mode == "heatmap":
            img, heatmap = self.__getitem__(idx)

            assert img.shape == torch.Size([3, 256, 256]), "Resize to 256x256"

            heatmap = transforms.Resize((256, 256))(
                heatmap
            )  # Resize the heatmap to 256x256
            img = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array(
                [0.485, 0.456, 0.406]
            )  # Unnormalize
            img = np.clip(img, 0, 1)  # Clip to [0, 1]
            plt.imshow(img)
            plt.imshow(heatmap.squeeze().cpu(), alpha=0.4, cmap="hot")

        plt.axis("off")
        plt.show()



if __name__ == "__main__":
    root = "/home/xz/Dev/Dream/data/vivid/"
    train_files, val_files, test_files = _split_phases(root)
    v = VividDataset(
        "/home/xz/Dev/Dream/data/vivid",
        file_list=train_files,
        mode="train",
        use_random_crop=True,
    )

    # for i, data in enumerate(v):
    #     img, map = data[0], data[1]
        
    #     visualize_img_and_heatmap(img, map)
        
    #     print(i)
    
    # # img = restore_image_from_quadrants(img)
    # # visualize_restored_image(img)
    # visualize_img_and_heatmap(img, map)
    # # visualize_quadrants(img)

    loader_dict = build_loader(root, 4, True)
    for imgs, heatmaps in loader_dict['train']:
        print(imgs.shape, heatmaps.shape)
