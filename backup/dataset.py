import torch
from torch import nn
import os
import cv2
import numpy as np
import albumentations as A
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

class WgisdDataset(Dataset):
    def __init__(self, data_path, img_transform=None) -> None:
        super(WgisdDataset, self).__init__()
        self.data_path = data_path
        self.img_transform = img_transform
        self.img_path = os.path.join(data_path, 'images')
        self.ann_path = os.path.join(data_path, 'annotations')
        self.img_list = os.listdir(self.img_path)


    def __len__(self):
        return len(self.img_list)

    #TODO changeable sigma
    def _create_heatmap(self, points, heatmap_size=(256, 256)):
        sigma=1
        scale = 8 # 2048 / 8 = 256
        # 检查 sigma 是否是 torch.Tensor 类型
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.ones(len(points)) * sigma

        # 缩放点坐标
        points = points / scale
        points = torch.tensor(points, dtype=torch.float32)

        # 生成网格坐标
        x = torch.arange(0, heatmap_size[0], 1)
        y = torch.arange(0, heatmap_size[1], 1)
        x, y = torch.meshgrid(x, y, indexing='xy')
        x, y = x.unsqueeze(0), y.unsqueeze(0)

        heatmaps = torch.zeros(1, 1, heatmap_size[0], heatmap_size[1])

        # 计算每个点的高斯热力图并合并
        for indices in torch.arange(len(points)):
            mu_x, mu_y = points[indices, 0].view(-1, 1, 1), points[indices, 1].view(-1, 1, 1)
            heatmaps_ = torch.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma[indices].view(-1, 1, 1) ** 2))
            heatmaps_ = torch.max(heatmaps_, dim=0).values
            heatmaps_ = heatmaps_.reshape(1, 1, heatmap_size[0], heatmap_size[1])
            heatmaps = torch.maximum(heatmaps, heatmaps_)
         # 删除不必要的维度
        heatmaps = heatmaps.squeeze(0)
        return heatmaps.float()

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.img_list[idx])
        img = Image.open(img_path)
        transform = A.Compose([
            A.LongestMaxSize(1024),
            A.PadIfNeeded(1024, 1024, border_mode=0, value=(0, 0, 0), position=A.PadIfNeeded.PositionType.TOP_LEFT),
        ])

        img = Image.fromarray(transform(image=np.array(img))['image']) # PIL image
        img = self.img_transform(img) #(3, 256, 256)

        ann_path = os.path.join(self.ann_path, self.img_list[idx].split('.')[0]+'-berries.txt')
        dot_ann = np.loadtxt(ann_path) # np: (n, 2)
        heatmap = self._create_heatmap(dot_ann) # [1, 256, 256]
        return img, heatmap


    def visualize(self, idx, mode='ann'):
        """_summary_
        visualize data: idx; ann - show annotations; heatmap - show heatmap
        Args:
            idx (_type_): _description_
            mode (str, optional): _description_. Defaults to 'ann'.
        """
        img_path = os.path.join(self.img_path, self.img_list[idx])
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        if mode == 'ann':
            ann_path = os.path.join(self.ann_path, self.img_list[idx].split('.')[0]+'-berries.txt')
            dot_ann = np.loadtxt(ann_path)
            plt.imshow(image)
            plt.scatter(dot_ann[:,0], dot_ann[:,1], color='r', s=1)
        elif mode == 'heatmap':
            img, heatmap = self.__getitem__(idx)

            assert img.shape == torch.Size([3, 256, 256]), "Resize to 256x256"

            heatmap = transforms.Resize((256, 256))(heatmap)  # Resize the heatmap to 256x256
            img = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Unnormalize
            img = np.clip(img, 0, 1)  # Clip to [0, 1]
            plt.imshow(img)
            plt.imshow(heatmap.squeeze().cpu(), alpha=0.4, cmap='hot')

        plt.axis('off')
        plt.show()


def main():
    data_path = 'data/berry_dataset/train'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    d = WgisdDataset(data_path, img_transform=transform)
    print(d[1][1].shape)
    d.visualize(1, 'heatmap')
    dd = DataLoader(d, batch_size=8, shuffle=True, num_workers=8)
    for img, heatmap in dd:
        print(img.shape, heatmap.shape)


if __name__ == "__main__":
    main()