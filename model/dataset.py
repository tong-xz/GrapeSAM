import torch
from torch import nn
import os
import cv2
import numpy as np

class WgisdDataset(nn.Module):
    def __init__(self, img_path, dot_ann_path) -> None:
        super(WgisdDataset, self).__init__()
        self.img_path = img_path
        self.dot_ann_path = dot_ann_path
        self.img_list = os.listdir(img_path)

    def __len__(self):
        return len(os.listdir(self.img_path))

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_path, self.img_list[idx])
        dot_ann_path = os.path.join(self.dot_ann_path, self.img_list[idx].split('.')[0]+'-berries.txt')
        img= np.load(img_path) # np: (1365, 2048, 3)
        dot_ann = np.loadtxt(dot_ann_path, dtype=np.uint8) # np: (n, 2)
        return img, dot_ann


def main():
    img_path = '/Users/tongxiangzhi/Dev/Dream/data/img_npy'
    dot_path = '/Users/tongxiangzhi/Dev/Dream/data/berries'
    d = WgisdDataset(img_path, dot_path)
    print(d[0])
    print(d[0][0].shape)
    print(d[0][1].shape)


if __name__ == "__main__":
    main()