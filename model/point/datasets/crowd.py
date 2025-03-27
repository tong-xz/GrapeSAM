from PIL import Image
import PIL
import torch.utils.data as data
import os
import scipy.io as io
from glob import glob
import json
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import math


def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right - inner_left, 0.0) * np.maximum(
        inner_down - inner_up, 0.0
    )
    return inner_area

def find_dis(point):
    square = np.sum(point * point, axis=1)
    dis = np.sqrt(
        np.maximum(
            square[:, None] - 2 * np.matmul(point, point.T) + square[None, :], 0.0
        )
    )
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis


def get_im_list(root_path, json_file):
    with open(json_file) as f:
        im_list = json.load(f)
    im_list = [os.path.join(root_path, x.split("/")[-1]) for x in im_list]
    return im_list

def get_img_list_txt(root_path, txt_file):
    with open(os.path.join(root_path, txt_file)) as f:
        im_list = f.readlines()
    im_list = [os.path.join(root_path, "imgs", x.strip()) for x in im_list]
    return im_list

def train_val(im_list, ratio=0.9):
    N = int(float(len(im_list)) * ratio)
    idx = torch.randperm(len(im_list))
    train_list = [im_list[i] for i in idx[0:N]]
    val_list = [im_list[i] for i in idx[N + 1 :]]
    return train_list, val_list


class Crowd(data.Dataset):
    def __init__(
        self,
        root_path,
        crop_size,
        downsample_ratio,
        method="train",
        resize=False,
        im_list=None,
        noise=0,
        use_file_list=False,
    ):

        self.noise = noise
        self.root_path = root_path
        self.resize = resize
        if im_list is None:
            if use_file_list:    
                if method == "train":
                    self.im_list = get_img_list_txt(root_path, "train.txt")
                elif method == "val":
                    self.im_list = get_img_list_txt(root_path, "val.txt")
                elif method == "test":
                    self.im_list = get_img_list_txt(root_path, "test.txt")
                else:
                    raise Exception("not implement")
                
            else: 
                all_imgs = os.listdir(os.path.join(root_path, "imgs"))
                # use first 80% as training set, the rest as validation set
                if method == "train":
                    self.im_list = [os.path.join(root_path, "imgs", x) for x in all_imgs[:int(len(all_imgs) * 0.8)]]
                elif method == "val":
                    self.im_list = [os.path.join(root_path, "imgs", x) for x in all_imgs[int(len(all_imgs) * 0.8):]]

        else:
            self.im_list = im_list

        # check all files in the list exist, if not, just skip
        for im_path in self.im_list:
            if not os.path.exists(im_path):
                print(f"Skip {im_path}")
                self.im_list.remove(im_path)

        self.method = method
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return 1 * len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item % len(self.im_list)]
        gd_path = img_path.replace("imgs", "anns/points")
        # relpace the extension to npy
        gd_path = gd_path.split(".")[0] + ".npy"
        img = Image.open(img_path).convert("RGB")
        keypoints = np.load(gd_path)

        if self.method == "train":
            return self.train_transform_with_crop(img, keypoints)
        elif self.method == "val":
            img, keypoints
            img = self.trans(img)
            name = os.path.basename(img_path).split(".")[0]
            if len(keypoints) == 0:
                keypoints = torch.zeros(size=(1, 1))
            return img, keypoints, name

    def train_transform_with_crop(self, img, keypoints):
        """random crop image patch and find people in it"""
        wd, ht = img.size
        st_size = min(wd, ht)
        if st_size < self.c_size:
            c_size = 512
        else:
            c_size = self.c_size
        assert st_size >= self.c_size
        i, j, h, w = random_crop(ht, wd, c_size, c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) < 1:
            if random.random() > 0.5:
                img = F.hflip(img)
            return (
                self.trans(img),
                torch.from_numpy(keypoints.copy()).float(),
                torch.from_numpy(keypoints.copy()).float(),
                st_size,
            )
        # nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)
        # calculate the distance between the center of the person and the nearest edge
        nearest_dis = np.clip(find_dis(keypoints).squeeze(1), 4.0, 128.0)
        points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
        points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
        bbox = np.concatenate((points_left_up, points_right_down), axis=1)
        inner_area = cal_innner_area(j, i, j + w, i + h, bbox)
        origin_area = nearest_dis * nearest_dis
        ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
        mask = ratio >= 0.3

        target = ratio[mask]
        keypoints = keypoints[mask]
        keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
        return (
            self.trans(img),
            torch.from_numpy(keypoints.copy()).float(),
            torch.from_numpy(target.copy()).float(),
            st_size,
        )
