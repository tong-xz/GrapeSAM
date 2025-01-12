from torch.utils.data import Dataset
import os
from PIL import Image
import json

import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import cv2


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_boxes_on_image(raw_image, boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.show()


def show_masks_on_image(raw_image, masks, random_colors=False):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    ax = plt.gca()

    for mask in masks:
        # 处理COCO格式的分割标注
        if isinstance(mask, list):  # polygon format
            # 创建二值mask
            binary_mask = np.zeros(
                (raw_image.size[1], raw_image.size[0]), dtype=np.uint8
            )
            for polygon in mask:
                # 将多边形坐标转换为整数数组
                poly = np.array(polygon).reshape((-1, 2)).astype(np.int32)
                # 填充多边形区域
                binary_mask = cv2.fillPoly(binary_mask, [poly], 1)
            show_mask(binary_mask, ax, random_color=random_colors)

    plt.axis("on")
    plt.show()


class VividDataset(Dataset):
    def __init__(self, data_root, txt_path, json_path) -> None:
        super(VividDataset, self).__init__()
        self.data_root = data_root
        self.img_path = os.path.join(data_root, "imgs")
        self.file_list = open(txt_path, "r").read().splitlines()
        self.coco = COCO(json_path)
        self.file_to_id = {
            item["file_name"]: item["id"] for item in self.coco.dataset["images"]
        }

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        file_name = self.file_list[index]
        img_path = os.path.join(self.img_path, file_name)

        # Get annotations for this image
        id = self.file_to_id[file_name]
        ann_ids = self.coco.getAnnIds(imgIds=[id])
        annotations = self.coco.loadAnns(ann_ids)

        gt_masks = [item["segmentation"] for item in annotations]

        # Calculate bboxes from masks
        # TODO gt bboxes has problems, now use mask to calculate bboxes
        gt_bboxes = []
        for mask in gt_masks:
            # Convert polygon to points array
            points = np.concatenate(
                [np.array(polygon).reshape(-1, 2) for polygon in mask]
            )
            # Get min/max coordinates
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            gt_bboxes.append([x_min, y_min, x_max, y_max])

        return img_path, gt_bboxes, gt_masks


if __name__ == "__main__":
    data_root = "/home/xz/Dev/GrapeSAM/data/vivid"
    txt_path = os.path.join(data_root, "test.txt")
    json_path = os.path.join(data_root, "instances_default_v4.json")

    dataset = VividDataset(data_root, txt_path, json_path)
    img_path, bboxes, masks = dataset[0]
    img = Image.open(img_path).convert("RGB")

    show_masks_on_image(img, masks, random_colors=True)
    show_boxes_on_image(img, bboxes)
