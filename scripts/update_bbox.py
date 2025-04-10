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


def update_bbox(mask):
    gt_bboxes = []
    if len(mask) > 0:
        # Convert polygon to points array
        points = np.concatenate([np.array(polygon).reshape(-1, 2) for polygon in mask])
        # Get min/max coordinates
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        gt_bboxes.append([x_min, y_min, x_max, y_max])

    return gt_bboxes


if __name__ == "__main__":
    data_root = "/home/xz/Dev/GrapeSAM/data/vivid"
    json_path = os.path.join(data_root, "instances_default_v4.json")
    img_path = os.path.join(data_root, "imgs")

    coco = COCO(json_path)
    img_ids = coco.getImgIds()

    # 加载 COCO 文件
    coco = COCO(json_path)
    img_ids = coco.getImgIds()

    # 创建一个新的 COCO 格式的字典
    updated_coco_data = {
        "info": coco.dataset.get("info", {}),
        "licenses": coco.dataset.get("licenses", []),
        "images": coco.dataset.get("images", []),
        "categories": coco.dataset.get("categories", []),
        "annotations": [],
    }

    # 遍历所有图像和注释
    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        annotations = coco.loadAnns(ann_ids)

        for ann in annotations:
            # 更新 bbox
            ann["bbox"] = update_bbox(ann["segmentation"])
            # 将更新后的注释添加到新的 COCO 数据中
            updated_coco_data["annotations"].append(ann)

    # 保存更新后的 COCO 文件
    updated_json_path = os.path.join(data_root, "instances_updated.json")
    with open(updated_json_path, "w") as f:
        json.dump(updated_coco_data, f, indent=4)

    print(f"Updated COCO annotations saved to {updated_json_path}")
