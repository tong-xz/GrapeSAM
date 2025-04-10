from torch.utils.data import Dataset
import os
from PIL import Image
import json

import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import cv2


def show_mask(mask, ax, random_color=False, alpha=0.6):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, alpha])
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


def show_masks_on_image(raw_image, masks, random_colors=True, alpha=0.9):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    ax = plt.gca()

    if isinstance(masks, np.ndarray):  # Direct binary masks
        for mask in masks:
            show_mask(mask, ax, random_color=random_colors, alpha=alpha)
    else:
        raise ValueError("Unsupported mask format. Masks should be a NumPy array.")

    plt.axis("on")
    plt.show()


def update_bbox(item):

    gt_masks = [item["segmentation"] for item in annotations]

    gt_bboxes = [item["bbox"] for item in annotations]


def contains_empty_list(cascaded_list):
    for element in cascaded_list:
        if isinstance(element, list) and not element:
            return True
    return False


if __name__ == "__main__":
    data_root = "/home/xz/Documents/Vivid"

    json_path = os.path.join(data_root, "instances_updated.json")
    coco = COCO(json_path)

    # Get all image IDs
    img_ids = coco.getImgIds()

    # Load image metadata
    images = coco.loadImgs(img_ids)

    # Iterate over each image
    for img_info in images:
        # Get annotation IDs for the current image
        ann_ids = coco.getAnnIds(imgIds=img_info["id"])
        annotations = coco.loadAnns(ann_ids)

    
