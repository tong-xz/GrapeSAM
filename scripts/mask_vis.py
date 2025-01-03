import json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import numpy as np
import cv2
import random
import os


def visualize_coco_masks(annotation_file, image_id):
    """
    Visualize COCO masks for a given image ID with different colors.

    Args:
        annotation_file (str): Path to the COCO annotation JSON file.
        image_id (int): ID of the image to visualize.
    """
    # Load the COCO data
    coco = COCO(annotation_file)

    # Load image information
    image_info = coco.loadImgs(image_id)[0]
    height, width = image_info["height"], image_info["width"]

    # Initialize an empty mask with RGB channels
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    # Load annotations for the image
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)

    # Assign random colors to each annotation
    for annotation in annotations:
        if "segmentation" in annotation:
            segmentation = annotation["segmentation"]
            color = [random.randint(0, 255) for _ in range(3)]  # Random RGB color

            if isinstance(segmentation, list):  # Polygon format
                for seg in segmentation:
                    poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [poly], color)
            elif isinstance(segmentation, dict):  # RLE format
                rle = mask_utils.frPyObjects(segmentation, height, width)
                binary_mask = mask_utils.decode(rle)
                mask[binary_mask > 0] = color  # Apply color to the mask

    # Display the combined mask
    plt.figure(figsize=(12, 8))
    plt.imshow(mask)
    plt.axis("off")
    plt.title(f"COCO Masks for Image ID: {image_id}")
    plt.show()


# Example usage
annotation_file = "/home/xz/Documents/Vivid/ann.json"  # Replace with your file path
image_id = 1  # Replace with your desired image ID
visualize_coco_masks(annotation_file, image_id)
