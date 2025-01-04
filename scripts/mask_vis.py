import json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import numpy as np
import cv2
import random
import os


def visualize_coco_masks(annotation_file, image_name):
    """
    Visualize COCO masks for a given image name with different colors.

    Args:
        annotation_file (str): Path to the COCO annotation JSON file.
        image_name (str): Name of the image file to visualize.
    """
    # Load the COCO data
    coco = COCO(annotation_file)

    # Find image ID by file name
    for img in coco.dataset["images"]:
        if img["file_name"] == image_name:
            image_id = img["id"]
            break
    else:
        raise ValueError(f"Image '{image_name}' not found in the dataset")

    # Load image information
    image_info = coco.loadImgs(image_id)[0]
    height, width = image_info["height"], image_info["width"]

    # Load the original image with better error handling
    img_dir = "/home/xz/Documents/Vivid/imgs"  # Updated image directory
    image_path = os.path.join(img_dir, image_name)

    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Try to load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize an empty mask with alpha channel
    mask = np.zeros((height, width, 4), dtype=np.uint8)  # Changed to 4 channels (RGBA)

    # Load annotations for the image
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)

    # Assign random colors to each annotation with alpha
    for annotation in annotations:
        if "segmentation" in annotation:
            segmentation = annotation["segmentation"]
            color = [random.randint(0, 255) for _ in range(3)] + [
                127
            ]  # Random RGB color with 50% alpha

            if isinstance(segmentation, list):  # Polygon format
                for seg in segmentation:
                    poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [poly], color)
            elif isinstance(segmentation, dict):  # RLE format
                rle = mask_utils.frPyObjects(segmentation, height, width)
                binary_mask = mask_utils.decode(rle)
                mask[binary_mask > 0] = color  # Apply color to the mask

    # Blend the mask with the original image
    mask_rgb = mask[:, :, :3].astype(float) / 255
    mask_alpha = mask[:, :, 3:].astype(float) / 255
    image = image.astype(float) / 255

    blended = image * (1 - mask_alpha) + mask_rgb * mask_alpha
    blended = (blended * 255).astype(np.uint8)

    # Display the blended result
    plt.figure(figsize=(12, 8))
    plt.imshow(blended)
    plt.axis("off")
    plt.title(f"COCO Masks for Image ID: {image_id}")
    plt.show()


# Example usage
annotation_file = "/home/xz/Documents/Vivid/ann.json"
image_name = "1.png"  # Replace with your image file name
visualize_coco_masks(annotation_file, image_name)
