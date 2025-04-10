# ... existing imports ...
import math
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO


def visualize_all_coco_masks(
    annotation_file, img_dir, output_dir, images_per_figure=25
):
    """
    Visualize all COCO masks in a grid format and save to output directory.

    Args:
        annotation_file (str): Path to COCO annotation file
        img_dir (str): Directory containing images
        output_dir (str): Directory to save visualization results
        images_per_figure (int): Number of images per figure
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load COCO dataset
    coco = COCO(annotation_file)
    all_images = coco.dataset["images"]

    # Calculate grid dimensions
    grid_size = int(math.sqrt(images_per_figure))
    if grid_size * grid_size < images_per_figure:
        grid_size += 1

    # Process images in batches
    for batch_idx in range(0, len(all_images), images_per_figure):
        batch_images = all_images[batch_idx : batch_idx + images_per_figure]

        # Create figure
        fig = plt.figure(figsize=(20, 20))

        for idx, img_info in enumerate(batch_images):
            image_id = img_info["id"]
            image_name = img_info["file_name"]

            # Load and process image
            image_path = os.path.join(img_dir, image_name)
            if not os.path.exists(image_path):
                continue

            image = cv2.imread(image_path)
            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]

            # Create mask
            mask = np.zeros((height, width, 4), dtype=np.uint8)

            # Get annotations
            ann_ids = coco.getAnnIds(imgIds=[image_id])
            annotations = coco.loadAnns(ann_ids)

            # Draw annotations
            for ann in annotations:
                if "segmentation" in ann:
                    color = [random.randint(0, 255) for _ in range(3)] + [255]
                    seg = ann["segmentation"]

                    if isinstance(seg, list):
                        for polygon in seg:
                            poly = np.array(polygon).reshape((-1, 2)).astype(np.int32)
                            cv2.fillPoly(mask, [poly], color)
                    elif isinstance(seg, dict):
                        rle = mask_utils.frPyObjects(seg, height, width)
                        binary_mask = mask_utils.decode(rle)
                        # binary_mask = align_mask_with_image(binary_mask, height, width)
                        mask[binary_mask > 0] = color

            # Blend image and mask
            mask_rgb = mask[:, :, :3].astype(float) / 255
            mask_alpha = mask[:, :, 3:].astype(float) / 255
            image = image.astype(float) / 255
            blended = image * (1 - mask_alpha) + mask_rgb * mask_alpha
            blended = (blended * 255).astype(np.uint8)

            # Add to subplot
            plt.subplot(grid_size, grid_size, idx + 1)
            plt.imshow(blended)
            plt.title(image_name, fontsize=8)
            plt.axis("off")

        # Adjust layout and save
        plt.tight_layout()
        output_path = os.path.join(
            output_dir, f"visualization_{batch_idx//images_per_figure + 1}.png"
        )
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        plt.close()


if __name__ == "__main__":
    annotation_file = "/home/xz/Documents/Vivid/ann_v3.json"
    img_dir = "/home/xz/Documents/Vivid/imgs"
    output_dir = "/home/xz/Documents/outcome"

    visualize_all_coco_masks(annotation_file, img_dir, output_dir)
