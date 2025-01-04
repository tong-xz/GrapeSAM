import json
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import numpy as np
import cv2
import random
import os


def align_mask_with_image(mask, target_height, target_width):
    """
    Resize mask to match target image dimensions.

    Args:
        mask (np.ndarray): Input mask array (height, width, 4)
        target_height (int): Target height
        target_width (int): Target width

    Returns:
        np.ndarray: Resized mask matching target dimensions
    """
    if mask.shape[:2] != (target_height, target_width):
        print(
            f"Resizing mask from {mask.shape[:2]} to ({target_height}, {target_width})"
        )
        return cv2.resize(
            mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST
        )
    return mask


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
    image_id = None
    for img in coco.dataset["images"]:
        if img["file_name"] == image_name:
            image_id = img["id"]
            break

    if image_id is None:
        raise ValueError(f"Image '{image_name}' not found in the dataset")

    # Load image information
    image_info = coco.loadImgs(image_id)[0]
    height, width = image_info["height"], image_info["width"]
    print(f"Mask size from image info: height={height}, width={width}")

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

    # Get actual image dimensions
    actual_height, actual_width = image.shape[:2]
    print(f"Actual image dimensions: height={actual_height}, width={actual_width}")

    # Initialize mask with actual image dimensions instead of COCO dimensions
    mask = np.zeros((actual_height, actual_width, 4), dtype=np.uint8)

    # Load annotations for the image
    annotation_ids = coco.getAnnIds(
        imgIds=[image_id]
    )  # Ensure we only get annotations for this specific image
    annotations = coco.loadAnns(annotation_ids)


    # Print mask size from segmentation data
    for annotation in annotations:
        if "segmentation" in annotation:
            segmentation = annotation["segmentation"]
            if isinstance(segmentation, dict):  # RLE format
                # Get size from RLE format
                if "size" in segmentation:
                    rle_height, rle_width = segmentation["size"]
                    print(f"Mask size from RLE: height={rle_height}, width={rle_width}")
            elif isinstance(segmentation, list):  # Polygon format
                # For polygon format, you can get the bounding box
                bbox = annotation.get("bbox", [])  # [x,y,width,height]
                if bbox:
                    print(
                        f"Mask bbox: x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]}"
                    )

    # Assign random colors to each annotation with alpha
    for annotation in annotations:
        if "segmentation" in annotation:
            segmentation = annotation["segmentation"]
            color = [random.randint(0, 255) for _ in range(3)] + [255]

            if isinstance(segmentation, list):  # Polygon format
                for seg in segmentation:
                    poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                    # Scale polygon points if necessary
                    if (height, width) != (actual_height, actual_width):
                        scale_y = actual_height / height
                        scale_x = actual_width / width
                        poly[:, 0] = poly[:, 0] * scale_x
                        poly[:, 1] = poly[:, 1] * scale_y
                    cv2.fillPoly(mask, [poly], color)
            elif isinstance(segmentation, dict):  # RLE format
                rle = mask_utils.frPyObjects(segmentation, height, width)
                binary_mask = mask_utils.decode(rle)
                # Resize binary mask if necessary
                binary_mask = align_mask_with_image(
                    binary_mask, actual_height, actual_width
                )
                mask[binary_mask > 0] = color

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
annotation_file = "/home/xz/Documents/Vivid/ann_v2.json"
image_name = "IMG_7362.jpeg"  # Replace with your image file name
visualize_coco_masks(annotation_file, image_name)
