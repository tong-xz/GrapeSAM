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


def visualize_coco_masks(annotation_file, image_name, transparency=0.5, ax=None, save_path=None):
    """
    Visualize COCO masks for a given image name with different colors.

    Args:
        annotation_file (str): Path to the COCO annotation JSON file.
        image_name (str): Name of the image file to visualize.
        transparency (float): Transparency level for masks (0.0 to 1.0, where 1.0 is opaque)
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates a new figure.
        save_path (str, optional): Path to save the visualization. If None, displays the plot.
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
                
                # Handle nested list case
                if isinstance(bbox, list) and len(bbox) == 1 and isinstance(bbox[0], list):
                    bbox = bbox[0]
                
                # Check if bbox has at least 4 elements before accessing
                if bbox and len(bbox) >= 4:
                    print(
                        f"Mask bbox: x={bbox[0]}, y={bbox[1]}, width={bbox[2]}, height={bbox[3]}"
                    )
                else:
                    print(f"Warning: Invalid bbox format: {bbox}")

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

    # Try to load corresponding .npy file with points
    points = None
    npy_path = os.path.join("/home/xz/Documents/Vivid/anns", os.path.splitext(image_name)[0] + ".npy")
    if os.path.exists(npy_path):
        try:
            points = np.load(npy_path)
            print(f"Loaded points from {npy_path}, shape: {points.shape}")
        except Exception as e:
            print(f"Error loading points file: {e}")
    else:
        print(f"Points file not found: {npy_path}")

    # Create a new figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Show the original image first
    ax.imshow(image)
    
    # Create a separate RGBA mask for overlay
    rgba_mask = np.zeros((actual_height, actual_width, 4), dtype=np.float32)
    rgba_mask[:, :, :3] = mask_rgb
    rgba_mask[:, :, 3] = mask_alpha[:, :, 0] * transparency  # Use adjustable transparency
    
    # Overlay the transparent mask
    ax.imshow(rgba_mask)
    
    # Plot points if available
    if points is not None:
        # Check the shape of points to determine how to plot them
        if len(points.shape) == 2 and points.shape[1] >= 2:
            # Assuming points are in format [x, y, ...] or [[x, y], ...]
            ax.scatter(points[:, 0], points[:, 1], c='yellow', s=10, marker='o', alpha=0.8)
            print(f"Plotted {len(points)} points")
        else:
            print(f"Unexpected points shape: {points.shape}, cannot plot")
    
    ax.axis("off")
    
    # If this is a standalone plot (not part of a grid), show or save it
    if save_path is not None and ax is None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()


def visualize_multiple_images(annotation_file, image_names, output_path, transparency=0.6, grid_size=(3, 3), subplot_spacing=0.0):
    """
    Visualize multiple images in a grid and save the result.
    
    Args:
        annotation_file (str): Path to the COCO annotation JSON file
        image_names (list): List of image file names to visualize
        output_path (str): Path to save the output visualization
        transparency (float): Transparency level for masks
        grid_size (tuple): Grid dimensions (rows, cols)
        subplot_spacing (float): Spacing between subplots (0.0 means no spacing)
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    axes = axes.flatten()
    
    # Set figure background to transparent
    fig.patch.set_alpha(0.0)
    
    # Hide any unused subplots
    for i in range(len(image_names), rows*cols):
        axes[i].axis('off')
        axes[i].set_visible(False)
    
    # Process each image
    for i, image_name in enumerate(image_names[:rows*cols]):
        try:
            print(f"Processing image {i+1}/{len(image_names)}: {image_name}")
            visualize_coco_masks(annotation_file, image_name, transparency, ax=axes[i])
            # Remove axis padding to make images more compact
            axes[i].set_xmargin(0)
            axes[i].set_ymargin(0)
            # Make subplot background transparent
            axes[i].patch.set_alpha(0.0)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            axes[i].text(0.5, 0.5, f"Error: {str(e)}", 
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Adjust layout with customizable spacing
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=subplot_spacing, hspace=subplot_spacing)
    
    # Determine file format and save with appropriate settings
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        # Save as high-quality JPEG (without quality parameter which isn't supported in some versions)
        plt.savefig(output_path, bbox_inches='tight', dpi=600)
    else:
        # For other formats (like PNG), save with transparency
        plt.savefig(output_path, bbox_inches='tight', dpi=600, transparent=True)
    
    print(f"Saved visualization to {output_path}")
    plt.close()


def visualize_coco_masks_points(annotation_file, image_name, transparency=0.5, ax=None, save_path=None):
    """
    Visualize COCO masks and points for a given image.

    Args:
        annotation_file (str): Path to the COCO annotation JSON file.
        image_name (str): Name of the image file to visualize.
        transparency (float): Transparency level for masks (0.0 to 1.0, where 1.0 is opaque)
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates a new figure.
        save_path (str, optional): Path to save the visualization. If None, displays the plot.
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

    # Load the original image
    img_dir = "/home/xz/Documents/Vivid/imgs"
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

    # Initialize mask with actual image dimensions
    mask = np.zeros((actual_height, actual_width, 4), dtype=np.uint8)

    # Load annotations for the image
    annotation_ids = coco.getAnnIds(imgIds=[image_id])
    annotations = coco.loadAnns(annotation_ids)

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

    # Try to load corresponding .npy file with points
    points = None
    npy_path = os.path.join("/home/xz/Documents/Vivid/anns", os.path.splitext(image_name)[0] + ".npy")
    if os.path.exists(npy_path):
        try:
            points = np.load(npy_path)
            print(f"Loaded points from {npy_path}, shape: {points.shape}")
        except Exception as e:
            print(f"Error loading points file: {e}")
    else:
        print(f"Points file not found: {npy_path}")

    # Create a new figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Show the original image first
    ax.imshow(image)
    
    # Create a separate RGBA mask for overlay
    rgba_mask = np.zeros((actual_height, actual_width, 4), dtype=np.float32)
    rgba_mask[:, :, :3] = mask_rgb
    rgba_mask[:, :, 3] = mask_alpha[:, :, 0] * transparency  # Use adjustable transparency
    
    # Overlay the transparent mask
    ax.imshow(rgba_mask)
    
    # Plot points if available
    if points is not None:
        # Check the shape of points to determine how to plot them
        if len(points.shape) == 2 and points.shape[1] >= 2:
            # Assuming points are in format [x, y, ...] or [[x, y], ...]
            ax.scatter(points[:, 0], points[:, 1], c='yellow', s=10, marker='o', alpha=0.8)
            print(f"Plotted {len(points)} points")
        else:
            print(f"Unexpected points shape: {points.shape}, cannot plot")
    
    ax.axis("off")
    
    # If this is a standalone plot (not part of a grid), show or save it
    if save_path is not None and ax is None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    elif ax is None:
        plt.show()


def visualize_multiple_combined(annotation_file, image_names, output_path, transparency=0.6, grid_size=(3, 3), subplot_spacing=0.0):
    """
    Visualize multiple images with masks and points in a grid and save the result.
    
    Args:
        annotation_file (str): Path to the COCO annotation JSON file
        image_names (list): List of image file names to visualize
        output_path (str): Path to save the output visualization
        transparency (float): Transparency level for masks
        grid_size (tuple): Grid dimensions (rows, cols)
        subplot_spacing (float): Spacing between subplots
    """
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*5))
    axes = axes.flatten()
    
    # Set figure background to transparent
    fig.patch.set_alpha(0.0)
    
    # Hide any unused subplots
    for i in range(len(image_names), rows*cols):
        axes[i].axis('off')
        axes[i].set_visible(False)
    
    # Process each image
    for i, image_name in enumerate(image_names[:rows*cols]):
        try:
            print(f"Processing image {i+1}/{len(image_names)}: {image_name}")
            visualize_coco_masks_points(annotation_file, image_name, transparency, ax=axes[i])
            # Remove axis padding
            axes[i].set_xmargin(0)
            axes[i].set_ymargin(0)
            # Make subplot background transparent
            axes[i].patch.set_alpha(0.0)
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            axes[i].text(0.5, 0.5, f"Error: {str(e)}", 
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    
    # Adjust layout
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=subplot_spacing, hspace=subplot_spacing)
    
    # Save the visualization
    plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)
    print(f"Saved visualization to {output_path}")
    plt.close()


# Example usage
if __name__ == "__main__":
    annotation_file = "/home/xz/Documents/Vivid/instances_default_v5.json"
    
    # Single image visualization with masks and points
    # visualize_coco_masks_points(annotation_file, "1.png", save_path="combined_visualization.jpg")
    
    # Multiple images visualization
    image_names = [
        "1.png", 
        "IMG_9365.jpg", 
        "immature_184.png",
        # Add more image names as needed
    ]
    
    # Choose which visualization to run
    visualize_multiple_images(annotation_file, image_names, "mask_visualization_grid.jpg", transparency=0.6)
    # visualize_multiple_combined(annotation_file, image_names, "combined_visualization_grid.jpg", transparency=0.6)
