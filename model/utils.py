import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import math
from typing import Optional, Tuple, Any, List
from torch import Tensor
import yaml
import os
from PIL import Image
import gc
import cv2
import random
import matplotlib.colors as mcolors


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# ----------------SAM visualization related--------------------------------
from matplotlib import pyplot as plt
import numpy as np
import torch


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


def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis("on")
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.show()


def show_points_and_boxes_on_image(raw_image, boxes, input_points, input_labels=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
        labels = np.ones_like(input_points[:, 0])
    else:
        labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    for box in boxes:
        show_box(box, plt.gca())
    plt.axis("on")
    plt.show()


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


# # Use Agg backend for faster rendering (disable interactive mode)
# plt.switch_backend("Agg")


def show_mask(mask, ax, random_color=False, color=None, alpha=0.6):
    """Apply a colored mask overlay on the image."""
    if random_color:
        color = np.random.rand(3)  # Generate random RGB values
    elif color is not None:
        color = np.array(color)  # Convert color tuple to numpy array
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255])  # Default blue color

    mask_image = np.zeros(
        (*mask.shape[-2:], 4), dtype=np.float32
    )  # Pre-allocate memory
    mask_image[..., :3] = mask[..., None] * color  # Apply color
    mask_image[..., 3] = mask * alpha  # Apply transparency

    ax.imshow(mask_image, interpolation="nearest", alpha=alpha)


def show_masks_on_image(
    raw_image, masks, title=None, alpha=0.6, show_background=True, save_path=None
):
    """
    Optimized function to overlay segmentation masks on an image.
    """

    # Convert masks to NumPy if it's a PyTorch tensor
    if torch.is_tensor(masks):
        masks = masks.cpu().numpy()

    # Ensure masks are at least 3D (batch dimension)
    masks = np.atleast_3d(masks)

    # Pre-create figure and axis
    fig, ax = plt.subplots(
        figsize=(10, 10), dpi=300
    )  # Increased DPI for publication quality

    # Display background image only once
    if show_background:
        ax.imshow(raw_image, interpolation="nearest")

    # Vectorized mask overlay
    # for mask in masks:
    #     show_mask(mask, ax, random_color=True, alpha=alpha)

    batch_size = 10
    for i in range(0, len(masks), batch_size):
        batch_masks = masks[i : i + batch_size]
        for mask in batch_masks:
            show_mask(mask, ax=ax, random_color=True, alpha=alpha)

        del batch_masks

    if title:
        ax.set_title(title, fontsize=14)

    ax.axis("off")

    # Save or show image
    if save_path is not None:
        save_file = os.path.join(save_path, f"{title}.png")
        fig.savefig(
            save_file, bbox_inches="tight", pad_inches=0.1, dpi=100
        )  # Optimized saving
    else:
        plt.show()

    plt.close(fig)  # Close figure immediately to free memory


def show_grape_and_berry(
    raw_image,
    grape_instances,
    berry_instances,
    title=None,
    alpha=0.6,
    save_path=None,
    dpi=100,
    show_grape_indices=False,
):
    """
    Display grape and berry instance masks with adjustable background transparency.

    Args:
        raw_image: Input image (PIL Image or numpy array)
        grape_instances: Grape cluster instance masks (tensor or numpy array)
        berry_instances: Berry instance masks (tensor or numpy array)
        title: Title for the saved image
        alpha: Opacity of the masks (0-1)
        save_path: Path to save visualization
        dpi: Resolution for the output image
        show_grape_indices: Whether to show index numbers on grape instances
    """

    def get_image_dimensions(image):
        if isinstance(image, Image.Image):
            return image.height, image.width
        return image.shape[:2]

    def setup_figure(aspect_ratio, dpi):
        total_width = 18
        subplot_width = total_width / 2.2
        subplot_height = subplot_width / aspect_ratio

        fig = plt.figure(figsize=(total_width, subplot_height), dpi=dpi)
        gs = fig.add_gridspec(
            1,
            2,
            width_ratios=[1, 1],
            left=0.01,
            right=0.99,
            bottom=0.01,
            top=0.99,
            wspace=0.02,
        )
        return fig, gs

    def prepare_instances(instances):
        if torch.is_tensor(instances):
            instances = instances.cpu().numpy()
        return np.atleast_3d(instances)

    def overlay_masks(image, masks, alpha=0.6, white_bg=False, color_seed=None):
        """Generate colored overlay of instance masks on the image."""
        # Convert image to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Setup background
        h, w = masks.shape[1:3]
        white_background = np.full((h, w, 3), 255, dtype=np.uint8)
        image = white_background.copy() if white_bg or image is None else image

        # Ensure RGB format
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Generate colors
        if color_seed is not None:
            random.seed(color_seed)

        random_colors = [
            tuple(
                (np.array(mcolors.hsv_to_rgb([random.random(), 1, 1])) * 255).astype(
                    int
                )
            )
            for _ in range(len(masks))
        ]

        # Create mask overlay
        mask_combined = np.zeros_like(image, dtype=np.uint8)
        for idx, mask in enumerate(masks):
            binary_mask = (mask > 0).astype(np.uint8)
            mask_color = np.full_like(image, random_colors[idx], dtype=np.uint8)
            mask_combined = cv2.add(mask_combined, mask_color * binary_mask[..., None])

        return cv2.addWeighted(image, 1 - alpha, mask_combined, alpha, 0), random_colors

    def add_grape_indices(ax, instances, colors):
        """Add index numbers to grape instances."""
        for idx, mask in enumerate(instances):
            if not np.any(mask):
                continue

            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0:
                continue

            centroid_y = int(np.mean(y_indices))
            centroid_x = int(np.mean(x_indices))
            text_color = "white" if np.mean(colors[idx]) < 128 else "black"

            ax.text(
                centroid_x,
                centroid_y,
                str(idx),
                color=text_color,
                fontsize=12,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1),
            )

    # Main execution
    if save_path:
        plt.switch_backend("Agg")

    # Setup figure
    h, w = get_image_dimensions(raw_image)
    fig, gs = setup_figure(w / h, dpi)
    ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # Prepare instance masks
    grape_instances = prepare_instances(grape_instances)
    berry_instances = prepare_instances(berry_instances)

    # Generate overlays
    grape_overlay, grape_colors = overlay_masks(
        raw_image, grape_instances, alpha=alpha, color_seed=42
    )
    berry_overlay, _ = overlay_masks(
        raw_image, berry_instances, alpha=alpha, color_seed=84
    )

    # Display results
    ax1.imshow(grape_overlay)
    ax2.imshow(berry_overlay)

    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.axis("off")

    if show_grape_indices:
        add_grape_indices(ax1, grape_instances, grape_colors)

    # Save or display
    if save_path:
        save_file = os.path.join(
            save_path, f"{title if title else 'grape_and_berry'}.png"
        )
        fig.savefig(
            save_file, bbox_inches="tight", pad_inches=0.5, facecolor="white", dpi=300
        )
    else:
        plt.show()

    plt.close(fig)


def show_grape_and_berry0(
    raw_image,
    grape_instances,
    berry_instances,
    title=None,
    alpha=0.6,
    save_path=None,
    dpi=100,
    show_grape_indices=False,
):
    """
    Optimized function to display grape and berry masks with adjustable background transparency.

    Args:
        raw_image: Input image
        grape_instances: Grape cluster instance masks
        berry_instances: Berry instance masks
        title: Title for the saved image
        alpha: Opacity of the masks
        save_path: Path to save visualization
        dpi: Resolution for the output image
        show_grape_indices: If True, show index numbers on each grape instance in the first subplot
    """
    if save_path:
        plt.switch_backend("Agg")

    # Calculate aspect ratio of the input image
    if isinstance(raw_image, Image.Image):
        h, w = raw_image.height, raw_image.width
    else:
        h, w = raw_image.shape[:2]
    aspect_ratio = w / h

    # Adjust figure size based on aspect ratio while maintaining total width
    total_width = 18  # Reduced from 27 since we only need 2 subplots
    subplot_width = total_width / 2.2  # Leave some space for gaps
    subplot_height = subplot_width / aspect_ratio

    # Create figure with gridspec for more control
    fig = plt.figure(figsize=(total_width, subplot_height), dpi=dpi)
    gs = fig.add_gridspec(
        1,
        2,  # Changed from 3 to 2 subplots
        width_ratios=[1, 1],
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.99,
        wspace=0.02,
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # Convert tensors to numpy
    if torch.is_tensor(grape_instances):
        grape_instances = grape_instances.cpu().numpy()
    if torch.is_tensor(berry_instances):
        berry_instances = berry_instances.cpu().numpy()

    # Ensure 3D shape
    grape_instances = np.atleast_3d(grape_instances)
    berry_instances = np.atleast_3d(berry_instances)

    def overlay_masks(image, masks, alpha=0.6, white_bg=False, color_seed=None):
        """
        Overlay instance masks on an image with random colors.

        Parameters:
            image (np.ndarray or None): The base image to overlay the masks on.
            masks (np.ndarray): A stack of binary instance masks.
            alpha (float): Opacity of the masks.
            white_bg (bool): If True, enforce a pure white background.
            color_seed (int or None): Seed for random color generation to differentiate grape and berry masks.
        """

        # Convert PIL image to NumPy array
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Create a base white background
        h, w = masks.shape[1:3]
        white_background = np.full((h, w, 3), 255, dtype=np.uint8)

        if white_bg:
            # For third plot: Use a **pure white background** with NO blending
            image = white_background.copy()
        else:
            # Ensure a valid background
            image = white_background.copy() if image is None else image

        # Convert grayscale to RGB if needed
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Ensure different random colors for grape and berry masks
        if color_seed is not None:
            random.seed(color_seed)  # Set different seeds for grape and berry

        # Generate unique colors for each instance
        unique_masks = len(masks)
        random_colors = [
            tuple(
                (np.array(mcolors.hsv_to_rgb([random.random(), 1, 1])) * 255).astype(
                    int
                )
            )
            for _ in range(unique_masks)
        ]

        # Initialize mask overlay
        mask_combined = np.zeros_like(image, dtype=np.uint8)

        # Apply each mask
        for idx, mask in enumerate(masks):
            mask = (mask > 0).astype(np.uint8)  # Ensure binary mask
            mask_color = np.full_like(image, random_colors[idx], dtype=np.uint8)
            mask_combined = cv2.add(mask_combined, mask_color * mask[..., None])

        # Ensure correct data type
        mask_combined = mask_combined.astype(image.dtype)

        return cv2.addWeighted(image, 1 - alpha, mask_combined, alpha, 0), random_colors

    # Generate overlayed images with distinct colors
    grape_overlay, grape_colors = overlay_masks(
        raw_image, grape_instances, alpha=alpha, color_seed=42
    )
    berry_overlay, _ = overlay_masks(
        raw_image, berry_instances, alpha=alpha, color_seed=84
    )

    # Display results with equal aspect ratio
    ax1.imshow(grape_overlay)
    ax1.set_aspect("equal")
    ax1.axis("off")

    # Add grape instance indices if requested
    if show_grape_indices:
        # For each grape instance, find centroid and add text label
        for idx, mask in enumerate(grape_instances):
            # Calculate centroid of the mask
            if np.any(mask):  # Check if mask is not empty
                y_indices, x_indices = np.where(mask > 0)
                if len(y_indices) > 0:
                    centroid_y = int(np.mean(y_indices))
                    centroid_x = int(np.mean(x_indices))

                    # Add text with contrasting color to the mask color
                    text_color = (
                        "white" if np.mean(grape_colors[idx]) < 128 else "black"
                    )

                    # Add instance index with small box for better visibility
                    ax1.text(
                        centroid_x,
                        centroid_y,
                        str(idx),
                        color=text_color,
                        fontsize=12,
                        fontweight="bold",
                        ha="center",
                        va="center",
                        bbox=dict(
                            facecolor="white", alpha=0.5, edgecolor="none", pad=1
                        ),
                    )

    ax2.imshow(berry_overlay)
    ax2.set_aspect("equal")
    ax2.axis("off")

    # Save or display the figure
    if save_path:
        save_file = os.path.join(
            save_path, f"{title if title else 'grape_and_berry'}.png"
        )
        fig.savefig(
            save_file, bbox_inches="tight", pad_inches=0.5, facecolor="white", dpi=300
        )
    else:
        plt.show()

    plt.close(fig)


# ----------------SAM scale related--------------------------------
from PIL import Image


def scale_image_and_keypoints(img, keypoints, target_size=(1024, 1024)):
    """
    Scale the image to a specified size while maintaining aspect ratio and scale keypoint coordinates accordingly

    Args:
        img: PIL Image object (RGB)
        keypoints: numpy array with shape (N, 3) representing point coordinates and scale [[x1,y1,s1], [x2,y2,s2], ...]
        target_size: tuple, target dimensions (width, height)

    Returns:
        scaled_img: scaled RGB image
        scaled_keypoints: scaled keypoint coordinates
    """
    # Get original dimensions
    orig_w, orig_h = img.size
    target_w, target_h = target_size

    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    # Scale image
    scaled_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Create new RGB background
    final_img = Image.new("RGB", target_size, (0, 0, 0))

    # Calculate paste position (centered)
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    final_img.paste(scaled_img, (paste_x, paste_y))

    # Convert keypoints list to numpy array
    keypoints = np.array(keypoints[0])  # keypoints[0] since it's a nested list
    scaled_keypoints = keypoints.copy()

    scaled_keypoints[:, 0] = keypoints[:, 0] * scale + paste_x
    scaled_keypoints[:, 1] = keypoints[:, 1] * scale + paste_y

    scaled_keypoints = scaled_keypoints.tolist()
    scaled_keypoints = [[[point] for point in scaled_keypoints]]

    return final_img, scaled_keypoints


# brought from https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/utils/transforms.py#L16
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length
        )
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(
        self, coords: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(
        self, boxes: np.ndarray, original_size: Tuple[int, ...]
    ) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(
            image.shape[2], image.shape[3], self.target_length
        )
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(
        oldh: int, oldw: int, long_side_length: int
    ) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)
