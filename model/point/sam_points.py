"""
This is to establish the whole pipleline of using point prompts to get berry segmentations
"""

"""
SAM original evaluation based on bbox prompts
"""

import torch
from transformers import SamModel, SamProcessor
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch


def get_points(npy_path):
    points_list = np.load(npy_path).tolist()

    """
    PPPM: Per Point Per Mask
    Single Mask Multiple Points: (1, 1, n, 2); e.g. [[[850, 1100]], [[2250, 1000]]]
    Multiple Masks Multiple Points: (1, n, 1, 2); e.g. [[[[850, 1100]], [[2250, 1000]]]]
    """
    PPPM = [[[point] for point in points_list]]
    return PPPM


def show_mask(mask, ax, random_color=False, color=None, alpha=0.6):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    elif color is not None:
        color = np.array(list(color) + [alpha])
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_masks_on_image(raw_image, masks, title=None, alpha=0.6, show_background=True):
    # Handle single mask case
    if len(masks.shape) == 4:
        masks = masks.squeeze()
    if len(masks.shape) == 2:  # Single mask
        masks = masks[None, ...]  # Add batch dimension

    # Create a single subplot
    plt.figure(figsize=(10, 10))

    # Only show the background image if show_background is True
    if show_background:
        plt.imshow(np.array(raw_image))

    # Show all masks on the same image with random colors
    for mask in masks:
        mask = mask.cpu().detach()
        show_mask(mask, plt.gca(), random_color=True, alpha=alpha)

    plt.title(title)
    plt.axis("off")
    plt.savefig(f"{title}.png")
    plt.show()


def sam_points_inference(model, processor, raw_image, points, multimask_output):
    inputs = processor(raw_image, input_points=points, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=multimask_output)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )
    scores = outputs.iou_scores
    return masks, scores


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    img_path = (
        "/home/xz/Dev/baseline-exp-playground/DATASET/vivid-close/test/immature_508.png"
    )
    npy_path = (
        "/home/xz/Dev/baseline-exp-playground/DATASET/vivid-close/test/immature_508.npy"
    )
    raw_image = Image.open(img_path).convert("RGB")

    points = get_points(npy_path)
    masks, scores = sam_points_inference(
        model, processor, raw_image, points, multimask_output=True
    )

    # use when multi-mask is true
    masks = masks[0][:, 0:1, :, :]

    show_masks_on_image(raw_image, masks, title="multi-mask-1082")
