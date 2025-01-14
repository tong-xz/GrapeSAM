"""
This is to establish the whole pipleline of using point prompts to get berry segmentations
"""

"""
SAM original evaluation based on bbox prompts
"""

import torch
from transformers import SamModel, SamProcessor
from exp_dataset import VividDataset
from torch.utils.data import DataLoader
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torch
from utils import show_masks_on_image


def get_points(npy_path):
    points_list = np.load(npy_path).tolist()

    """
    PPPM: Per Point Per Mask
    Single Mask Multiple Points: (1, 1, n, 2); e.g. [[[850, 1100]], [[2250, 1000]]]
    Multiple Masks Multiple Points: (1, n, 1, 2); e.g. [[[[850, 1100]], [[2250, 1000]]]]
    """
    PPPM = [[[point] for point in points_list]]
    return PPPM


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

    img_path = "/home/xz/Documents/Vivid/imgs/1082.png"
    npy_path = "/home/xz/Documents/Vivid/anns/1082.npy"
    raw_image = Image.open(img_path).convert("RGB")

    points = get_points(npy_path)
    masks, scores = sam_points_inference(
        model, processor, raw_image, points, multimask_output=True
    )

    # use when multi-mask is false
    # masks = masks[0]

    # use when multi-mask is true
    masks = masks[0][:, 0:1, :, :]

    show_masks_on_image(raw_image, masks, scores, title="multi-mask-1082")
