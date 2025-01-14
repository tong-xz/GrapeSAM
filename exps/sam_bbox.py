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


def sam_bbox_inference(model, processor, raw_image, bboxes):
    inputs = processor(raw_image, input_boxes=bboxes, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
    inputs.pop("pixel_values", None)
    inputs.update({"image_embeddings": image_embeddings})

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

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

    vivid_exp_dataset = VividDataset(
        data_root="/home/xz/Dev/GrapeSAM/data/vivid",
        txt_path="/home/xz/Dev/GrapeSAM/data/vivid/test.txt",
        json_path="/home/xz/Dev/GrapeSAM/data/vivid/instances_default_v4.json",
    )

    for batch in vivid_exp_dataset:
        img_path, bboxes, gt_masks = batch
        raw_image = Image.open(img_path).convert("RGB")
        gt_masks = torch.from_numpy(gt_masks).float()
        bboxes = [bboxes]

        pred_masks, pred_scores = sam_bbox_inference(
            model, processor, raw_image, bboxes
        )
        # masks: [torch.size([n, 1, h, w])]
        # gt_masks: [torch.size([n, h, w])]

        show_masks_on_image(raw_image, pred_masks[0], pred_scores, title="SAM")
        show_masks_on_image(raw_image, gt_masks, pred_scores, title="GT")
