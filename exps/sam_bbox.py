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
from torchvision.ops import masks_to_boxes
from utils import show_masks_on_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.iou import IntersectionOverUnion


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
    model_path = "/home/xz/Dev/GrapeSAM/pretrain/sam-vit-huge"
    data_root = "/home/xz/Dev/GrapeSAM/data/vivid/"

    model = SamModel.from_pretrained(model_path).to(device)
    processor = SamProcessor.from_pretrained(model_path)

    vivid_exp_dataset = VividDataset(
        data_root=data_root,
        txt_path=data_root + "test.txt",
        json_path=data_root + "instances_updated.json",
    )

    cnt = 0

    metric = MeanAveragePrecision(iou_type="segm")

    #     data_loader = DataLoader(vivid_exp_dataset, batch_size=1, shuffle=False)
    # data_iter = iter(data_loader)

    data_iter = iter(vivid_exp_dataset)

    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            break

        batch = next(data_iter)

        img_path, bboxes, gt_masks = batch

        raw_image = Image.open(img_path).convert("RGB")
        gt_masks = torch.from_numpy(gt_masks).float()
        if len(bboxes) != 0:

            bboxes = [bboxes]

            try:
                pred_masks, pred_scores = sam_bbox_inference(
                    model, processor, raw_image, bboxes
                )
                # print(pred_scores)
            except Exception as e:
                print(img_path)
                # breakpoint()
                continue

        preds = [
            dict(
                # masks=torch.tensor([mask_pred], dtype=torch.bool),
                masks=pred_masks[0].type(torch.bool).any(dim=0),
                scores=torch.tensor([1.0]),
                labels=torch.tensor([0]),
            )
        ]

        gts = [
            dict(
                masks=gt_masks.type(torch.bool).any(dim=0).unsqueeze(0),
                labels=torch.tensor([0]),
            )
        ]

        metric.update(preds, gts)

        # masks: [torch.size([n, 1, h, w])]
        # gt_masks: [torch.size([n, h, w])]

        # show_masks_on_image(raw_image, pred_masks[0], pred_scores, title="SAM")
        # show_masks_on_image(raw_image, gt_masks, pred_scores, title="GT")
        # import pdb

        # pdb.set_trace()
        cnt += 1

    result = metric.compute()
    print("AP:", result)
    print("cnt:", cnt)
