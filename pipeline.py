# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from model.mask2former import add_maskformer2_config
from model.mask2former.demo.predictor import VisualizationDemo
from model.utils import load_config
from model import build_gsam, build_loader
from model.point_decoder import PointDecoder
import torch
from eval_prompter import tensor_to_pil
from model.ops.ops import plot_results
import torch.nn.functional as F
import torchvision
import PIL
from matplotlib import pyplot as plt

# constants
WINDOW_NAME = "mask2former demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="pipeline config")

    # maskformer2
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )

    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser


def resize_and_pad_image(img, target_size=(1024, 1024), pad_color=(0, 0, 0)):
    """
    Resize an image to fit within target_size while maintaining aspect ratio,
    and pad with a specified color to reach the target size.

    :param img: Input image in BGR format.
    :param target_size: Tuple of (width, height) for the target size.
    :param pad_color: Tuple of (B, G, R) for the padding color.
    :return: Resized and padded image.
    """
    h, w = img.shape[:2]
    scale = min(target_size[1] / h, target_size[0] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = (target_size[1] - new_h) // 2
    bottom = target_size[1] - new_h - top
    left = (target_size[0] - new_w) // 2
    right = target_size[0] - new_w - left

    padded_img = cv2.copyMakeBorder(
        resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color
    )
    return padded_img


if __name__ == "__main__":
    # mask2former setup
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    CKPT_PATH = "/home/xz/Dev/GrapeSAM/weights/vivid6/point_decoder_11-13-07:37:21.pth"
    POINT_CONFIG = "/home/xz/Dev/GrapeSAM/config/prompter_huge.yaml"
    ROOT_DIR = "/home/xz/Dev/GrapeSAM/data/vivid-6t05"
    DEVICE = "cuda"

    # point model setup
    point_config = load_config(POINT_CONFIG)
    vision_encoder = build_gsam(point_config["vision_encoder"]).to(DEVICE)
    mask_decoder = build_gsam(point_config["mask_decoder"]).mask_decoder.to(DEVICE)
    # prompt_encoder = build_gsam(point_config['prompt_encoder']).to(DEVICE)
    test_loader = build_loader(ROOT_DIR, batch_size=1)["test"]

    point_decoder = PointDecoder(mask_decoder).to(DEVICE)
    point_decoder.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    point_decoder.eval()
    point_decoder.max_points = 2048
    point_decoder.nms_kernel_size = 3
    point_decoder.point_threshold = 0.15  # exp 0.28

    for img, _ in test_loader:
        start_time = time.time()
        point_img, mask_img = img.clone(), img.clone()

        # mask prediction
        mask_img = (
            mask_img.squeeze(0).permute(1, 2, 0).cpu().numpy()
        )  # Convert to (H,W,C) numpy array
        mask_img = (mask_img * 255).astype(np.uint8)  # Scale to 0-255 range
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

        predictions, visualized_output = demo.run_on_image(mask_img)
        pred_masks = predictions["instances"].pred_masks.cpu()
        # import pdb; pdb.set_trace()

        # mask merge
        merged_mask = pred_masks.any(dim=0).float()  # (1024, 1024)

        downsampled_mask = (
            F.interpolate(
                merged_mask.unsqueeze(0).unsqueeze(
                    0
                ),  # 添加 batch 和通道维度 -> (1, 1, 1024, 1024)
                size=(256, 256),  # 目标大小
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )  # 移除 batch 和通道维度 -> (256, 256)

        final_mask = downsampled_mask.unsqueeze(0).unsqueeze(0).cuda()

        # point prediction
        with torch.inference_mode(), torch.no_grad():
            # img: torch.Size([1, 3, 1024, 1024]);
            point_img = point_img.to(DEVICE)
            features = vision_encoder(point_img)[0].to(DEVICE)
            pred_points = point_decoder(features, masks=final_mask)["pred_points"]

        pred_points = pred_points.squeeze(0).cpu()

        logger.info(
            "{} and {} in {:.2f}s".format(
                (
                    "detected {} grape clusters".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished"
                ),
                "detected {} berries".format(len(pred_points)),
                time.time() - start_time,
            )
        )

        # TODO finish prompt encoder and mask decoder part

        # pred_points = pred_points.unsqueeze(0).cuda()
        # pred_masks = pred_masks.cuda()
        # sam_mask_outputs = mask_decoder(features, dense_prompt_embeddings=pred_masks, sparse_prompt_embeddings=pred_points,
        #                                  image_positional_embeddings=None, multimask_output=True)
        img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)  # Convert RGB to BGR
        plot_results(
            img,
            masks=pred_masks,
            points=pred_points,
            dot_size=12,
            save_path=args.output,
        )
