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
from model import build_gsam
from model.point_decoder import PointDecoder
import torch


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
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
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
    
    # point prompt
    parser.add_argument("--ckpt_path", type=str, required=True, help="checkpoint path")

    parser.add_argument("--point-config", type=str, required=True, help="point prompt config path")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    return parser



if __name__ == "__main__":
    # mask2former setup
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    # point model setup
    point_config = load_config(args.point_config)
    vision_encoder = build_gsam(point_config['vision_encoder'])
    mask_decoder = build_gsam(point_config['mask_decoder']).mask_decoder
    point_decoder = PointDecoder(mask_decoder).to(args.device)
    point_decoder.load_state_dict(torch.load(args.ckpt_path, map_location=args.device))
    point_decoder.eval()
    point_decoder.max_points = 2048
    point_decoder.nms_kernel_size = 3
    point_decoder.point_threshold = 0.10 # exp 0.28
    
    if len(args.input) == 1:
        args.input = glob.glob(os.path.expanduser(args.input[0]))
        assert args.input, "The input path(s) was not found"
        
    for path in tqdm.tqdm(args.input, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        breakpoint()
        
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        
        # import pdb; pdb.set_trace()
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
           
   