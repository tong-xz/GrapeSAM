import argparse
import os
import torch
from model import build_sam_vit_h, PointDecoder
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from model import VividDataset

def calculate_MAE(gt_points, pred_points):
    pass

def calculate_MSE(gt_points, pred_points):
    pass

def main(args):
    # prepare dataset and everything
    root_dir, ckp_path = args.root_dir, args.ckp_path
    test_txt_file = os.path.join(root_dir, 'test.txt')

    with open(test_txt_file, 'r') as f:
        test_list = [line.strip() for line in f]

    dataset = VividDataset(root_dir, test_list, mode='test')

    # init model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    sam = build_sam_vit_h().to(device).eval()
    point_mask_decoder = PointDecoder(sam).to(device).eval()
    point_mask_decoder.load_state_dict(torch.load(ckp_path, map_location=device))

    for idx, data in enumerate(dataset):
        img, gt_points = data[0], data[1]      

        with torch.inference_mode():
            features = sam.image_encoder(img).to(device)
            #TODO tune these parameters to see the best effect

            point_mask_decoder.max_points = 512
            point_mask_decoder.nms_kernel_size = 3
            point_mask_decoder.point_threshold = 0.2
            pred = point_mask_decoder(features)
            pred_points = pred['pred_points'].squeeze()

            # compare difference with gt and prediciton
            #TODO implement MAE MSE here



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test script arguments')
    parser.add_argument('--root_dir', type=str,required=True, help='root directory of the dataset folderS')
    parser.add_argument('--ckp_path', type=str, required=True, help='checkpoint path')

    args = parser.parse_args()
    main(args)