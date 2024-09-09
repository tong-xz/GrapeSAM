import torch
from model.segment_anything import build_sam_vit_h, SamPredictor
from model.util import predict_masks_by_points, show_all
import cv2
from matplotlib import pyplot as plt
import numpy as np


# load data
image = cv2.imread('/home/xz/Dev/Dream/data/redo-data/test/IMG_7362.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
input_points = np.load('/home/xz/Dev/Dream/data/redo-data/test/IMG_7362.npy')

# init model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam = build_sam_vit_h().to(device).eval()
predictor = SamPredictor(sam)

masks, _, _ = predict_masks_by_points(predictor, image, input_points)
masks = masks.cpu().detach()
show_all(image, masks, save_path='./test.png')