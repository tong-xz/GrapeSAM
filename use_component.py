import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("pretrain/sam-vit-base/").to(device)
processor = SamProcessor.from_pretrained("pretrain/sam-vit-base/")

raw_image = Image.open('/home/xz/Dev/Dream/data/vivid/imgs/5.png').convert("RGB")
points = np.load('/home/xz/Dev/Dream/data/vivid/anns/5.npy').reshape(1, -1, 1, 2)
input_points = points.tolist()
# input_labels = torch.ones(points.shape[0], dtype= torch.long).unsqueeze(0).unsqueeze(0)

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
image_embeddings = model.get_image_embeddings(inputs["pixel_values"])
inputs.pop("pixel_values", None)
inputs.update({"image_embeddings": image_embeddings})

with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores


import numpy as np
import matplotlib.pyplot as plt


def show_points_on_image(raw_image, input_points, input_labels=None):
    plt.figure(figsize=(10,10))
    plt.imshow(raw_image)
    input_points = np.array(input_points)
    if input_labels is None:
      labels = np.ones_like(input_points[:, 0])
    else:
      labels = np.array(input_labels)
    show_points(input_points, labels, plt.gca())
    plt.axis('on')
    plt.show()


def show_points(coords, labels, ax, marker_size=30):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=1.25)

import numpy as np
import matplotlib.pyplot as plt
import torch


def show_masks(raw_image, masks, scores):
    """
    Display multiple masks on the same image with random colors.

    Args:
        raw_image (PIL.Image or np.ndarray): The raw background image.
        masks (torch.Tensor or np.ndarray): Mask tensor of shape (N, C, H, W), where C is the channel count.
    """
    if len(masks.shape) != 4 or masks.shape[1] != 3:
        raise ValueError("Masks should have shape (N, C, H, W) with C=3 (channels)")

    nb_predictions = masks.shape[0]
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(raw_image))
    # import pdb; pdb.set_trace()

    for i in range(30):
        # 使用第一个通道显示每个掩码
        mask = masks[i, 1].cpu().detach().numpy() if isinstance(masks[i, 0], torch.Tensor) else masks[i, 0]
        
        # 随机颜色
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        plt.imshow(mask_image, alpha=0.8)  # 叠加显示具有透明度的掩码
        
    plt.axis("off")
    plt.title("Masks with random colors")
    plt.show()



show_masks(raw_image, masks[0], scores)
# show_points_on_image(raw_image, points.tolist())
