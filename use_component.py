import torch
from model.segment_anything import build_sam_vit_h, SamPredictor
from model.util import predict_masks, vis_pred
import cv2
from matplotlib import pyplot as plt
import numpy as np

from model.prompter import GSAMVisionEncoder, GSAMPromptEncoder, GSAMMaskDecoder, GSAMPositionalEmbedding
import numpy as np
import matplotlib.pyplot as plt
import gc

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()

def show_masks_on_image(raw_image, masks):
  plt.imshow(np.array(raw_image))
  ax = plt.gca()
  ax.set_autoscale_on(False)
  for mask in masks:
      show_mask(mask, ax=ax, random_color=True)
  plt.axis("off")
  plt.show()
  del mask
  gc.collect()

# load data
image = cv2.imread('/home/xz/Dev/Dream/data/vivid/imgs/5.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (1024, 1024))
# input_points = np.load('/home/xz/Dev/Dream/data/vivid/anns/5.npy')
input_points = [[[1353, 2730],[1409, 2822]]]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

image_tensor = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)   # numpy -> tensor
point_tensor = torch.tensor(input_points).to(device)
# input_labels = torch.ones(input_points.shape[0], dtype= torch.long).unsqueeze(0).unsqueeze(0).to(device)


hf_name = "pretrain/sam-vit-base/"
init_cfg = {'checkpoint': '/home/xz/Dev/Dream/pretrain/sam-vit-base/pytorch_model.bin'}
extra_cfg = {'output_hidden_states': True}

vision_encoder = GSAMVisionEncoder(hf_pretrain_name=hf_name, init_cfg=init_cfg, extra_cfg=extra_cfg, device=device)
prompt_encoder = GSAMPromptEncoder(hf_pretrain_name=hf_name, init_cfg=init_cfg, extra_cfg=None, device=device)
mask_decoder = GSAMMaskDecoder(hf_pretrain_name=hf_name, init_cfg=init_cfg, extra_cfg=None, device=device)
pe = GSAMPositionalEmbedding(hf_pretrain_name=hf_name, init_cfg=init_cfg, extra_cfg=None, device=device)


vision_output = vision_encoder(image_tensor)
import pdb; pdb.set_trace()

# prompt_embed = prompt_encoder(input_points = point_tensor, input_labels = input_labels, input_boxes = None, input_masks=None)


# outputs = mask_decoder(image_embeddings=img_embed[0], image_positional_embeddings=pos_embed, sparse_prompt_embeddings=prompt_embed[0], dense_prompt_embeddings=prompt_embed[1], multimask_output=True)

# masks = outputs[0].squeeze().squeeze().cpu().detach().numpy()

# show_masks_on_image(image, masks)