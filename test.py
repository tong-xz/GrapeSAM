from model.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from model.point_decoder import PointDecoder
import torch
from model.redo_dataset import RedoDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam = build_sam_vit_h().to(device).eval()
point_decoder = PointDecoder(sam).to(device)

# 加载已训练的模型参数
point_decoder.load_state_dict(torch.load('decoder.pth'))

# 将模型设置为评估模式
point_decoder.eval()

redo_dataset = RedoDataset('/home/xz/Dev/Dream/data/redo-data', 'test')
redo_loader = DataLoader(redo_dataset, batch_size=1)
imgs, heatmaps = next(iter(redo_loader))

# Corrected lines:
imgs = imgs.to(device)

with torch.no_grad():
    features = sam.image_encoder(imgs) 

outputs = point_decoder(features)['pred_heatmaps_nms']

print(outputs.shape)
outputs = outputs.squeeze().detach().cpu().numpy()

plt.imshow(outputs, alpha=0.5)
plt.show()