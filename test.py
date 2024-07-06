from model.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from model.point_decoder import PointDecoder
import torch
from model.dataset import WgisdDataset
import torchvision.transforms as transforms



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam = build_sam_vit_h().to(device).eval()
point_decoder = PointDecoder(sam).to(device)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# 加载已训练的模型参数
point_decoder.load_state_dict(torch.load('decoder.pth'))

# 将模型设置为评估模式
point_decoder.eval()

wgisd_test = WgisdDataset('/home/xz/Dev/Dream/data/berry_dataset/test', transform)
print(wgisd_test[0])
