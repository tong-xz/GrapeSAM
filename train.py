from model.Dataset import WgisdDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import sys
sys.path.insert(0, '/Users/tongxiangzhi/Dev/Dream')
from model.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l


BATCH_SIZE = 16
ROOT_DIR = '/Users/tongxiangzhi/Dev/Dream/data/berry_dataset/'


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dir, val_dir, test_dir  = ROOT_DIR + 'train', ROOT_DIR + 'val', ROOT_DIR + 'test'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    wgisd_train, wgisd_val, wgisd_test = WgisdDataset(train_dir, transform), WgisdDataset(val_dir, transform), WgisdDataset(test_dir, transform)
    wgisd_trainloader = DataLoader(wgisd_train, BATCH_SIZE, shuffle=True)
    wgisd_testloader = DataLoader(wgisd_test, BATCH_SIZE, shuffle=True)
    wgisd_valloader = DataLoader(wgisd_val, BATCH_SIZE, shuffle=True)
    
    imgs, heatmaps = next(iter(wgisd_trainloader))
    sam = build_sam_vit_h().cpu().eval()
    with torch.no_grad():
        features = sam.image_encoder(imgs) #torch.Size([16, 256, 64, 64])
    print(features.shape)
    print(heatmaps.shape)

if __name__ == '__main__':
    main()