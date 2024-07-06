from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import sys
sys.path.insert(0, '/home/xz/Dev/Dream')
from model.dataset import WgisdDataset
from model.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from model.point_decoder import PointDecoder
import torch.nn.functional as F
import torch.nn as nn

BATCH_SIZE = 4
ROOT_DIR = 'data/berry_dataset/'


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    train_dir, val_dir, test_dir  = ROOT_DIR + 'train', ROOT_DIR + 'val', ROOT_DIR + 'test'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    wgisd_train, wgisd_val, wgisd_test = WgisdDataset(train_dir, transform), WgisdDataset(val_dir, transform), WgisdDataset(test_dir, transform)
    train_loader = DataLoader(wgisd_train, BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(wgisd_test, BATCH_SIZE, shuffle=True,num_workers=4)
    val_loader = DataLoader(wgisd_val, BATCH_SIZE, shuffle=True,num_workers=4)
    

    sam = build_sam_vit_h().to(device).eval()
    point_mask_decoder = PointDecoder(sam).to(device)
    optimizer = torch.optim.AdamW(list(point_mask_decoder.parameters()), lr=0.0001, weight_decay=0.0)
    mseloss=nn.MSELoss()

    imgs, heatmaps = next(iter(train_loader))
    imgs = imgs.to(device)
    gt_heatmaps = heatmaps.to(device)


    num_epochs = 10
    for epoch in range(num_epochs):
        point_mask_decoder.train()
        running_loss = 0.0

        for imgs, heatmaps in train_loader:
            imgs = imgs.to(device)
            gt_heatmaps = heatmaps.to(device)

            # 冻结encoder参数
            with torch.no_grad():
                features = sam.image_encoder(imgs) # torch.Size([16, 256, 64, 64])
            
            # 训练decoder
            optimizer.zero_grad()
            pred_heatmaps = point_mask_decoder(features)['pred_heatmaps']
            loss = mseloss(pred_heatmaps, gt_heatmaps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}.3f")

        # Validation phase
        point_mask_decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps in val_loader:
                imgs = imgs.to(device)
                gt_heatmaps = heatmaps.to(device)

                features = sam.image_encoder(imgs)
                pred_heatmaps = point_mask_decoder(features)['pred_heatmaps']
                loss = mseloss(pred_heatmaps, gt_heatmaps)
                val_loss += loss.item()

        print(f"Validation Loss: {val_loss / len(val_loader)}")

    print("Training complete")
    torch.save(point_mask_decoder.state_dict(), 'decoder.pth')
    

if __name__ == '__main__':
    main()