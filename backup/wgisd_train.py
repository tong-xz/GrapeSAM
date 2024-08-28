from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import sys
sys.path.insert(0, '/home/xz/Dev/Dream')
from dataset import WgisdDataset
from model.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from model.point_decoder import PointDecoder
import torch.nn.functional as F
import torch.nn as nn
import wandb
import time

BATCH_SIZE = 4
ROOT_DIR = '/home/xz/Dev/Dream/data/wgisd_dataset/'


# TODO: change loss and point numbers
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

     
    wandb.login()

    run = wandb.init(
        # Set the project where this run will be logged
        project="Vivid",
        name='pseco1',
        tags=['init']
    )
    
    num_epochs = 100
    print(f'===Start===')
    for epoch in range(num_epochs):
        
        point_mask_decoder.train()
        running_loss = 0.0

        for imgs, heatmaps in train_loader:
            imgs = imgs.to(device)
            gt_heatmaps = heatmaps.to(device)

            import pdb; pdb.set_trace()
            # 冻结encoder参数
            with torch.no_grad():
                features = sam.image_encoder(imgs) # torch.Size([16, 256, 64, 64])
            
            # 训练decoder
            optimizer.zero_grad()
            import pdb; pdb.set_trace()
            pred_heatmaps = point_mask_decoder(features)['pred_heatmaps']

            loss = mseloss(pred_heatmaps, gt_heatmaps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        # Validation phase
        point_mask_decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_imgs, val_heatmaps in val_loader:
                val_imgs = imgs.to(device)
                val_gt_heatmaps = heatmaps.to(device)

                val_features = sam.image_encoder(val_imgs)
                val_pred_heatmaps = point_mask_decoder(val_features)['pred_heatmaps']
                loss = mseloss(val_pred_heatmaps, val_gt_heatmaps)
                val_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}, Validation Loss: {val_loss / len(val_loader)}")

        wandb.log({
                "Train":running_loss / len(train_loader),
                "Val": val_loss / len(val_loader),
            }, step=epoch)

    wandb.finish()
    current_timestamp = time.time()
    time_stamp= time.strftime("%m-%d-%H:%M:%S", time.localtime(current_timestamp))
    filename = f'/home/xz/Dev/Dream/weights/final_decoder_{time_stamp}.pth'
    torch.save(point_mask_decoder.state_dict(), filename)
    

if __name__ == '__main__':
    main()