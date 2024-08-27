from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import sys
from model.dataset import VividDataset, make_dataset
sys.path.insert(0, '/home/xz/Dev/Dream')
from model.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from model.point_decoder import PointDecoder
import torch.nn.functional as F
import torch.nn as nn
import wandb
from datetime import datetime

BATCH_SIZE = 4
ROOT_DIR = '/home/xz/Dev/Dream/data/redo-data'


# TODO: change loss and point numbers
def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    root_dir = '/home/xz/Dev/Dream/data'
    dataset_dict = make_dataset(root_dir)
    train_dataset, val_dataset, test_dataset = dataset_dict['train'], dataset_dict['val'], dataset_dict['test']

    train_loader, val_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=4), DataLoader(val_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    
    sam = build_sam_vit_h().to(device).eval()
    point_mask_decoder = PointDecoder(sam).to(device)
    optimizer = torch.optim.AdamW(list(point_mask_decoder.parameters()), lr=0.0001, weight_decay=0.0)
    mseloss=nn.MSELoss()

    num_epochs = 30

 
    wandb.login()

    run = wandb.init(
        # Set the project where this run will be logged
        project="Vivid",
        name='pseco',
        tags=['init']
    )

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
            pred_heatmaps = point_mask_decoder(features)['pred_heatmaps']
            # import pdb; pdb.set_trace()
            
            loss = mseloss(pred_heatmaps, gt_heatmaps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

       
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


        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.3f}, Validation Loss: {val_loss / len(val_loader)}")

        wandb.log({
            "Train":running_loss / len(train_loader),
            "Val": val_loss / len(val_loader),
        }, step=epoch)

    wandb.finish()
    print("Training complete")
    
    import time

    current_timestamp = time.time()
    time_stamp= time.strftime("%m-%d-%H:%M:%S", time.localtime(current_timestamp))
    filename = f'./weights/final_decoder_{time_stamp}.pth'
    torch.save(point_mask_decoder.state_dict(), filename)
    

if __name__ == '__main__':
    main()