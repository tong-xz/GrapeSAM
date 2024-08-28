
import torchvision.transforms as transforms
import torch
import sys
sys.path.insert(0, '/home/xz/Dev/Dream')
from model.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from model.point_decoder import PointDecoder
import torch.nn.functional as F
import torch.nn as nn
import wandb
from datetime import datetime
import argparse
from model import build_loader
import time


def train(config):
    # build dataloader
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    BATCH_SIZE, EPOCH_NUM, ROOT_DIR, USE_WANDB = config['batch_size'], config['epoch_num'], config['root_dir'], config['wandb']
    loader_dict = build_loader(root_dir=ROOT_DIR, batch_size=BATCH_SIZE)
    train_loader, val_loader, test_loader = loader_dict['train'], loader_dict['val'], loader_dict['test']

    # initialize sam related vairables
    sam = build_sam_vit_h().to(device).eval()
    point_mask_decoder = PointDecoder(sam).to(device)
    optimizer = torch.optim.AdamW(list(point_mask_decoder.parameters()), lr=0.0001, weight_decay=0.0)
    mseloss=nn.MSELoss()

    if USE_WANDB:
        wandb.login()

        run = wandb.init(
            # Set the project where this run will be logged
            project="Vivid",
            name='pseco',
            tags=['init']
        )

    # start training
    for epoch in range(EPOCH_NUM):
        point_mask_decoder.train()
        running_loss = 0.0

        for imgs, heatmaps in train_loader:
            imgs = imgs.to(device) # imgs has to be torch.Size([b, 3, 1024, 1024]) 
            gt_heatmaps = heatmaps.to(device)

            # import pdb; pdb.set_trace()
            # 冻结encoder参数
            with torch.no_grad():
                features = sam.image_encoder(imgs) # torch.Size([b, 256, 64, 64])
            
            # 训练decoder
            optimizer.zero_grad()
            pred_heatmaps = point_mask_decoder(features)['pred_heatmaps']
            import pdb; pdb.set_trace()
            
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

        print(f"Epoch [{epoch + 1}/{EPOCH_NUM}], Loss: {running_loss / len(train_loader):.3f}, Validation Loss: {val_loss / len(val_loader)}")

        if USE_WANDB:
            wandb.log({
                "Train":running_loss / len(train_loader),
                "Val": val_loss / len(val_loader),
            }, step=epoch)

    wandb.finish()
    print("Training complete")
    

    current_timestamp = time.time()
    time_stamp= time.strftime("%m-%d-%H:%M:%S", time.localtime(current_timestamp))
    filename = f'./weights/final_decoder_{time_stamp}.pth'
    torch.save(point_mask_decoder.state_dict(), filename)



def main(config):
    print(f'===========================START============================')
    for k,v in config.items():
        print(f'---Setting {k} as {v}')
    #TODO split train and eval to two functions
    train(config)
    print(f'===========================FINISH===========================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=4, action='store', type=int, required=True)
    parser.add_argument('--epoch_num', default=100, action='store', type=int, required=True)
    parser.add_argument('--root_dir', default='./data/vivid', action='store', type=str, required=True)
    parser.add_argument('--wandb', action='store_true')
    
    args = parser.parse_args()
    config = vars(args) # pass namespace dict 
    main(config)