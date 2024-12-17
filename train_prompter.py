import argparse
from model import build_loader, build_gsam
import torch
from model.point_decoder_n import PointDecoder
import torch.nn as nn
import time
import os
import wandb
import glob
from model.prompter import prompter
import yaml
import numpy as np
import random
'''
new train method based on hf weights and transformer functions
'''

def set_seed(seed):
    # For reproducibility across different runs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For reproducibility on the same machine
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Also need to set random seeds for numpy and random
    np.random.seed(seed)
    random.seed(seed)

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train(config):
    # build dataloader
    BATCH_SIZE = config["batch_size"]
    EPOCH_NUM = config["epoch_num"]
    ROOT_DIR = config["root_dir"]
    USE_WANDB = config["wandb"]
    SAVE_DIR = config["save_dir"]
    CONFIG_PATH = config["config"]

    SAM_CKPT = config["sam_ckpt"]
    HF_PRETRAIN_NAME = config["hf_pretrain_name"]


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loader_dict = build_loader(root_dir=ROOT_DIR, batch_size=BATCH_SIZE)
    train_loader, val_loader, test_loader = (
        loader_dict["train"],
        loader_dict["val"],
        loader_dict["test"],
    )

    cfg = load_config(CONFIG_PATH)
    
    vision_encoder = build_gsam(cfg['vision_encoder']).to(device).eval()
    mask_decoder = build_gsam(cfg['mask_decoder']).mask_decoder

    point_decoder = PointDecoder(mask_decoder).to(device)

    n_parameters = sum(p.numel() for p in point_decoder.parameters() if p.requires_grad)
    print("---Decoder Parameters: %.2fM" % (n_parameters / 1e6,))

    optimizer = torch.optim.AdamW(
        list(point_decoder.parameters()), lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.99)
    ) # 0.0001

    mseloss = nn.MSELoss()

    if USE_WANDB:
        wandb.login()

        run = wandb.init(
            # Set the project where this run will be logged
            project="Vivid-exp",
            name="just feature aggregation",
            tags=["init"],
        )


     # start training
    for epoch in range(EPOCH_NUM):
        start_time = time.time()
        point_decoder.train()
        running_loss = 0.0

        for imgs, heatmaps in train_loader:
            imgs = imgs.to(device)  # imgs has to be torch.Size([b, 3, 1024, 1024])
            gt_heatmaps = heatmaps.to(device)  # ()

            # 冻结encoder参数
            with torch.no_grad():
                vision_outputs = vision_encoder(imgs, output_hidden_states=True)  
                img_embeddings = vision_outputs[0] # torch.Size([b, 256, 64, 64])
                img_hidden_states = vision_outputs[1]
                
                del vision_outputs, img_embeddings

            features = prompter(cfg['prompter'])(img_hidden_states)
            
            # 训练decoder
            optimizer.zero_grad()
            pred_heatmaps = point_decoder(features)["pred_heatmaps"]  # (b, 1, 256, 256)

            loss = mseloss(pred_heatmaps, gt_heatmaps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()


        point_decoder.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps in val_loader:
                imgs = imgs.to(device)
                gt_heatmaps = heatmaps.to(device)

                features = vision_encoder(imgs)[0]
                pred_heatmaps = point_decoder(features)["pred_heatmaps"]

                loss = mseloss(pred_heatmaps, gt_heatmaps)
                val_loss += loss.item()
        
        end_time = time.time()
        print(
            f"Epoch [{epoch + 1}/{EPOCH_NUM}], Train Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Time: {(end_time - start_time)/60:.2f}min"
        )

        if USE_WANDB:
            wandb.log(
                {
                    "Train": running_loss / len(train_loader),
                    "Val": val_loss / len(val_loader),
                    # "MAE": MAE_loss,
                    # "RMSE": RMSE_loss
                },
                step=epoch,
            )

    if USE_WANDB:
        wandb.finish()
        print("Training complete")

    # create tmp dir for intermediate checkpoints
    tmp_save_dir = os.path.join(SAVE_DIR, 'tmp')
    os.makedirs(tmp_save_dir, exist_ok=True)

    # save intermediate checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        tmp_ckp_path = os.path.join(tmp_save_dir, f'point_decoder_epoch_{epoch+1}.pth')
        torch.save(point_decoder.state_dict(), tmp_ckp_path)
        print(f"Checkpoint from epoch {epoch+1} saved at {tmp_ckp_path}")
        
        # keep only latest 3 checkpoints
        tmp_ckps = sorted(glob.glob(os.path.join(tmp_save_dir, '*.pth')))
        if len(tmp_ckps) > 3:
            os.remove(tmp_ckps[0])  # remove oldest checkpoint
    
    # save checkpoint
    current_timestamp = time.time()
    time_stamp = time.strftime("%m-%d-%H:%M:%S", time.localtime(current_timestamp))
    os.makedirs(SAVE_DIR, exist_ok=True)
    ckp_save_path = os.path.join(SAVE_DIR, f"point_decoder_{time_stamp}.pth")
    torch.save(point_decoder.state_dict(), ckp_save_path)
    print(f"Models saved at {ckp_save_path}")



def main(config):
    print(f"===========================START============================")
    for k, v in config.items():
        print(f"---SETTING {k} AS {v}")
    set_seed(42)
    # TODO split train and eval to two functions
    train(config)
    print(f"===========================FINISH===========================")


if __name__ == "__main__":
    # python train.py --batch_size 4 --epoch_num 500 --sam_ckpt ./weights/sam_vit_h_4b8939.pth --wandb
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", default=1, action="store", type=int, required=True
    )
    parser.add_argument(
        "--epoch_num", default=100, action="store", type=int, required=True
    )

    parser.add_argument("--config", action="store", type=str, default="config/prompter_huge.yaml")

    parser.add_argument("--sam_ckpt", action="store", type=str)
    parser.add_argument("--hf_pretrain_name", action="store", type=str)
    parser.add_argument("--root_dir", action="store", type=str)
    parser.add_argument("--save_dir", action="store", type=str)
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()
    config = vars(args)  # pass namespace dict
    main(config)
