import os

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
import argparse
import torch
import torch.nn as nn
import wandb
import lightning.pytorch as pl
from model import build_loader, build_gsam
from model.point_decoder import PointDecoder
from model.prompter import prompter
from model.utils import load_config
import numpy as np
import random
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger


class TrainerLightning(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        # Initialize configurations
        self.config = config
        self.BATCH_SIZE = config["batch_size"]
        self.EPOCH_NUM = config["epoch_num"]
        self.USE_WANDB = config["wandb"]
        self.SAVE_DIR = config["save_dir"]
        self.CONFIG_PATH = config["config"]
        self.devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config file and build models
        self.cfg = load_config(self.CONFIG_PATH)
        self.vision_encoder = (
            build_gsam(self.cfg["vision_encoder"]).to(self.devices).eval()
        )
        self.mask_decoder = build_gsam(self.cfg["mask_decoder"]).mask_decoder
        self.point_decoder = PointDecoder(self.mask_decoder).to(self.devices)

        # Loss function
        self.mseloss = nn.MSELoss()

        # Log the number of parameters in the model
        n_parameters = sum(
            p.numel() for p in self.point_decoder.parameters() if p.requires_grad
        )
        print("---Decoder Parameters: %.2fM" % (n_parameters / 1e6,))

        # 将prompter作为模型的一个属性
        self.prompter_model = prompter(self.cfg["prompter"]).to(self.devices)

    def forward(self, features):
        return self.point_decoder(features)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train_dataloader(self):
        loader_dict = build_loader(
            root_dir=self.config["root_dir"], batch_size=self.BATCH_SIZE
        )
        return loader_dict["train"]

    def val_dataloader(self):
        loader_dict = build_loader(
            root_dir=self.config["root_dir"], batch_size=self.BATCH_SIZE
        )
        return loader_dict["val"]

    def test_dataloader(self):
        loader_dict = build_loader(
            root_dir=self.config["root_dir"], batch_size=self.BATCH_SIZE
        )
        return loader_dict["test"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            list(self.point_decoder.parameters())
            + list(self.prompter_model.parameters()),  # 添加prompter参数
            lr=1e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.99),
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs, heatmaps = batch
        imgs = imgs.to(self.device)
        gt_heatmaps = heatmaps.to(self.device)

        with torch.no_grad():
            vision_outputs = self.vision_encoder(imgs, output_hidden_states=True)
            img_embeddings = vision_outputs[0]
            img_hidden_states = vision_outputs[1]
            del vision_outputs, img_embeddings

        features = self.prompter_model(img_hidden_states)

        pred_heatmaps = self(features)["pred_heatmaps"]

        loss = self.mseloss(pred_heatmaps, gt_heatmaps)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, gt_point_nums = batch
        imgs = imgs.to(self.device)
        gt_point_nums = gt_point_nums.to(self.device)

        with torch.no_grad():
            vision_outputs = self.vision_encoder(imgs, output_hidden_states=True)
            img_embeddings = vision_outputs[0]
            img_hidden_states = vision_outputs[1]
            del vision_outputs, img_embeddings

            features = self.prompter_model(img_hidden_states)
            pred = self(features)

        pred_point_nums = pred["pred_points"].shape[1]

        mae = torch.abs(gt_point_nums - pred_point_nums).mean()
        rmse = torch.sqrt(torch.mean((gt_point_nums - pred_point_nums) ** 2))

        metrics = {
            "val_mae": mae,
            "val_rmse": rmse,
        }
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=imgs.shape[0],
        )

        return {
            "mae": mae,
            "rmse": rmse,
            "gt_point_nums": gt_point_nums,
            "pred_point_nums": pred_point_nums,
        }

    def on_epoch_end(self):
        if self.USE_WANDB:
            wandb.log(
                {
                    "Train Loss": self.trainer.callback_metrics["train_loss"].item(),
                    "Val MAE": self.trainer.callback_metrics["val_mae"].item(),
                    "Val RMSE": self.trainer.callback_metrics["val_rmse"].item(),
                }
            )


def main(config):
    print(f"===========================START============================")
    for k, v in config.items():
        print(f"---SETTING {k} AS {v}")

    # Initialize model
    model = TrainerLightning(config)

    # Seed for reproducibility
    model.set_seed(42)

    # Initialize wandb logger
    if config["wandb"]:
        wandb.login()
        wandb_logger = WandbLogger(
            project="Vivid-exp",
            name="fpn with original mask decoder",
            tags=["init"],
        )
    else:
        wandb_logger = None

    # Create a PyTorch Lightning Trainer
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["save_dir"], "checkpoints"),
        filename="point_decoder-{epoch:02d}-{val_mae:.2f}",
        monitor="val_mae",
        mode="min",
        save_top_k=5,
    )

    trainer = pl.Trainer(
        max_epochs=config["epoch_num"],
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        strategy=(DDPStrategy(find_unused_parameters=True)),
    )

    # Train the model
    trainer.fit(model)

    if config["wandb"]:
        wandb.finish()

    print(f"===========================FINISH===========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size", default=1, action="store", type=int, required=True
    )
    parser.add_argument(
        "--epoch_num", default=100, action="store", type=int, required=True
    )
    parser.add_argument(
        "--config", action="store", type=str, default="config/prompter_huge.yaml"
    )
    parser.add_argument("--root_dir", action="store", type=str)
    parser.add_argument("--save_dir", action="store", type=str, default="output")
    parser.add_argument("--wandb", action="store_true")

    args = parser.parse_args()
    config = vars(args)  # pass namespace dict
    main(config)
