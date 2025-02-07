import os

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All')}")
import argparse
import torch
import torch.nn as nn
import wandb
import lightning.pytorch as pl
from model import build_loader
from model.utils import load_config
from model.sam_hf import GSamModel
import numpy as np
import random
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger

import numpy as np
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from model.mask.mask2former import add_maskformer2_config
from model.mask.predictor import Mask2FormerRunner

class TrainerLightning(pl.LightningModule):
    def __init__(self, config, devices="cpu"):
        super().__init__()
        # sam config
        # Initialize configurations
        self.config = config
        self.BATCH_SIZE = config["batch_size"]
        self.EPOCH_NUM = config["epoch_num"]
        self.USE_WANDB = config["wandb"]
        self.SAVE_DIR = config["save_dir"]
        self.CONFIG_PATH = config["config"]
        self.devices = devices

        # Load config file and build models
        self.cfg = load_config(self.CONFIG_PATH)

        # DONT USE HARD CODED PATHS
        self.sam_model = GSamModel.from_pretrained(
            self.cfg["vision_encoder"]["hf_pretrain_name"]
        ).to(self.devices)
        self.vision_encoder = self.sam_model.vision_encoder.to(self.devices)
        self.mask_decoder = self.sam_model.mask_decoder.to(self.devices)

        # mask2former
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file("config/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
        cfg.merge_from_list(["MODEL.WEIGHTS", "output/model_0214999.pth"])
        cfg.freeze()
        self.mask2former = Mask2FormerRunner(cfg)

    def forward(self, imgs, coarse_mask=None):
        vision_outputs = self.vision_encoder(imgs, output_hidden_states=True)
        img_embeddings = vision_outputs[0]
        img_hidden_states = vision_outputs[1]
        # del vision_outputs, img_embeddings
        redictions, visualized_output = self.mask2former.run_on_image(imgs)

        # TODO
        fine_mask = self.mask_decoder(
            features,
            dense_prompt_embeddings=coarse_mask,
            # sparse_prompt_embeddings=pred["pred_points"],
            sparse_prompt_embeddings=None,
            image_positional_embeddings=None,
            multimask_output=True,
        )
        return fine_mask

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    # new dataset and dataloader required
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
        # TODO change to coarse mask input pipeline model
        optimizer = torch.optim.AdamW(
            list(self.point_decoder.parameters())
            + list(self.prompter_model.parameters()),  # 添加prompter参数
            lr=1e-4,
            weight_decay=1e-5,
            betas=(0.9, 0.99),
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        # TODO change to coarse mask input
        imgs, heatmaps = batch
        imgs = imgs.to(self.device)
        gt_heatmaps = heatmaps.to(self.device)
        return None

    def validation_step(self, batch, batch_idx): ...

    def on_epoch_end(self):
        if self.USE_WANDB:
            wandb.log(
                {
                    "Train Loss": self.trainer.callback_metrics["train_loss"].item(),
                    "Val MAE": self.trainer.callback_metrics["val_mae"].item(),
                    "Val RMSE": self.trainer.callback_metrics["val_rmse"].item(),
                }
            )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        imgs = batch  # batch is a list of images
        imgs = imgs.to(self.device)
        with torch.no_grad():
            pred = self(imgs)

        return pred


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
            project="Vivid-mask",
            name="coarse mask input",
            tags=["init"],
        )
    else:
        wandb_logger = None

    # Create a PyTorch Lightning Trainer
    # TODO save appripriate checkpoint not point decoder
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


def get_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, action="store", type=int)
    parser.add_argument("--epoch_num", default=100, action="store", type=int)
    parser.add_argument(
        "--config", action="store", type=str, default="config/prompter_aruix.yaml"
    )
    parser.add_argument("--root_dir", action="store", type=str)
    parser.add_argument("--save_dir", action="store", type=str, default="output")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--devices", action="store", type=str, default="cpu")
    args = parser.parse_args()
    config = vars(args)  # pass namespace dict
    return config


if __name__ == "__main__":
    config = get_arg()
    main(config)
