import os

print(f"Using GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'All')}")
from model.utils import load_config
from model.sam_hf import GSamModel
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from model.mask2former import add_maskformer2_config
from model.predictor import Mask2FormerRunner
import numpy as np
from model.dataset import _convert
from detectron2.data.detection_utils import read_image
from model.sam_hf import GSamModel
import torchvision.transforms as transforms
from PIL import Image


class MaskPipeline:
    def __init__(self, devices="cpu"):
        self.devices = devices
        # Load config file and build models
        self.cfg = load_config("config/prompter_aruix.yaml")

        # DONT USE HARD CODED PATHS
        self.sam_model = GSamModel.from_pretrained(
            "/data/models/sam/huggingface/sam-vit-huge/"
        ).to(self.devices)
        self.vision_encoder = self.sam_model.vision_encoder.to(self.devices)
        self.vision_encoder.eval()
        self.mask_decoder = self.sam_model.mask_decoder.to(self.devices)
        self.mask_decoder.eval()
        self.sam_model.eval()
        # mask2former
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(
            "config/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml"
        )
        cfg.merge_from_list(["MODEL.WEIGHTS", "output/model_0214999.pth"])
        cfg.freeze()
        self.mask2former = Mask2FormerRunner(cfg)

        self.img_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __call__(self, img_path):
        # get corase mask
        img = read_image(img_path, format="BGR")
        coarse_mask, visualized_output = self.mask2former.run_on_image(img)
        breakpoint()

        # get fine mask
        img = Image.open(img_path).convert("RGB")
        img = self.img_transform(img)
        img, _ = _convert(img, np.array([[0.0, 0.0]]), (1024, 1024))
        img = img.unsqueeze(0)
        features = self.vision_encoder(img)
        fine_mask = self.mask_decoder(
            features,
            dense_prompt_embeddings=coarse_mask,
            sparse_prompt_embeddings=None,
            image_positional_embeddings=None,
            multimask_output=True,
        )



if __name__ == "__main__":
    img_paths = [
        "/data/datasets/grape/Vivid/imgs/777.png",
        # "/data/datasets/grape/Vivid/images/778.png",
    ]
    mask_pipeline = MaskPipeline()
    for img_path in img_paths:
        coarse_mask = mask_pipeline(img_path)
        breakpoint()
