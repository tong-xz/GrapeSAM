from model import PointModel, utils, sam
from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import SamModel, SamProcessor
import os
from tqdm import tqdm
import time


class Pipeline:
    def __init__(self, point_model_path, img_save_path) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(
            self.device
        )
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

        # berry part
        self.point_model = PointModel(point_model_path)
        self.trans = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.img_save_path = img_save_path
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)

        # grape cluster part

    def segment_berry(self, img_path, img_title, img_save_path):
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.trans(img).unsqueeze(0).to("cuda")
        # scatter point prediction
        pred_points, pred_points_score = self.point_model(img_tensor)
        # sam segment by points
        best_masks, best_scores = sam.predict_by_points(
            self.sam_model,
            self.sam_processor,
            img,
            pred_points,
            optimal=True,
            multimask_output=True,
        )

        utils.show_masks_on_image(
            img,
            best_masks,
            title=img_title,
            alpha=0.6,
            show_background=False,
            save_path=img_save_path,
        )

        # Release GPU memory
        del img_tensor, pred_points, pred_points_score, best_masks, best_scores
        torch.cuda.empty_cache()

    def segment_grape_cluster(self, img_path): ...

    def process_folder(self, input_folder, format):
        """
        Process all images in the input folder and save the results in the output folder.
        """
        # Get list of files with matching format first
        image_files = [f for f in os.listdir(input_folder) if f.endswith(tuple(format))]

        for filename in tqdm(image_files, desc="Processing images", unit="image"):
            img_path = os.path.join(input_folder, filename)
            berry_img_name = os.path.splitext(filename)[0] + "_berry"
            self.segment_berry(img_path, berry_img_name, self.img_save_path)
            time.sleep(1)  # sleep for gpu memory clear


if __name__ == "__main__":
    ckpt_path = "/home/xz/Dev/baseline-exp-playground/GeneralizedLoss-Counting-Pytorch/output/0203-202057/best_val.pth"
    img_path = "/home/xz/Downloads/feb-test/converted/"
    output_path = "/home/xz/Downloads/feb-test/converted/berry/"
    pipeline = Pipeline(
        ckpt_path,
        output_path,
    )
    pipeline.process_folder(img_path, "png")
