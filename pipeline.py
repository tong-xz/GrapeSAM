from model import PointModel, utils, sam
from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import SamModel, SamProcessor


class Pipeline:
    def __init__(self, point_model_path) -> None:
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

        # grape cluster part

    def segment_berry(self, img_path, img_title, save_fig):
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
            save_fig=save_fig,
        )

    def segment_grape(self, img_path): ...


if __name__ == "__main__":
    ckpt_path = "/home/xz/Dev/baseline-exp-playground/GeneralizedLoss-Counting-Pytorch/output/0203-202057/best_val.pth"
    img_path = "/home/xz/Downloads/feb-test/converted/IMG_2710.png"
    save_name = "2710"

    pipeline = Pipeline(ckpt_path)
    pipeline.segment_berry(img_path, save_name, True)
