from model import PointModel, utils, sam
from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import SamModel, SamProcessor


ckpt_path = "/home/xz/Dev/baseline-exp-playground/GeneralizedLoss-Counting-Pytorch/output/0203-202057/best_val.pth"
img_path = "/home/xz/Dev/GrapeSAM/data/exp/imgs/25A_back_grape_jpg.png"
save_name = "0203"
img = Image.open(img_path).convert("RGB")

trans = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

img_tensor = trans(img).unsqueeze(0).to("cuda")

point_model = PointModel(ckpt_path)

pred_points, pred_points_score = point_model(img_tensor)

# scaled_img, scaled_keypoints = utils.scale_image_and_keypoints(img, pred_points)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)


# TODO WHAT DOES SEGMENTATION MAP MEAN AND CAN WE DIRECTLY USE IT？
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
best_masks, best_scores = sam.predict_by_points(
    sam_model,
    sam_processor,
    img,
    pred_points,
    optimal=True,
    multimask_output=True,
)
print(best_masks.shape)

utils.show_masks_on_image(
    img, best_masks, title=save_name, alpha=0.6, show_background=False
)
