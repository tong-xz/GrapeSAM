import numpy as np
from model.dataset import _convert
from model.mask2former_demo import run_on_image
from detectron2.data.detection_utils import read_image
from train import TrainerLightning, get_arg
import torchvision.transforms as transforms
from PIL import Image


def predict_mask(img_path):
    print("Predicting...")
    img_BGR = read_image(img_path, format="BGR")
    predictions, visualized_output = run_on_image(img_BGR)
    return predictions, visualized_output


def predict_points(img_path, coarse_mask):
    img = Image.open(img_path).convert("RGB")
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = img_transform(img)
    img, _ = _convert(img, np.array([[0.0, 0.0]]), (1024, 1024))
    model = TrainerLightning.load_from_checkpoint(
        "output/checkpoints/point_decoder-epoch=05-val_mae=0.01.ckpt",
        config=get_arg(),
        strict=False,
    )
    model = model.to("cpu")
    model = model.eval()
    pred = model(img.unsqueeze(0))
    return pred


if __name__ == "__main__":
    img_paths = [
        "/data/datasets/grape/Vivid/imgs/777.png",
        # "/data/datasets/grape/Vivid/images/778.png",
    ]
    for img_path in img_paths:
        coarse_mask = predict_mask(img_path)
        predictions = predict_points(img_path, coarse_mask)
        breakpoint()
