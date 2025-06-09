from model import PointModel, utils, sam
from PIL import Image
import torchvision.transforms as transforms
import torch
from transformers import SamModel, SamProcessor
import argparse
import numpy as np
from pathlib import Path


def process_single_image(
    img_path, point_model, sam_model, sam_processor, save_dir, save_vis=True
):
    """Process a single image and save results

    Args:
        img_path: Path to input image
        point_model: Point detection model
        sam_model: SAM model
        sam_processor: SAM processor
        save_dir: Directory to save results
        save_vis: Whether to save visualization results
    """
    img = Image.open(img_path).convert("RGB")

    # Preprocess image
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    img_tensor = trans(img).unsqueeze(0).to(point_model.device)

    # Get predictions
    pred_points, pred_points_score = point_model(img_tensor)

    # Reshape points to 2D array (N x 2)
    points_array = np.array(pred_points)
    points_array = points_array.reshape(-1, 2)

    # Save predicted points
    save_name = img_path.stem
    points_save_path = save_dir / f"{save_name}_points.npy"
    np.save(points_save_path, points_array)

    # Generate and save visualization if requested
    if save_vis:
        # show the points on image
        utils.show_points_on_image(
            img, points_array, title=save_name, save_path=save_dir
        )
        # best_masks, best_scores = sam.predict_by_points(
        #     sam_model,
        #     sam_processor,
        #     img,
        #     pred_points,
        #     optimal=True,
        #     multimask_output=True,
        # )

        # utils.show_masks_on_image(
        #     img,
        #     best_masks,
        #     title=save_name,
        #     alpha=0.6,
        #     show_background=False,
        #     save_path=save_dir,
        # )


def main():
    parser = argparse.ArgumentParser(
        description="Process images with point detection and SAM"
    )
    parser.add_argument(
        "--img-dir", type=str, required=True, help="Directory containing input images"
    )
    parser.add_argument(
        "--point-ckpt",
        type=str,
        default="/home/xz/Dev/baseline-exp-playground/GeneralizedLoss-Counting-Pytorch/output/0203-202057/best_val.pth",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--save-dir", type=str, default="output", help="Directory to save results"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--save-vis", action="store_true", help="Save visualization results"
    )
    parser.add_argument(
        "--sam-pth",
        type=str,
        default="facebook/sam-vit-huge",
        help="Path/name to the sam model checkpoint file",
    )

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    # Initialize models
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    point_model = PointModel(args.point_ckpt)
    sam_model = SamModel.from_pretrained(args.sam_pth).to(device)
    sam_processor = SamProcessor.from_pretrained(args.sam_pth)

    # Process all images in the input directory
    img_dir = Path(args.img_dir)
    supported_formats = [".jpg", ".jpeg", ".png", ".bmp"]
    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() in supported_formats:
            print(f"Processing {img_path}")
            process_single_image(
                img_path, point_model, sam_model, sam_processor, save_dir, args.save_vis
            )


if __name__ == "__main__":
    main()
