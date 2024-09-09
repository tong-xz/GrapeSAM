import argparse
import os
import torch
from model import build_sam_vit_h, PointDecoder
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from model import VividDataset


def calculate_MAE(gt_points, pred_points):
    mae = torch.abs(gt_points - pred_points).mean()
    return mae.item()


def calculate_MSE(gt_points, pred_points):
    mse = torch.sqrt(torch.mean((gt_points - pred_points) ** 2))
    return mse.item()


def restore_image_from_quadrants(quadrants):
    """
    Restore the full image from four quadrants and convert it to a PIL image.

    :param quadrants: Dictionary of 4 cropped quadrants with keys '1', '2', '3', '4'
                      Each quadrant should be a tensor of shape (C, H, W).
    :return: Restored image as a PIL Image.
    """
    # Concatenate horizontally (top row: '1' and '2', bottom row: '3' and '4')
    top_row = torch.cat(
        (quadrants["1"], quadrants["2"]), dim=2
    )  # Concatenate along width (W)
    bottom_row = torch.cat(
        (quadrants["3"], quadrants["4"]), dim=2
    )  # Concatenate along width (W)

    # Concatenate vertically (top_row and bottom_row)
    # TODO maybe vit need this
    full_image_tensor = torch.cat(
        (top_row, bottom_row), dim=1
    )  # Concatenate along height (H)

    # Now we integrate the logic from tensor_to_pil directly into this function
    # Define the inverse normalization transformation
    inv_transform = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
    )

    # Apply the inverse normalization
    img = inv_transform(
        full_image_tensor
    )  # Assuming the tensor is normalized with (C, H, W)

    # Convert tensor to NumPy array and transpose to shape (H, W, C)
    img_np = img.cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)

    # Clip values to [0, 1] range for visualization
    img_np = np.clip(img_np, 0, 1)

    # Convert NumPy array to PIL Image
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))

    return img_pil


def main(args):
    # prepare dataset and everything
    root_dir, ckp_path = args.root_dir, args.ckp_path
    test_txt_file = os.path.join(root_dir, "test.txt")

    with open(test_txt_file, "r") as f:
        test_list = [line.strip() for line in f]

    dataset = VividDataset(root_dir, test_list, mode="test")

    # init model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam = build_sam_vit_h().to(device).eval()
    point_mask_decoder = PointDecoder(sam).to(device).eval()
    point_mask_decoder.load_state_dict(torch.load(ckp_path, map_location=device))

    total_mae, total_mse = 0, 0
    count = 0

    for idx, data in enumerate(dataset):
        img_dict, gt_points = data[0], data[1]

        with torch.inference_mode():
            features = sam.image_encoder(img).to(device)

            # Set parameters for point mask decoder
            point_mask_decoder.max_points = 512
            point_mask_decoder.nms_kernel_size = 3
            point_mask_decoder.point_threshold = 0.2

            pred = point_mask_decoder(features)
            pred_points = pred["pred_points"].squeeze()

            # Calculate MAE and MSE
            mae = calculate_MAE(gt_points, pred_points)
            mse = calculate_MSE(gt_points, pred_points)

            print(f"Image {idx}: MAE = {mae}, MSE = {mse}")

            total_mae += mae
            total_mse += mse
            count += 1

    # Calculate overall MAE and MSE
    final_mae = total_mae / count
    final_mse = total_mse / count

    print(f"Final MAE: {final_mae}, Final MSE: {final_mse}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script arguments")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="root directory of the dataset folder",
    )
    parser.add_argument("--ckp_path", type=str, required=True, help="checkpoint path")
    parser.add_argument(
        "--save_dir", type=str, help="Directory to save test image results"
    )
    parser.add_argument(
        "--is_cropped",
        action="store_true",
        help="Indicate whether trained based on cropped image",
    )

    args = parser.parse_args()
    main(args)
