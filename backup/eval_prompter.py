import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats
from model import build_loader, build_gsam
from model.point_decoder import PointDecoder
from model.ops.ops import plot_results
import os
import yaml


def tensor_to_pil(tensor_img):
    img = tensor_img.squeeze(0)
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    return img_pil


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def plot_r_square(gt_values, pred_values, save_path="./output/r_square.png"):
    """Plot R-square correlation between ground truth and predicted values"""
    plt.figure(figsize=(10, 10))

    # 计算R-square
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        gt_values, pred_values
    )
    r_squared = r_value**2

    # 绘制散点图
    plt.scatter(gt_values, pred_values, c="blue", alpha=0.5)

    # 绘制最佳拟合线
    x_range = np.linspace(min(gt_values), max(gt_values), 100)
    plt.plot(x_range, slope * x_range + intercept, "r", label=f"R² = {r_squared:.4f}")

    # 绘制理想的y=x线
    plt.plot(
        [min(gt_values), max(gt_values)],
        [min(gt_values), max(gt_values)],
        "k--",
        alpha=0.5,
        label="y=x",
    )

    plt.xlabel("Ground Truth Count")
    plt.ylabel("Predicted Count")
    plt.title("Correlation between Ground Truth and Predicted Counts")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图片
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()

    return r_squared


def eval(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    # load configuration & setup model
    cfg = load_config(args.config)
    vision_encoder = build_gsam(cfg["vision_encoder"]).to(device).eval()
    mask_decoder = build_gsam(cfg["mask_decoder"]).mask_decoder

    # point decoder setup
    point_decoder = PointDecoder(mask_decoder).to(device)
    point_decoder.load_state_dict(torch.load(args.ckpt_path, map_location=device))
    point_decoder.eval()
    point_decoder.max_points = 2048
    point_decoder.nms_kernel_size = 3
    point_decoder.point_threshold = 0.10  # exp 0.28

    test_loader = build_loader(args.root_dir, batch_size=1)["test"]

    total_mae = 0.0
    total_squared_error = 0.0
    all_gt_counts = []
    all_pred_counts = []
    cnt = 0

    with torch.inference_mode(), torch.no_grad():
        for img, gt_points in test_loader:

            img, gt_points = img.cuda(), gt_points.cuda().sum()

            features = vision_encoder(img)[0]

            pred = point_decoder(features)
            pred_points_num = pred["pred_points"].shape[1]
            err = abs(gt_points - pred_points_num)

            all_gt_counts.append(float(gt_points.cpu()))
            all_pred_counts.append(pred_points_num)
            total_mae += err
            total_squared_error += err**2
            cnt += 1

            if args.vis:
                img_pil = tensor_to_pil(img)
                plot_results(
                    img_pil,
                    points=pred["pred_points"].squeeze().cpu(),
                    dot_size=8,
                    save_path=args.output_dir,
                    error=err,
                )

    mae = float(total_mae / cnt)
    rmse = float((total_squared_error / cnt) ** 0.5)

    r_squared = 0
    if args.vis:
        r_squared = plot_r_square(
            all_gt_counts, all_pred_counts, save_path=f"{args.save_dir}/r_square.png"
        )

    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r_squared:.4f}, cnt: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script arguments")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="root directory of the dataset folders",
    )
    parser.add_argument("--ckpt_path", type=str, required=True, help="checkpoint path")
    parser.add_argument(
        "--output_dir", type=str, default="./output2", help="output directory"
    )
    parser.add_argument("--config", type=str, required=True, help="config path")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--vis", action="store_true", help="whether output visualization images"
    )

    args = parser.parse_args()
    eval(args)
