import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import stats

args = None


def train_collate(batch):
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[
        1
    ]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes


def parse_args():
    parser = argparse.ArgumentParser(description="Test ")
    parser.add_argument(
        "--data-dir", default="../../data/UCF_Bayes", help="training data directory"
    )
    parser.add_argument("--save-dir", default="./model.pth", help="model path")
    parser.add_argument("--device", default="0", help="assign device")
    args = parser.parse_args()
    return args


def _nms(heat, kernel):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def convert_heatmap_to_points(
    outputs, nms_kernel_size=3, point_threshold=0.1, max_points=1024
):
    outputs = F.interpolate(outputs, scale_factor=8, mode="bilinear")
    pred_heatmaps_nms = _nms(outputs.detach().clone(), nms_kernel_size)
    pred_points, pred_points_score = (
        torch.zeros(1, max_points, 2).cuda(),
        torch.zeros(1, max_points).cuda(),
    )
    m = 0
    for i in range(1):  # since batch size is 1
        points = torch.nonzero((pred_heatmaps_nms[i] > point_threshold).squeeze())
        points = torch.flip(points, dims=(-1,))
        pred_points_score_ = pred_heatmaps_nms[
            i, 0, points[:, 1], points[:, 0]
        ].flatten(0)

        idx = torch.argsort(pred_points_score_, dim=0, descending=True)[
            : min(max_points, pred_points_score_.size(0))
        ]
        points = points[idx]
        pred_points_score_ = pred_points_score_[idx]

        pred_points[i, : points.size(0)] = points
        pred_points_score[i, : points.size(0)] = pred_points_score_
        m = max(m, points.size(0))

    pred_points = pred_points[:, :m]
    pred_points_score = pred_points_score[:, :m]
    pred_points = pred_points * 4

    return pred_points, pred_points_score


def argmax_points(density_map, th=0.110):
    prob_outputs = F.interpolate(density_map, scale_factor=8, mode="bilinear")
    maxpool_output = F.max_pool2d(prob_outputs, 3, 1, 1)
    maxpool_output = torch.eq(maxpool_output, prob_outputs)
    maxpool_output = maxpool_output.type(torch.cuda.FloatTensor) * prob_outputs
    maxpool_output = maxpool_output[0][0].detach().cpu().numpy()
    maxpool_output[maxpool_output < th] = 0
    y, x = maxpool_output.nonzero()
    pred_points = [[x, y] for x, y in zip(x, y)]
    return pred_points


def plot_regression(pred_counts, gt_counts, save_path="r_square_plot.png"):
    """
    Creates an R-square plot comparing predicted counts vs ground truth counts
    Args:
        pred_counts: List of predicted counts
        gt_counts: List of ground truth counts
        save_path: Path to save the plot
    """
    # Convert to numpy arrays if they aren't already
    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)

    # Calculate R-squared
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        gt_counts, pred_counts
    )
    r_squared = r_value**2

    # Create the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(gt_counts, pred_counts, c="blue", alpha=0.5)

    # Add the regression line
    line = slope * gt_counts + intercept
    plt.plot(gt_counts, line, "r", label=f"R² = {r_squared:.3f}")

    # Add the perfect prediction line (y=x)
    max_count = max(max(pred_counts), max(gt_counts))
    plt.plot([0, max_count], [0, max_count], "--k", label="y=x")

    # Customize the plot
    plt.xlabel("Ground Truth Count", fontsize=12)
    plt.ylabel("Predicted Count", fontsize=12)
    plt.title("Predicted vs Ground Truth Counts", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.7)

    # Make plot square with equal axes
    plt.axis("square")

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.strip()  # set vis gpu

    datasets = Crowd(
        os.path.join(args.data_dir, "test"), 512, 8, method="val"
    )
    dataloader = torch.utils.data.DataLoader(
        datasets, 1, shuffle=False, num_workers=1, pin_memory=False
    )

    model = vgg19()
    device = torch.device("cuda")
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir), device))

    # Use optimal threshold for final evaluation
    epoch_minus = []
    pred_counts = []
    gt_counts = []

    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, "the batch size should equal to 1"
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            # Convert outputs to points
            pred_points, pred_points_score = convert_heatmap_to_points(outputs)
            pred_num = pred_points.shape[1]

            # Store counts for regression plot
            pred_counts.append(pred_num)
            gt_counts.append(len(count[0]))

            temp_minu = len(count[0]) - pred_num
            print(name, "minus", temp_minu, len(count[0]), pred_num, torch.sum(outputs))
            epoch_minus.append(temp_minu)

    # Generate R-square plot
    plot_regression(pred_counts, gt_counts, save_path="counting_regression.png")

    # Calculate metrics
    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = "Final Test: mae {}, mse {}".format(mae, mse)
    print(log_str)
