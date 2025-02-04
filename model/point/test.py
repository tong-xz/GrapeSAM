import torch
import os
import numpy as np
from datasets.crowd import Crowd
from models.vgg import vgg19
import argparse
import torch.nn.functional as F

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


def argmax_points(density_map, th=0.1):
    prob_outputs = F.interpolate(density_map, scale_factor=8, mode="bilinear")
    maxpool_output = F.max_pool2d(prob_outputs, 3, 1, 1)
    maxpool_output = torch.eq(maxpool_output, prob_outputs)
    maxpool_output = maxpool_output.type(torch.cuda.FloatTensor) * prob_outputs
    maxpool_output = maxpool_output[0][0].detach().cpu().numpy()
    maxpool_output[maxpool_output < th] = 0
    y, x = maxpool_output.nonzero()
    pred_points = [[x, y] for x, y in zip(x, y)]
    return pred_points


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.strip()  # set vis gpu

    datasets = Crowd(
        os.path.join(args.data_dir, "test"), 512, 8, is_gray=False, method="val"
    )
    dataloader = torch.utils.data.DataLoader(
        datasets, 1, shuffle=False, num_workers=1, pin_memory=False
    )

    import pdb

    pdb.set_trace()

    model = vgg19()
    device = torch.device("cuda")
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir), device))
    epoch_minus = []

    for inputs, count, name in dataloader:
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, "the batch size should equal to 1"
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

            # Convert outputs to points
            pred_points, pred_points_score = convert_heatmap_to_points(outputs)
            # pred_points = argmax_points(outputs)

            temp_minu = len(count[0]) - pred_points.shape[1]
            print(name, temp_minu, len(count[0]), pred_points.shape[1])

            epoch_minus.append(temp_minu)

    epoch_minus = np.array(epoch_minus)
    mse = np.sqrt(np.mean(np.square(epoch_minus)))
    mae = np.mean(np.abs(epoch_minus))
    log_str = "Final Test: mae {}, mse {}".format(mae, mse)
    print(log_str)
