import argparse
import torch

from model.dataset import VividDataset, _split_phases


def eval(sam, point_mask_decoder, dataloader):
    # init model
    total_mae = 0.0
    total_mse = 0.0
    total_nae = 0.0
    total_sre = 0.0
    point_mask_decoder.eval()
    point_mask_decoder.max_points = 512
    point_mask_decoder.nms_kernel_size = 3
    point_mask_decoder.point_threshold = 0.2
    with torch.inference_mode(), torch.no_grad():
        for img, gt_points in dataloader:
            img, gt_points = img.cuda(), gt_points.cuda().sum()
            features = sam.image_encoder(img)
            # TODO tune these parameters to see the best effect

            pred = point_mask_decoder(features)
            pred_points = torch.sum(
                pred["pred_points_score"] > point_mask_decoder.point_threshold
            )

            # compare difference with gt and prediciton
            err = abs(gt_points - pred_points)
            total_mae += err
            total_mse += err**2
            total_nae += err / gt_points
            total_sre += err**2 / gt_points

        cnt = len(dataloader)
        mae = float(total_mae / cnt)
        mse = float((total_mse / cnt) ** 0.5)
        nae = float(total_nae / cnt)
        sre = float((total_sre / cnt) ** 0.5)
        print(
            f"[Metrics] MAE: {mae:.2f}, MSE: {mse:.2f}, NAE: {nae:.2f}, SRE: {sre:.2f}"
        )

    return {
        "mae": mae,
        "mse": mse,
        "nae": nae,
        "sre": sre,
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from model import build_sam_vit_h, PointDecoder

    parser = argparse.ArgumentParser(description="Test script arguments")
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="root directory of the dataset folders",
    )
    parser.add_argument("--ckp_path", type=str, required=True, help="checkpoint path")
    parser.add_argument("--sam_ckpt", type=str, default=None)
    args = parser.parse_args()
    # prepare dataset and everything
    root_dir, ckp_path = args.root_dir, args.ckp_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam = build_sam_vit_h(args.sam_ckpt).to(device).eval()
    point_mask_decoder = PointDecoder(sam).to(device).eval()
    point_mask_decoder.load_state_dict(torch.load(ckp_path, map_location=device))
    train_files, val_files, test_files = _split_phases(args.root_dir)
    test_dataset = VividDataset(args.root_dir, test_files, mode="test")
    test_loader = DataLoader(
        test_dataset, 4, shuffle=True, num_workers=4, pin_memory=True
    )
    eval(sam, point_mask_decoder, test_loader)
