import argparse
import torch

from model.dataset import build_loader


def eval(sam, point_mask_decoder, dataloader, use_crop):
    # init model
    total_mae = 0.0
    total_mse = 0.0
    total_nae = 0.0
    total_sre = 0.0
    point_mask_decoder.eval()
    point_mask_decoder.max_points = 1024
    point_mask_decoder.nms_kernel_size = 3
    point_mask_decoder.point_threshold = 0.2
    if not use_crop:
        with torch.inference_mode(), torch.no_grad():
            for img, gt_points in dataloader:
                img, gt_points = img.cuda(), gt_points.cuda().sum()
                features = sam.image_encoder(img)
                # TODO tune these parameters to see the best effect
                
                # not right
                pred = point_mask_decoder(features)

                import pdb; pdb.set_trace()
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
    else:
        return None


if __name__ == "__main__":
   
    from torch.utils.data import DataLoader
    from model import build_sam_vit_h, PointDecoder

    parser = argparse.ArgumentParser(description="Test script arguments")
    parser.add_argument( "--root_dir", type=str, required=True, help="root directory of the dataset folders")
    parser.add_argument("--ckp_path", type=str, required=True, help="checkpoint path")
    parser.add_argument("--sam_ckpt", type=str, default=None)
    parser.add_argument("--use_rcrop", action="store_true", help="if use random crop when training")
    args = parser.parse_args()
    
    root_dir, ckp_path, use_rcrop = args.root_dir, args.ckp_path, args.use_rcrop

    # prepare dataset and everything
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam = build_sam_vit_h(args.sam_ckpt).to(device).eval()
    point_mask_decoder = PointDecoder(sam).to(device).eval()
    point_mask_decoder.load_state_dict(torch.load(ckp_path, map_location=device))


    test_loader = build_loader(root_dir, batch_size=4, use_rcrop=use_rcrop)['test']

    eval(sam, point_mask_decoder, test_loader, use_rcrop)
