import argparse
import torch
from model.dataset import build_loader

def eval(sam, point_decoder, dataloader):
    # init model
    total_mae = 0.0
    total_squared_error = 0.0  # Changed from total_mse
    point_decoder.eval()
    point_decoder.max_points = 2048
    point_decoder.nms_kernel_size = 3
    point_decoder.point_threshold = 0.05

    with torch.inference_mode(), torch.no_grad():
        for img, gt_points in dataloader:
            img, gt_points = img.cuda(), gt_points.cuda().sum()
            features = sam.image_encoder(img)
            
            pred = point_decoder(features)
            pred_points_num = pred["pred_points"].shape[1]
            
            # compare difference with gt and prediciton
            err = abs(gt_points - pred_points_num)
            total_mae += err
            total_squared_error += err**2  # Changed from total_mse
    
    cnt = len(dataloader)
    mae = float(total_mae / cnt)
    rmse = float((total_squared_error / cnt) ** 0.5)  # Changed calculation to RMSE
    
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")  # Updated print statement

    return {
        "mae": mae,
        "rmse": rmse,  # Changed from mse to rmse
    }

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from model import build_sam_vit_h, PointDecoder
    
    parser = argparse.ArgumentParser(description="Test script arguments")
    parser.add_argument(
        "--root_dir", type=str, required=True, help="root directory of the dataset folders"
    )
    parser.add_argument("--ckp_path", type=str, required=True, help="checkpoint path")
    parser.add_argument("--sam_ckpt", type=str, default=None)
    args = parser.parse_args()
    
    root_dir, ckp_path = args.root_dir, args.ckp_path
    
    # prepare dataset and everything
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam = build_sam_vit_h(args.sam_ckpt).to(device).eval()
    point_decoder = PointDecoder(sam).to(device).eval()
    point_decoder.load_state_dict(torch.load(ckp_path, map_location=device))
    
    test_loader = build_loader(root_dir, batch_size=1, use_rcrop=False)['test']
    
    eval(sam, point_decoder, test_loader)