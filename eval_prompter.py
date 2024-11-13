import argparse
import torch
from model import build_loader, build_gsam
from model.point_decoder_n import PointDecoder



def eval(vision_encoder, mask_decoder, test_loader):
    # init model
    total_mae = 0.0
    total_squared_error = 0.0  # Changed from total_mse
    point_decoder.eval()
    point_decoder.max_points = 2048
    point_decoder.nms_kernel_size = 3
    point_decoder.point_threshold = 0.05


    with torch.inference_mode(), torch.no_grad():
        for img, gt_points in test_loader:
            img, gt_points = img.cuda(), gt_points.cuda().sum()
            features = vision_encoder(img)[0]
            
            pred = point_decoder(features)
            pred_points_num = pred["pred_points"].shape[1]
            
            # compare difference with gt and prediciton
            err = abs(gt_points - pred_points_num)
            total_mae += err
            total_squared_error += err**2  # Changed from total_mse
    
    cnt = len(test_loader)
    mae = float(total_mae / cnt)
    rmse = float((total_squared_error / cnt) ** 0.5)  # Changed calculation to RMSE
    
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")  # Updated print statement


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Test script arguments")
    parser.add_argument(
        "--root_dir", type=str, required=True, help="root directory of the dataset folders"
    )
    parser.add_argument("--ckp_path", type=str, required=True, help="checkpoint path")

    args = parser.parse_args()
    
    root_dir, ckp_path = args.root_dir, args.ckp_path
    
    # prepare dataset and everything
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    cfg = {
        'type': 'GSAMVisionEncoder',
        'hf_pretrain_name': "pretrain/sam-vit-huge/",
        'init_cfg': {'checkpoint': '/home/xz/Dev/GrapeSAM/pretrain/sam-vit-huge/pytorch_model.bin'},
        'extra_cfg': None,
        'device': device
    }
    vision_encoder = build_gsam(cfg).to(device).eval()

    cfg1 = {
        'type': 'GSAMMaskDecoder',
        'hf_pretrain_name': "pretrain/sam-vit-huge/",
        'init_cfg': {'checkpoint': '/home/xz/Dev/GrapeSAM/pretrain/sam-vit-huge/pytorch_model.bin'},
        'extra_cfg': None,
        'device': device
    }
    mask_decoder = build_gsam(cfg1).mask_decoder

    point_decoder = PointDecoder(mask_decoder).to(device).eval()

    point_decoder.load_state_dict(torch.load(ckp_path, map_location=device))
    
    test_loader = build_loader(root_dir, batch_size=1)['test']
    
    eval(vision_encoder, mask_decoder, test_loader)


    # python3 eval_prompter.py --root_dir ./data/vivid-6t05/ --ckp_path ./weights/vivid6/point_decoder_11-12-18\:57\:35.pth 