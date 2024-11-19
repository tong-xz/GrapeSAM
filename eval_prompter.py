import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats
from model import build_loader, build_gsam
from model.point_decoder_n import PointDecoder
from model.ops.ops import plot_results
import os

def tensor_to_pil(tensor_img):
    img = tensor_img.squeeze(0)
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(img_np, 0, 1)
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    return img_pil

def plot_r_square(gt_values, pred_values, save_path='./output/r_square.png'):
    """Plot R-square correlation between ground truth and predicted values"""
    plt.figure(figsize=(10, 10))
    
    # 计算R-square
    slope, intercept, r_value, p_value, std_err = stats.linregress(gt_values, pred_values)
    r_squared = r_value ** 2
    
    # 绘制散点图
    plt.scatter(gt_values, pred_values, c='blue', alpha=0.5)
    
    # 绘制最佳拟合线
    x_range = np.linspace(min(gt_values), max(gt_values), 100)
    plt.plot(x_range, slope * x_range + intercept, 'r', label=f'R² = {r_squared:.4f}')
    
    # 绘制理想的y=x线
    plt.plot([min(gt_values), max(gt_values)], 
             [min(gt_values), max(gt_values)], 
             'k--', alpha=0.5, label='y=x')
    
    plt.xlabel('Ground Truth Count')
    plt.ylabel('Predicted Count')
    plt.title('Correlation between Ground Truth and Predicted Counts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    return r_squared

def eval(vision_encoder, test_loader, vis, save_dir='./output'):
    total_mae = 0.0
    total_squared_error = 0.0
    point_decoder.eval()
    point_decoder.max_points = 2048
    point_decoder.nms_kernel_size = 3
    point_decoder.point_threshold = 0.28 # exp 0.28
    
    # 存储所有的真实值和预测值
    all_gt_counts = []
    all_pred_counts = []

    with torch.inference_mode(), torch.no_grad():
        for img, gt_points in test_loader:
            img, gt_points = img.cuda(), gt_points.cuda().sum()
            features = vision_encoder(img)[0]
            
            pred = point_decoder(features)
            pred_points_num = pred["pred_points"].shape[1]
            err = abs(gt_points - pred_points_num)
            
            # 存储真实值和预测值
            all_gt_counts.append(float(gt_points.cpu()))
            all_pred_counts.append(pred_points_num)
            
            if vis:
                img_pil = tensor_to_pil(img)
                plot_results(img_pil, points=pred['pred_points'].squeeze(), dot_size=8, 
                            save_path=save_dir, error=err)
            
            total_mae += err
            total_squared_error += err**2
    
    cnt = len(test_loader)
    mae = float(total_mae / cnt)
    rmse = float((total_squared_error / cnt) ** 0.5)

    r_squared = 0
    if vis:
        r_squared = plot_r_square(all_gt_counts, all_pred_counts, 
                                save_path=f'{save_dir}/r_square.png')
    
    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r_squared:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script arguments")
    parser.add_argument(
        "--root_dir", type=str, required=True, help="root directory of the dataset folders"
    )
    parser.add_argument("--ckp_path", type=str, required=True, help="checkpoint path")
    parser.add_argument("--output_dir", type=str, default="./output", help="output directory")
    parser.add_argument("--vis", action="store_true", help="whether output visualization images")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    point_decoder.load_state_dict(torch.load(args.ckp_path, map_location=device))
    
    test_loader = build_loader(args.root_dir, batch_size=1)['test']
    
    eval(vision_encoder, test_loader, args.vis,args.output_dir)