from model.dataset import WgisdDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import sys
import gc
sys.path.insert(0, '/home/xz/Dev/Dream')
from model.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l

BATCH_SIZE = 4
ROOT_DIR = 'data/berry_dataset/'

def print_cuda_memory():
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
    print(f"Max Memory Allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"Max Memory Reserved: {torch.cuda.max_memory_reserved() / 1024 ** 2:.2f} MB")

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_dir, val_dir, test_dir  = ROOT_DIR + 'train', ROOT_DIR + 'val', ROOT_DIR + 'test'
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    wgisd_train, wgisd_val, wgisd_test = WgisdDataset(train_dir, transform), WgisdDataset(val_dir, transform), WgisdDataset(test_dir, transform)
    wgisd_trainloader = DataLoader(wgisd_train, BATCH_SIZE, shuffle=True)
    wgisd_testloader = DataLoader(wgisd_test, BATCH_SIZE, shuffle=True)
    wgisd_valloader = DataLoader(wgisd_val, BATCH_SIZE, shuffle=True)
    
    imgs, heatmaps = next(iter(wgisd_trainloader))

    imgs = imgs.to(device)
    heatmaps = heatmaps.to(device)

    sam = build_sam_vit_h().to(device).eval()

    with torch.no_grad():
        features = sam.image_encoder(imgs)

    print(features.shape)
    print(heatmaps.shape)

    # 打印 CUDA 内存使用情况
    print_cuda_memory()

    # 强制清理 CUDA 内存
    del imgs, heatmaps, features
    torch.cuda.empty_cache()
    gc.collect()

    # 再次打印 CUDA 内存使用情况
    print_cuda_memory()

if __name__ == '__main__':
    main()