from model.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor, build_sam, build_sam_vit_b, build_sam_vit_h, build_sam_vit_l
from model.point_decoder import PointDecoder
import torch
from model.redo_dataset import RedoDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
sam = build_sam_vit_h().to(device).eval()
point_decoder = PointDecoder(sam).to(device)

# 加载已训练的模型参数
point_decoder.load_state_dict(torch.load('final_decoder.pth'))

# 将模型设置为评估模式
point_decoder.eval()

redo_dataset = RedoDataset('/home/xz/Dev/Dream/data/redo-data', 'test')
redo_loader = DataLoader(redo_dataset, batch_size=1)
imgs, heatmaps = next(iter(redo_loader))


imgs = imgs.to(device)

with torch.no_grad():
    features = sam.image_encoder(imgs) 

outputs = point_decoder(features)

print(outputs['pred_points'])

def visualize_points_on_image(tensor_points, img_tensor, point_size=20, alpha=0.7):
    # Convert tensors to numpy if they're not already
    if isinstance(tensor_points, torch.Tensor):
        points = tensor_points.cpu().numpy()
    else:
        points = tensor_points

    if isinstance(img_tensor, torch.Tensor):
        img = img_tensor.cpu().numpy()
    else:
        img = img_tensor

    # Flatten the points if necessary
    if points.ndim > 2:
        points = points.reshape(-1, 2)

    # Prepare the image
    img = img.squeeze().transpose(1, 2, 0)  # Change from (1, 3, 1024, 1024) to (1024, 1024, 3)
    img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0, 1]

    # Create the plot
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.scatter(points[:, 0], points[:, 1], c='red', s=point_size, alpha=alpha)
    
    # Set the plot limits and labels
    plt.xlim(0, img.shape[1])
    plt.ylim(img.shape[0], 0)  # Invert y-axis to match image coordinates
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Point Distribution on Image')
    
    # Remove axis ticks for cleaner look
    plt.xticks([])
    plt.yticks([])
    
    plt.show()


visualize_points_on_image(outputs['pred_points'], imgs)



outputs = outputs['pred_heatmaps_nms'].squeeze().detach().cpu().numpy()

plt.imshow(outputs, alpha=0.5)
plt.show()