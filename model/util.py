import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import math
from typing import Optional

# ----------------RSPrompter related--------------------------------

class SinePositionalEncoding(nn.Module):
    """Position encoding with sine and cosine functions.

    This implementation follows the method described in
    'End-to-End Object Detection with Transformers' (https://arxiv.org/pdf/2005.12872).

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. Used only when `normalize` is True. Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): Offset added to embedding when doing normalization.
            Defaults to 0.
    """

    def __init__(self,
                 num_feats: int,
                 temperature: int = 10000,
                 normalize: bool = False,
                 scale: float = 2 * math.pi,
                 eps: float = 1e-6,
                 offset: float = 0.) -> None:
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'When normalize is set, scale should be a float or int.'
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: torch.Tensor, input: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward function for SinePositionalEncoding.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values represent
                ignored positions, while zero values mean valid positions.
                Shape [bs, h, w].
            input (Tensor, optional): Input image/feature Tensor.
                Shape [bs, c, h, w].

        Returns:
            pos (Tensor): Position embedding with shape [bs, num_feats*2, h, w].
        """
        assert not (mask is None and input is None), "Either 'mask' or 'input' must be provided."

        if mask is not None:
            B, H, W = mask.size()
            device = mask.device
            mask = mask.to(torch.int)
            not_mask = 1 - mask  # Logical NOT operation
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
        else:
            B, _, H, W = input.shape
            device = input.device
            x_embed = torch.arange(1, W + 1, dtype=torch.float32, device=device).view(1, 1, -1).repeat(B, H, 1)
            y_embed = torch.arange(1, H + 1, dtype=torch.float32, device=device).view(1, -1, 1).repeat(B, 1, W)

        if self.normalize:
            y_embed = (y_embed + self.offset) / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, -1:] + self.eps) * self.scale

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self) -> str:
        """String representation of the module."""
        return (f'{self.__class__.__name__}(num_feats={self.num_feats}, '
                f'temperature={self.temperature}, normalize={self.normalize}, '
                f'scale={self.scale}, eps={self.eps})')



# ----------------SAM related--------------------------------

def predict_masks(predictor, img: np.ndarray, points: torch.Tensor, device="cuda"):
    """_summary_
    predict masks by using points prompts
    Args:
        predictor (_type_): predictor object created through SAM api
        img (np.ndarray): np.ndarray (H, W, 3)
        points (torch.Tensor): annotation points np.ndarray (N, 2)
        device (str, optional): _description_. Defaults to "cuda".

    Returns:
        _type_: _description_
    """
    
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points).to(device)

    
    if isinstance(img, torch.Tensor):
        if img.dim() == 4:
            img = img.squeeze(0).cpu().numpy() # remove batch dim
        else:
            img = img.cpu().numpy()

        assert img.shape[2] == 3, f"img shape should be (H, W, 3) but got {img.shape}"

    points= points.unsqueeze(1)
    transformed_points = predictor.transform.apply_coords_torch(points, img.shape[:2])
    labels = torch.ones(points.shape[0], dtype= torch.long).unsqueeze(1).to(device)

    predictor.set_image(img)

    masks, scores, logits = predictor.predict_torch(
        point_coords=transformed_points,
        point_labels=labels,
        boxes=None,
        multimask_output=True,
    )
    return masks, scores, logits



def vis_pred(image: np.ndarray, masks, save_path=None, dpi=200):
    """_summary_

    Args:
        image (np.ndarray): img numpy array
        masks (_type_): masks get from predictor
        save_path (_type_, optional): detailed path not directory. Defaults to None.
        dpi (int, optional): _description_. Defaults to 200.
    """

    if isinstance(image, torch.Tensor) and image.device.type == "cuda":
        image = image.cpu().numpy()

    if isinstance(masks, torch.Tensor) and masks.device.type == "cuda":
        masks = masks.cpu().numpy()


    plt.figure(figsize=(20, 10))
    
    # Left subplot: original image + masks
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(image)
    _show_masks(masks, ax1, random_colors=True, alpha=0.6)
    ax1.axis('off')
    ax1.set_title("Image with Masks")
    
    # Right subplot: masks only
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(np.zeros_like(image))  
    _show_masks(masks, ax2, random_colors=True, alpha=1.0)  # Use fully opaque masks
    ax2.axis('off')
    ax2.set_title(f"Berry Masks:{masks.shape[0]}")

    # Adjust layout and save the figure
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi)
        print(f'Figure saved to {save_path}')
    else:
        plt.show()    



def _show_masks(masks, ax, random_colors=False, alpha=0.35):
    if len(masks) == 0:
        return
    if masks.ndim == 3:
        num_masks, h, w = masks.shape
    elif masks.ndim == 4:
        num_masks, _, h, w = masks.shape
    else:
        raise ValueError(f"Unexpected mask shape: {masks.shape}")
    
    # Create an RGBA image to store all masks
    masks_image = np.zeros((h, w, 4), dtype=np.float32)
    
    # Set initial transparency to 0
    masks_image[:, :, 3] = 0
    
    # Iterate over each mask and add to masks_image
    for i in range(num_masks):
        if random_colors:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, alpha])
        
        if masks.ndim == 4:
            mask = masks[i, 0]  # Use the first channel
        else:
            mask = masks[i]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        # Only update masks_image where the mask is True
        mask_bool = mask > 0.5
        masks_image[mask_bool] = mask_image[mask_bool]
    
    # Display the masks
    ax.imshow(masks_image)




# ----------------Visualization related--------------------------------
def visualize_img_and_heatmap(img, heatmap=None, keypoints=None):
    """
    Visualize an image, its corresponding heatmap (if exists), and an image with keypoints (if exists) side by side.

    :param img: Tensor image of shape (C, H, W)
    :param heatmap: Heatmap tensor of shape (1, H, W) or None
    :param keypoints: Numpy array of shape (N, 2) representing the (x, y) coordinates of keypoints, or None
    """
    img_np = img.permute(1, 2, 0).numpy()  # 转换为 (H, W, C) 形式

    # 对图像进行反标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)  # 限制值范围到 [0, 1]

    # 准备子图的数量：1个原图，1个可选的keypoints图，1个可选的heatmap图
    num_plots = 1 + (1 if keypoints is not None else 0) + (1 if heatmap is not None else 0)
    fig, axs = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6))

    # 显示原图像
    axs[0].imshow(img_np)
    axs[0].set_title('Image')
    axs[0].axis('off')  # 隐藏坐标轴

    plot_idx = 1

    # 如果有关键点，显示带关键点的图像
    if keypoints is not None and len(keypoints) > 0:
        axs[plot_idx].imshow(img_np)
        axs[plot_idx].scatter(keypoints[:, 0], keypoints[:, 1], s=10, c='red', marker='o')  # 绘制关键点
        axs[plot_idx].set_title('Image with keypoints')
        axs[plot_idx].axis('off')  # 隐藏坐标轴
        plot_idx += 1

    # 如果有热力图，显示热力图
    if heatmap is not None:
        heatmap_np = heatmap.squeeze().numpy()
        axs[plot_idx].imshow(heatmap_np, cmap='hot')  # 显示热力图
        axs[plot_idx].set_title('Heatmap')
        axs[plot_idx].axis('off')  # 隐藏坐标轴

    plt.show()



def visualize_quadrants(quadrants):
    """
    Visualize the four quadrants of the image in the order: 
    top-left (1), top-right (2), bottom-left (3), bottom-right (4),
    with reduced spacing between the images.
    
    :param quadrants: Dictionary of 4 cropped quadrants with keys '1', '2', '3', '4'
    """
    # Convert quadrants to numpy arrays for visualization
    quadrants_np = {k: v.permute(1, 2, 0).numpy() for k, v in quadrants.items()}  # (C, H, W) -> (H, W, C)

    # Clip the data to ensure it's in the valid range for imshow
    quadrants_np = {k: np.clip(v, 0, 1) if v.dtype == np.float32 else np.clip(v, 0, 255).astype(np.uint8) 
                    for k, v in quadrants_np.items()}

    # Create a figure with 2 rows and 2 columns
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Arrange the quadrants in the following order:
    # Top-left -> Top-right -> Bottom-left -> Bottom-right
    positions = [('1', 0, 0), ('2', 0, 1), ('3', 1, 0), ('4', 1, 1)]
    
    for idx, row, col in positions:
        axs[row, col].imshow(quadrants_np[idx])
        axs[row, col].set_title(f'{idx}', color='red')
        axs[row, col].axis('off')  # Hide axes

    # Adjust spacing between plots
    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Reduce horizontal and vertical space

    plt.show()




