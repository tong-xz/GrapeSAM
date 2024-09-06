import numpy as np
from matplotlib import pyplot as plt



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




