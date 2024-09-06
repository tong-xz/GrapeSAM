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



def visualize(img_tensor, keypoints):
    """
    For annotation visualization
    Visualize a tensor image with keypoints.
    """
    # Convert tensor to numpy format for visualization
    img_array = img_tensor.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
    
    # Normalize the image to [0, 1] range if it has been standardized
    img_array = img_array * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_array = np.clip(img_array, 0, 1)

    fig, ax = plt.subplots()
    ax.imshow(img_array)

    # Plot keypoints
    x, y = keypoints[:, 0], keypoints[:, 1]
    ax.scatter(x, y, c='r', s=10)  

    plt.axis('off')  
    plt.show()