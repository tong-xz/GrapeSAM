import numpy as np
from matplotlib import pyplot as plt



def visualize_img_and_heatmap(img, heatmap, keypoints):
    """
    Visualize an image, its corresponding heatmap, and an image with keypoints side by side.

    :param img: Tensor image of shape (C, H, W)
    :param heatmap: Heatmap tensor of shape (1, H, W)
    :param keypoints: Numpy array of shape (N, 2) representing the (x, y) coordinates of keypoints
    """
    img_np = img.permute(1, 2, 0).numpy()  # 转换为 (H, W, C) 形式

    # 对图像进行反标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)  # 限制值范围到 [0, 1]

    heatmap_np = heatmap.squeeze().numpy()  

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # 显示原图像
    axs[0].imshow(img_np)
    axs[0].set_title('Image')
    axs[0].axis('off')  # 隐藏坐标轴

    # 在图像上绘制关键点
    axs[1].imshow(img_np)
    if len(keypoints) > 0:
        axs[1].scatter(keypoints[:, 0], keypoints[:, 1], s=10, c='red', marker='o')  # 用红色圆点绘制关键点
    axs[1].set_title('Image with keypoints')
    axs[1].axis('off')  # 隐藏坐标轴

    # 显示热力图
    axs[2].imshow(heatmap_np, cmap='hot')  # 使用 'hot' colormap 来显示热力图
    axs[2].set_title('Heatmap')
    axs[2].axis('off')  # 隐藏坐标轴

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