import torch.nn.functional as  F

import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter

# def rescale_img_points(img, keypoints, target_size=(2048, 2048)):
#     """
#     Resize the image tensor while maintaining aspect ratio, pad to target size, and adjust keypoints accordingly.
    
#     :param img: Tensor image of shape (C, H, W)
#     :param keypoints: Numpy array of shape (N, 2), where N is the number of keypoints and each keypoint is (x, y)
#     :param target_size: Desired output size as a tuple (target_height, target_width)
#     :return: Resized and padded image tensor, adjusted keypoints numpy array
#     """
    
#     _, original_height, original_width = img.shape
    
#     # Calculate the scale factor while maintaining aspect ratio
#     scale = min(target_size[0] / original_height, target_size[1] / original_width)
    
#     # Calculate new size
#     new_height = int(original_height * scale)
#     new_width = int(original_width * scale)
    
#     # Resize the image
#     img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)
    
#     # Adjust the keypoints
#     keypoints = keypoints * scale
    
#     # Calculate padding
#     pad_height = (target_size[0] - new_height) // 2
#     pad_width = (target_size[1] - new_width) // 2
    
#     # Apply padding to the image
#     img = F.pad(img, (pad_width, target_size[1] - new_width - pad_width, pad_height, target_size[0] - new_height - pad_height))
    
#     # Adjust keypoints based on the padding
#     keypoints[:, 0] += pad_width  # Adjust x coordinates
#     keypoints[:, 1] += pad_height # Adjust y coordinates
    
#     return img, keypoints

def rescale_img_points(img, keypoints, img_target_size=(2048, 2048), point_target_size=(256, 256)):
    """
    Resize the image tensor while maintaining aspect ratio, pad to target size, and adjust keypoints to a different scale,
    ensuring they are within the point target size range.
    
    :param img: Tensor image of shape (C, H, W)
    :param keypoints: Numpy array of shape (N, 2), where N is the number of keypoints and each keypoint is (x, y)
    :param img_target_size: Desired output size for the image as a tuple (target_height, target_width)
    :param point_target_size: Desired scale for the points as a tuple (target_height, target_width)
    :return: Resized and padded image tensor, adjusted keypoints numpy array
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    
    _, original_height, original_width = img.shape
    
    # Calculate the scale factor for the image while maintaining aspect ratio
    img_scale = min(img_target_size[0] / original_height, img_target_size[1] / original_width)
    
    # Calculate new size for the image
    new_height = int(original_height * img_scale)
    new_width = int(original_width * img_scale)
    
    # Resize the image
    img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)
    
    # Calculate padding for the image
    pad_height = (img_target_size[0] - new_height) // 2
    pad_width = (img_target_size[1] - new_width) // 2
    
    # Apply padding to the image
    img = F.pad(img, (pad_width, img_target_size[1] - new_width - pad_width, pad_height, img_target_size[0] - new_height - pad_height))
    
    # Calculate the scale factors for the points
    point_scale_w = point_target_size[1] / img_target_size[1]
    point_scale_h = point_target_size[0] / img_target_size[0]
    
    # Adjust the keypoints
    keypoints = keypoints.astype(np.float32)
    keypoints[:, 0] = (keypoints[:, 0] * img_scale + pad_width) * point_scale_w
    keypoints[:, 1] = (keypoints[:, 1] * img_scale + pad_height) * point_scale_h
    
    # Clip the keypoints to ensure they are within the point target size range
    np.clip(keypoints[:, 0], 0, point_target_size[1] - 1, out=keypoints[:, 0])
    np.clip(keypoints[:, 1], 0, point_target_size[0] - 1, out=keypoints[:, 1])
    
    return img, keypoints

def create_heatmap(points, kernel_size=3, sigma=0.5, img_size=(256, 256)):
    """_summary_
    Generate heatmap follow the idea in: https://github.com/dylran/crowddiff/blob/main/cc_utils/preprocess_ucf.py
    Use conv layer to process the points

    Args:
        points (_type_): _description_
        img_size (tuple, optional): _description_. Defaults to (256, 256).
    """
    density = np.zeros(img_size[:2])
    for x, y in points:
        x, y = int(x), int(y)
        density[y, x]=1.

    # create density kernel
    kernel = np.zeros((kernel_size, kernel_size))
    mid_point = kernel_size // 2
    kernel[mid_point, mid_point]=1
    kernel = gaussian_filter(kernel, sigma=sigma)

    # Guassian kernel
    guassian_kernel = GaussianKernel(kernel)
    density = torch.tensor(density).unsqueeze(0)

    density_map = guassian_kernel(density)
    density_map = density_map.unsqueeze(0).float() #(256, 256) -> (1, 256, 256)
    density_map = density_map.detach().numpy()
    return density_map
    


import torch.nn as nn

class GaussianKernel(nn.Module):
    def __init__(self, kernel_weights):
        super().__init__()
        self.kernel = nn.Conv2d(1, 1, kernel_weights.shape, bias=False, padding='same')
        kernel_weights = torch.tensor(kernel_weights).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            self.kernel.weight = nn.Parameter(kernel_weights)
    
    def forward(self, density):
        return self.kernel(density).squeeze()
