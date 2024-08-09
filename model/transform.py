import torch.nn.functional as  F

import torch
import torch.nn.functional as F
import numpy as np

def convert(img, keypoints, target_size=(2048, 2048)):
    """
    Resize the image tensor while maintaining aspect ratio, pad to target size, and adjust keypoints accordingly.
    
    :param img: Tensor image of shape (C, H, W)
    :param keypoints: Numpy array of shape (N, 2), where N is the number of keypoints and each keypoint is (x, y)
    :param target_size: Desired output size as a tuple (target_height, target_width)
    :return: Resized and padded image tensor, adjusted keypoints numpy array
    """
    
    _, original_height, original_width = img.shape
    
    # Calculate the scale factor while maintaining aspect ratio
    scale = min(target_size[0] / original_height, target_size[1] / original_width)
    
    # Calculate new size
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)
    
    # Resize the image
    img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)
    
    # Adjust the keypoints
    keypoints = keypoints * scale
    
    # Calculate padding
    pad_height = (target_size[0] - new_height) // 2
    pad_width = (target_size[1] - new_width) // 2
    
    # Apply padding to the image
    img = F.pad(img, (pad_width, target_size[1] - new_width - pad_width, pad_height, target_size[0] - new_height - pad_height))
    
    # Adjust keypoints based on the padding
    keypoints[:, 0] += pad_width  # Adjust x coordinates
    keypoints[:, 1] += pad_height # Adjust y coordinates
    
    return img, keypoints
