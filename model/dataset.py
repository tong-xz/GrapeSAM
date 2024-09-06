import random
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from .util import visualize_img_and_heatmap


def _split_phases(root_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    '''
    Define filenames for Train; Test; Validation phases and store in three respective .txt files
    @param folder: root directory of the dataset
    @return names in list without suffix 
    '''
    assert train_ratio + val_ratio + test_ratio == 1.0, "ratio sum must be 1"
    print(f'---Split dataset: train-{train_ratio}; val-{val_ratio}; test-{test_ratio}')

    img_dir = os.path.join(root_dir, 'images')
    all_files = os.listdir(img_dir)
    all_files = [os.path.splitext(file)[0] for file in all_files if os.path.isfile(os.path.join(img_dir, file))]
    random.shuffle(all_files)
    
    total_files = len(all_files)
    train_split_index = int(total_files * train_ratio)
    val_split_index = train_split_index + int(total_files * val_ratio)
    
    train_files = all_files[:train_split_index]
    val_files = all_files[train_split_index:val_split_index]
    test_files = all_files[val_split_index:]

    # create and write list in .txt files
    txt_file_lists = {'train.txt': train_files, 'val.txt': val_files, 'test.txt': test_files}
    for k, v in txt_file_lists.items():
        txt_path = os.path.join(root_dir, k)
        with open(txt_path, 'w') as f:
            for item in v:
                f.write(f'{item}\n')
    
    return train_files, val_files, test_files



def _convert(img, keypoints, target_size=(2048, 2048)):
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

    img = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode='bilinear', align_corners=False).squeeze(0)

    keypoints = keypoints * scale
    
    # Calculate padding
    pad_height = (target_size[0] - new_height) // 2
    pad_width = (target_size[1] - new_width) // 2
    
    # Apply padding to the image
    img = F.pad(img, (pad_width, target_size[1] - new_width - pad_width, pad_height, target_size[0] - new_height - pad_height))

    keypoints[:, 0] += pad_width  # Adjust x coordinates
    keypoints[:, 1] += pad_height # Adjust y coordinates
    
    return img, keypoints



def random_crop(img, keypoints, crop_size=(1024, 1024)):
    """
    Randomly crop the image and keypoints to a specified size.
    
    :param img: Tensor image of shape (C, H, W)
    :param keypoints: Numpy array of shape (N, 2), where N is the number of keypoints and each keypoint is (x, y)
    :param crop_size: Tuple (height, width) specifying the size of the random crop
    :return: Cropped image tensor, adjusted keypoints numpy array
    """
    
    _, original_height, original_width = img.shape
    crop_height, crop_width = crop_size
    
    # Ensure crop size is not larger than the original image size
    assert crop_height <= original_height and crop_width <= original_width, "Crop size must be smaller than image size"
    
    # Randomly choose top-left corner for the crop
    top = random.randint(0, original_height - crop_height)
    left = random.randint(0, original_width - crop_width)
    
    # Crop the image
    img = img[:, top:top + crop_height, left:left + crop_width]
    
    # Adjust keypoints based on the crop
    keypoints[:, 0] -= left  # Adjust x coordinates
    keypoints[:, 1] -= top   # Adjust y coordinates

    # Remove keypoints that are outside the crop
    valid_indices = (keypoints[:, 0] >= 0) & (keypoints[:, 0] <= crop_width) & \
                    (keypoints[:, 1] >= 0) & (keypoints[:, 1] <= crop_height)
    keypoints = keypoints[valid_indices]
    
    return img, keypoints




# def _create_heatmap(img, points, heatmap_size=(256, 256)):
#     '''
#     pseco style heatmap: https://github.com/Hzzone/PseCo/blob/main/fsc147/2_train_heatmap.ipynb
#     '''
#     sigma=0.5
#     scale = 8 # 2048 / 8 = 256
#     # 检查 sigma 是否是 torch.Tensor 类型
#     if not isinstance(sigma, torch.Tensor):
#         sigma = torch.ones(len(points)) * sigma

#     # 缩放点坐标
#     points = points / scale
#     points = torch.tensor(points, dtype=torch.float32)

#     # 生成网格坐标
#     x = torch.arange(0, heatmap_size[0], 1)
#     y = torch.arange(0, heatmap_size[1], 1)
#     x, y = torch.meshgrid(x, y, indexing='xy')
#     x, y = x.unsqueeze(0), y.unsqueeze(0)

#     heatmaps = torch.zeros(1, 1, heatmap_size[0], heatmap_size[1])

#     # 计算每个点的高斯热力图并合并
#     for indices in torch.arange(len(points)):
#         mu_x, mu_y = points[indices, 0].view(-1, 1, 1), points[indices, 1].view(-1, 1, 1)
#         heatmaps_ = torch.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma[indices].view(-1, 1, 1) ** 2))
#         heatmaps_ = torch.max(heatmaps_, dim=0).values
#         heatmaps_ = heatmaps_.reshape(1, 1, heatmap_size[0], heatmap_size[1])
#         heatmaps = torch.maximum(heatmaps, heatmaps_)

#     heatmaps = heatmaps.squeeze(0)
#     img = transforms.Resize(heatmap_size)(img)
#     return heatmaps.float()


def _create_heatmap( points, img_size, heatmap_size=(256, 256), sigma=1.0, normalize=True):
    """
    Generate a heatmap for crowd counting tasks.
    
    :param img: Tensor image
    :param points: Array of points (N, 2)
    :param heatmap_size: Size of the heatmap (height, width)
    :param sigma: Standard deviation for Gaussian kernel
    :param scale: Scale factor to downsize points # 2048 / 8 = 256
    :param normalize: Whether to normalize the heatmap
    :return: Heatmap tensor
    """
    assert isinstance(img_size, tuple) and img_size[0]==img_size[1], "img_size type should be tuple and square shape"
    scale = img_size[0]/heatmap_size[0] # 2048/256 = 8


    if not isinstance(sigma, torch.Tensor):
        sigma = torch.ones(len(points)) * sigma

    points = points / scale
    points = torch.tensor(points, dtype=torch.float32)

    x = torch.arange(0, heatmap_size[0], 1)
    y = torch.arange(0, heatmap_size[1], 1)
    x, y = torch.meshgrid(x, y, indexing='xy')
    x, y = x.unsqueeze(0), y.unsqueeze(0)

    heatmap = torch.zeros(1, 1, heatmap_size[0], heatmap_size[1])

    for i in range(len(points)):
        mu_x, mu_y = points[i, 0].view(-1, 1, 1), points[i, 1].view(-1, 1, 1)
        heatmap_ = torch.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma[i].view(-1, 1, 1) ** 2))
        heatmap_ = heatmap_.reshape(1, 1, heatmap_size[0], heatmap_size[1])
        heatmap += heatmap_

    if normalize:
        heatmap /= heatmap.max()

    heatmap=heatmap.squeeze(1)

    return heatmap.float()




class VividDataset(Dataset):
    def __init__(self, data_root, file_list, mode='train', use_random_crop=False) -> None:
        super(VividDataset, self).__init__()
        self.data_root = data_root
        self.img_path = os.path.join(data_root, 'images')
        self.ann_path = os.path.join(data_root, 'anns')
        self.file_list = file_list
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.mode = mode
        self.use_random_crop = use_random_crop
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        # read img and npy 
        item_name = self.file_list[index]
        img_file_path, dot_ann_path = os.path.join(self.img_path, item_name+'.png'), os.path.join(self.ann_path, item_name+'.npy')
        img = Image.open(img_file_path).convert('RGB')
        if self.img_transform:
            img = self.img_transform(img)
        keypoints = np.load(dot_ann_path) #np.ndarray: (n, 2)

        # random crop images
        if self.use_random_crop:
            if self.mode == 'train':
                img, keypoints = _convert(img, keypoints, target_size=(2048, 2048)) # maybe larger img size
                img, keypoints =random_crop(img, keypoints, crop_size=(1024, 1024))
                heatmap = _create_heatmap(keypoints, img_size=(1024, 1024),sigma=1)
                del keypoints
                return img, heatmap
            else:
                #TODO add crop code when testing
                pass
            
        # use original whole image
        else:
            target_img_size = (1024, 1024) # ViT can only take (1024, 1024) image
            img, keypoints = _convert(img, keypoints, target_img_size) # resize to target size
            heatmap = _create_heatmap(keypoints, img_size=target_img_size, sigma=1) # (1, 256, 256) make heatmap based on the img and points 
            if self.mode == 'train':
                del keypoints
                return img, heatmap
            else:
                return img, keypoints

        

def build_loader(root_dir, batch_size):
    '''
    the only function exposed to the outer class to build dataloaders

    @param ROOT_DIR root dir of the entire dataset
    @param BATCH_SIZE 
    @return dict: three respective dataloaders
    '''
    train_files, val_files, test_files = _split_phases(root_dir)
    train_dataset, test_dataset, val_dataset = VividDataset(root_dir, train_files), VividDataset(root_dir, test_files), VividDataset(root_dir, val_files)

    # loader
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=12, pin_memory=True)
    test_loader =  DataLoader(test_dataset, batch_size, shuffle=True, num_workers=12, pin_memory=True)
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}


if __name__ == '__main__':
    root = '/home/xz/Dev/Dream/data/vivid/'
    train_files, val_files, test_files = _split_phases(root)
    v = VividDataset('/home/xz/Dev/Dream/data/vivid', file_list=train_files)

    img, map= v[0]
    visualize_img_and_heatmap(img, map)
    print(img.shape, map.shape)
