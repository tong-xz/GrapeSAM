from torch.utils.data import Dataset
import os
from glob import glob
from PIL import Image
import numpy as np
from torchvision.transforms import transforms
import torch
import  matplotlib.pyplot as plt
import cv2
import torch.nn.functional as  F

import sys
sys.path.insert(0, '/home/xz/Dev/Dream')
from model.ops.utils import rescale_img_points
from model.ops.utils import create_heatmap


class RedoDataset(Dataset):
    def __init__(self, data_path, phase):
        self.data_path = data_path
        self.phase = phase
        self.phase_folder = os.path.join(data_path, phase)
        self.im_list = sorted(glob(os.path.join(self.phase_folder, '*.jpeg')))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.im_list)

    
    def __getitem__(self, index):
        img_path = self.im_list[index % len(self.im_list)]
        gd_path = img_path.replace('jpeg', 'npy')
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        keypoints = np.load(gd_path)
        #TODO add crop
        img, keypoints = rescale_img_points(img, keypoints, img_target_size=(1024, 1024), point_target_size=(256, 256))

        heatmap = create_heatmap(keypoints, kernel_size=3, sigma=0.5, img_size=(256, 256))
        return img, heatmap


    def visualize(self, idx, mode='ann'):
        """_summary_
        visualize data: idx; ann - show annotations; heatmap - show heatmap
        Args:
            idx (_type_): _description_
            mode (str, optional): _description_. Defaults to 'ann'.
        """
        img_path = self.im_list[idx]

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))

        if mode == 'ann':
            ann_path = img_path.replace('jpeg', 'npy')
            dot_ann = np.load(ann_path)[:,:2]
            plt.imshow(image)
            plt.scatter(dot_ann[:,0], dot_ann[:,1], color='r', s=5)
        elif mode == 'heatmap':
            img, heatmap = self.__getitem__(idx)
            
            img = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Unnormalize
            img = np.clip(img, 0, 1)  # Clip to [0, 1]
            plt.imshow(img)
            # import pdb; pdb.set_trace()
            plt.imshow(heatmap.squeeze(), alpha=0.4, cmap='hot')

        plt.axis('off')
        plt.show()


def show_tensor_image_with_dots(tensor_image, keypoints):
    """
    Convert a tensor image to a NumPy array, denormalize it, and display it with keypoints (dots).
    
    :param tensor_image: Tensor image of shape (C, H, W)
    :param keypoints: Numpy array of shape (N, 2) with keypoints (x, y)
    """
    # Denormalize the image if it was normalized (optional step)
    unnormalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    tensor_image = unnormalize(tensor_image)
    
    # Convert the tensor to a NumPy array
    np_image = tensor_image.permute(1, 2, 0).numpy()
    
    # Clip values to [0, 1] for valid image display
    np_image = np.clip(np_image, 0, 1)
    
    plt.figure(figsize=(20, 20))
    # Display the image
    plt.imshow(np_image)
    
    # Overlay the keypoints (dots)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], s=30, c='red', marker='o')
    
    plt.axis('off')  # Hide the axis
    plt.show()



def visualize_heatmap(heatmap, title="Heatmap"):
    heatmap = np.squeeze(heatmap, axis=0)
    plt.figure(figsize=(10, 8))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Density')
    plt.title(title)
    plt.axis('off')  # This removes the axis ticks and labels
    plt.show()



def main():
    path = '/home/xz/Dev/Dream/data/redo-data'
    redo = RedoDataset(path, 'train')
    img, heatmap = redo[10]
    print(heatmap.shape)
    # from torch.utils.data import DataLoader
    
    # redo_loader = DataLoader(redo, batch_size=4)
    # print(next(iter(redo_loader)))
    visualize_heatmap(heatmap)
    # show_tensor_image_with_dots(img, points)



if __name__ == '__main__':
    main()