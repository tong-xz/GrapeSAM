import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools.mask import decode

class RedoDataset(Dataset):
    def __init__(self, img_path, json_path, coco_json_path, transform=None):
        super(RedoDataset, self).__init__()
        self.img_path = img_path
        self.transform = transform
        
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.items = self.data['items']
        self.coco = COCO(coco_json_path)

         # Create a mapping from file name to COCO image ID
        self.filename_to_id = {os.path.basename(img['file_name']): img['id'] for img in self.coco.dataset['images']}


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        
        item = self.items[idx]
        
        # Load image
        img_name = item['id'].replace("Redo Examples/", "") + ".jpg"
        img_path = os.path.join(self.img_path, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # Load annotations
        annotations = item['annotations']
        points = [(ann['points'][i], ann['points'][i+1]) for ann in annotations for i in range(0, len(ann['points']), 2)]
        points = torch.tensor(points, dtype=torch.float32)
        
        # Get the corresponding image ID from the file name
        coco_img_id = self.filename_to_id.get(os.path.basename(img_name))
        print(coco_img_id, img_name)
        
        # Load mask from COCO annotations using the image ID
        
        ann_ids = self.coco.getAnnIds(imgIds=coco_img_id, iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)[0]['segmentation']


        if self.transform:
            image = self.transform(image)
        
        return image, points, anns


    def get_img_info(self, idx):
        item = self.items[idx]
        return {"height": item['image']['size'][1], "width": item['image']['size'][0]}

    def visualize(self, idx):
        item = self.items[idx]
        
        # 加载图像
        img_name = item['id'].replace("Redo Examples/", "") + ".jpg"
        img_path = os.path.join(self.img_path, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 获取标注
        annotations = item['annotations']
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)
        
        # 绘制标注点
        for ann in annotations:
            if 'points' in ann:
                points = ann['points']
                x_points = points[0::2]
                y_points = points[1::2]
                ax.scatter(x_points, y_points, c='red', s=20)
        
        ax.set_title(f"Image: {img_name}")
        ax.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    # 定义变换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 创建数据集
    dataset = RedoDataset(
        img_path='/home/xz/Dev/Dream/data/redo/images/output',
        json_path='/home/xz/Dev/Dream/data/redo/annotations/updated_annotations.json',
        coco_json_path='/home/xz/Dev/Dream/data/redo/annotations/instance.json',
        transform=transform
    )
    
    a, b, c = dataset[3]
    print(c)

if __name__ == '__main__':
    main()