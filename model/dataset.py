import random
import os
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as npy
from PIL import Image

def _split_phases(folder, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    '''
    Define filenames for Train; Test; Validation phases
    @param folder: img folder or ann folder both work
    @return names in list without suffix
    '''
    assert train_ratio + val_ratio + test_ratio == 1.0, "ratio sum must be 1"

    all_files = os.listdir(folder)
    all_files = [os.path.splitext(file)[0] for file in all_files if os.path.isfile(os.path.join(folder, file))]
    random.shuffle(all_files)
    
    total_files = len(all_files)
    train_split_index = int(total_files * train_ratio)
    val_split_index = train_split_index + int(total_files * val_ratio)
    
    train_files = all_files[:train_split_index]
    val_files = all_files[train_split_index:val_split_index]
    test_files = all_files[val_split_index:]
    
    return train_files, val_files, test_files


class VividDataset(Dataset):
    def __init__(self, data_root, file_list) -> None:
        super(VividDataset, self).__init__()
        self.data_root = data_root
        self.img_path = os.path.join(data_root, 'images')
        self.ann_path = os.path.join(data_root, 'anns')
        self.file_list = file_list
        self.img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        item_name = self.file_list[index]
        img_file_path, dot_ann_path = os.path.join(self.img_path, item_name+'.png'), os.path.join(self.ann_path, item_name+'.npy')
        img = Image.open(img_file_path).convert('RGB')
        if self.img_transform:
            img = self.img_transform(img)
        dot_ann = npy.load(dot_ann_path) #np.ndarray: (n, 2)
        return img, dot_ann
        



if __name__ == '__main__':
    root = '/home/xz/Dev/Dream/data/images'
    train_files, val_files, test_files = _split_phases(root)
    v = VividDataset('/home/xz/Dev/Dream/data', file_list=train_files)
    import pdb; pdb.set_trace()
    print(v[0])
