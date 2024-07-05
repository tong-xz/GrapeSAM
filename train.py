from model.Dataset import WgisdDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

BATCH_SIZE = 16
ROOT_DIR = '/Users/tongxiangzhi/Dev/Dream/data/berry_dataset/'

def main():
    train_dir, val_dir, test_dir  = ROOT_DIR + 'train', ROOT_DIR + 'val', ROOT_DIR + 'test'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    wgisd_train, wgisd_val, wgisd_test = WgisdDataset(train_dir, transform), WgisdDataset(val_dir, transform), WgisdDataset(test_dir, transform)
    wgisd_trainloader = DataLoader(wgisd_train, BATCH_SIZE, shuffle=True)
    imgs, heatmaps = next(iter(wgisd_trainloader))
    print(imgs.shape)
    print(heatmaps.shape)

if __name__ == '__main__':
    main()