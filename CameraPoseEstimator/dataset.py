import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import os
from PIL import Image 

DATA_DIR = 'CPEDataset'
BATCH_SIZE = 64
LABEL_DICT = {
    '0': [0.0, 0.0],
    '1': [-0.3, -0.3],
    '2': [-0.3, 0.0],
    '3': [-0.3, 0.3],
    '4': [-0.15, -0.15],
    '5': [-0.15, 0.0],
    '6': [-0.15, 0.15],
    '7': [0.0, -0.3],
    '8': [0.0, -0.15],
    '9': [0.0, 0.3],
    '10': [0.0, 0.15],
    '11': [0.3, -0.3],
    '12': [0.3, 0.0],
    '13': [0.3, 0.3],
    '14': [0.15, -0.15],
    '15': [0.15, 0.0],
    '16': [0.15, 0.15]
}

class CameraPoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes_list = os.listdir(data_dir)
        self.img_dataset = []
        for cls in self.classes_list:
            assert cls in LABEL_DICT, f"Class {cls} not in LABEL_DICT"
            cls_img_list = os.listdir(os.path.join(data_dir, cls))
            cls_img_list = [f for f in cls_img_list if f.endswith('.jpg') or f.endswith('.png')]
            for img_name in cls_img_list:
                img_path = os.path.join(data_dir, cls, img_name)
                self.img_dataset.append((img_path, LABEL_DICT[cls]))
        print(f"Total images found: {len(self.img_dataset)}")

    def __len__(self):
        return len(self.img_dataset)

    def __getitem__(self, idx):
        img_path, label = self.img_dataset[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.float32)
        return image, label

    @staticmethod
    def get_transforms():
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transform

def get_dataloaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE, shuffle=True):
    image_datasets = {x: CameraPoseDataset(os.path.join(data_dir, x), transform=CameraPoseDataset.get_transforms()) for x in ['train', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle) for x in ['train', 'test']}
    return dataloaders