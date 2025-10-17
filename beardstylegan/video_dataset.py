import torch
from PIL import Image
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import os
import logging

DATA_DIR = "temp_dataset"
LANDMARK_DIR = "temp_dataset_landmark"
CROP_SIZE = 512
BATCH_SIZE = 6

class ImagePairDataset(torch.utils.data.Dataset):
    def __init__(self, input_image_dir, landmark_dir, transform = None, device='cuda'):
        print("Input image dir is: ", input_image_dir, os.path.isdir(input_image_dir))
        assert os.path.isdir(input_image_dir), f"Input directory {input_image_dir} not found"
        assert os.path.isdir(landmark_dir), f"Input directory {landmark_dir} not found"
        self.input_image_dir = input_image_dir
        self.landmark_dir = landmark_dir
        self.transform = transform
        self.device = device
        self.means = (0.5, 0.5, 0.5)
        self.std_devs = (0.5, 0.5, 0.5)
        super().__init__()

        self.valid_pairs = []
        list_vid_dirs = os.listdir(self.input_image_dir)
        for vid in list_vid_dirs:
            vid_dir = os.path.join(self.input_image_dir, vid)
            image_list = os.listdir(vid_dir)
            for source_img in image_list:
                for target_img in image_list:
                    if source_img == target_img:
                        continue
                    self.valid_pairs.append((vid, source_img, target_img))
        
        if not self.valid_pairs:
            raise RuntimeError("No valid image pairs found. Check file paths and directory content.")
        
        print("-------- Dataset Initialized Successfully ---------")
        print(f"----- found {len(self.valid_pairs)} valid image pairs -----")

    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        vid, source_img, target_img = self.valid_pairs[idx]
        source_img_path = os.path.join(self.input_image_dir, vid, source_img)
        source_landmark_path = os.path.join(self.landmark_dir, vid, source_img)
        target_img_path = os.path.join(self.input_image_dir, vid, target_img)
        target_landmark_path = os.path.join(self.landmark_dir, vid, target_img)

        try:
            source_image = Image.open(source_img_path).convert('RGB')
            source_landmark = Image.open(source_landmark_path).convert('RGB')
            target_image = Image.open(target_img_path).convert('RGB')
            target_landmark = Image.open(target_landmark_path).convert('RGB')
            sample = {
                'input_image': source_image,
                'landmark_input_image': source_landmark,
                'target_image': target_image,
                'landmark_target_image': target_landmark
            }
            if self.transform:
                sample = self.transform(sample)
            return sample
        except (FileNotFoundError, IOError) as e:
            logging.error(f"Could not read files for index {idx}. File: {e.filename}. Skipping.")
            # Increment index and try again with the next valid pair
            return self.__getitem__((idx + 1) % self.__len__())
    
    class ToTensor(object):
        def __init__(self, device='cuda'):
            self.device = device
        def __call__(self, sample):
            sample['input_image'] = TF.to_tensor(sample['input_image']).float().to(self.device)
            sample['target_image'] = TF.to_tensor(sample['target_image']).float().to(self.device)
            sample['landmark_input_image'] = TF.to_tensor(sample['landmark_input_image']).float().to(self.device)
            sample['landmark_target_image'] = TF.to_tensor(sample['landmark_target_image']).float().to(self.device)
            return sample
    
    class ToResize(object):
        def __init__(self):
            pass

        def __call__(self, sample):
            sample['input_image'] = TF.resize(sample['input_image'], (CROP_SIZE, CROP_SIZE))
            sample['target_image'] = TF.resize(sample['target_image'], (CROP_SIZE, CROP_SIZE))
            sample['landmark_input_image'] = TF.resize(sample['landmark_input_image'], (CROP_SIZE, CROP_SIZE))
            sample['landmark_target_image'] = TF.resize(sample['landmark_target_image'], (CROP_SIZE, CROP_SIZE))
            return sample
    
    @staticmethod
    def get_transforms(device='cuda'):
        transform = transforms.Compose([
            ImagePairDataset.ToResize(),
            ImagePairDataset.ToTensor(device=device),
        ])
        return transform

def get_dataloaders(dataset_dir=DATA_DIR, landmark_dir=LANDMARK_DIR, batch_size=BATCH_SIZE, shuffle=True):
    image_datasets = {x: ImagePairDataset(os.path.join(dataset_dir, x), os.path.join(landmark_dir, x), transform=ImagePairDataset.get_transforms()) for x in ['train']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=0) for x in ['train']}
    return dataloaders_dict