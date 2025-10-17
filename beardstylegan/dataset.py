import torch
from PIL import Image
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
import os
import logging

# Some parameters used below are defined here - ideally they come from a config file that can be easily changed
INPUT_CLASSES = [1, 6, 7, 8, 9, 11]
TARGET_CLASSES = [1, 6, 7, 8, 9]
IMAGE_SIZE = (1024, 1024)
CROP_SIZE = 512
LOGFILE = "./ageGAN.log"
DATADIR = '/home/shadab/beardstyledata/beardstyledata'
SEG_DIFF_DIR = "/home/shadab/beardstyledata/beardstyledata/masks/masks"
BATCH_SIZE = 6
logging.basicConfig(level=logging.DEBUG, filename=LOGFILE)


def imageLoader(path):
    pass

class ImagePairsDataset(datasets.DatasetFolder):
    def __init__(self, input_image_dir, loader=imageLoader, transform=None, extensions=[".jpg", ".jpeg", ".png"], device='cuda'):
        print("Input image dir is: ", input_image_dir, os.path.isdir(input_image_dir))
        assert os.path.isdir(input_image_dir), f"Input directory {input_image_dir} not found"
        self.input_image_dir = input_image_dir
        self.transform = transform
        self.device = device
        self.means = (0.5, 0.5, 0.5)
        self.std_devs = (0.5, 0.5, 0.5)
        super().__init__(input_image_dir, loader=loader, transform=transform, extensions=extensions)
        self.input_classes = INPUT_CLASSES
        self.target_classes = TARGET_CLASSES

        # --- INTEGRATED FROM dataset2.py: New file discovery and pair validation logic ---
        print("Building valid image pairs from reference directory...")
        try:
            # (i) Use a specified reference class to get the filenames
            reference_class = 1 # Using class '1' as the reference
            class_dir = os.path.join(self.input_image_dir, str(int(reference_class)))
            if not os.path.isdir(class_dir):
                raise FileNotFoundError(f"Reference class directory not found: {class_dir}")

            # Get all potential filenames from the reference directory
            base_files = [
                f for f in os.listdir(class_dir)
                if os.path.splitext(f)[1].lower() in self.extensions
            ]
            print(f"Using class '{reference_class}' as reference. Found {len(base_files)} base images.")
            
            # (ii) Create a list of valid pairs that exist on disk
            self.valid_pairs = []
            for file in base_files:
                for input_class in self.input_classes:
                    for target_class in self.target_classes:
                        # Define paths for the input and target images
                        input_path = os.path.join(self.input_image_dir, str(int(input_class)), file)
                        target_path = os.path.join(self.input_image_dir, str(int(target_class)), file)
                        
                        # A pair is valid only if both image files exist
                        if os.path.exists(input_path) and os.path.exists(target_path):
                            self.valid_pairs.append((input_class, target_class, file))

            if not self.valid_pairs:
                raise RuntimeError("No valid image pairs found. Check file paths and directory content.")
        
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred during pair building: {e}")
            raise e
        # --- END OF INTEGRATED SECTION ---

        print("--------------------")
        print("Dataset Initialized Successfully")
        print(f"   - Found {len(self.valid_pairs)} valid image pairs.")
        print(f"   - Input Classes: {self.input_classes}, Target Classes: {self.target_classes}")
        print(f"   - Total Dataset Size (__len__): {self.__len__()}")
        print("--------------------")

    def __len__(self):
        # The length is the total number of valid pairs found
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        open_succeeded = False
        while not open_succeeded:
            # --- MODIFIED: Get pair information directly from the valid_pairs list ---
            input_class, target_class, filename = self.valid_pairs[idx]
            
            seg_diff_dir = SEG_DIFF_DIR
            filename1 = os.path.join(self.input_image_dir, str(int(input_class)), filename)
            filename2 = os.path.join(self.input_image_dir, str(int(target_class)), filename)
            seg_diff_file = os.path.join(seg_diff_dir, filename)
            
            try:
                image1 = Image.open(filename1).convert('RGB')
                image2 = Image.open(filename2).convert('RGB')
                seg_diff = Image.open(seg_diff_file).convert('L')
                # Print for first few items
                # if idx < 3:
                #     print(f"[Debug idx={idx}] -> File: {filename}, In: {input_class}, Target: {target_class}")
                sample = {
                    'input_image': image1,
                    'target_image': image2,
                    'seg_diff': seg_diff,
                    'input_age': input_class,
                    'target_age': target_class
                }
                if self.transform:
                    sample = self.transform(sample)
                return sample
            except (FileNotFoundError, IOError) as e:
                logging.error(f"Could not read files for index {idx}. File: {e.filename}. Skipping.")
                # Increment index and try again with the next valid pair
                return self.__getitem__((idx + 1) % self.__len__())

    class PairedAugmentations(object):
        """
        Applies consistent augmentations.
        - Resize augmentation is applied ONLY to input and target images.
        - Rotation, Flipping is applied to input, target, and mask.
        """
        def __call__(self, sample):
            input_img = sample['input_image']
            target_img = sample['target_image']
            seg_mask = sample['seg_diff']
            
            # Get original size to resize back to
            original_size = input_img.size # (width, height)

            # --- 1. Resizing Augmentation (Input/Target only) ---
            p_resize = torch.rand(1).item()
            target_resize = None
            if p_resize < 0.1:        # 10% chance for 128x128
                target_resize = (512, 512)
            elif p_resize < 0.2:    # 10% chance for 64x64
                target_resize = (256, 256)
            # elif p_resize < 0.3:    # 10% chance for 256x256
            #     target_resize = (256, 256)

            if target_resize:
                input_img = TF.resize(input_img, target_resize, interpolation=TF.InterpolationMode.BICUBIC)
                # target_img = TF.resize(target_img, target_resize, interpolation=TF.InterpolationMode.BICUBIC)
                input_img = TF.resize(input_img, original_size[::-1], interpolation=TF.InterpolationMode.BICUBIC)
                # target_img = TF.resize(target_img, original_size[::-1], interpolation=TF.InterpolationMode.BICUBIC)

            # --- 2. Random Rotation (All three) ---
            # NEWLY ADDED SECTION
            if torch.rand(1).item() < 0.2: # 50% chance to rotate
                # Generate one random angle between -30 and 30 degrees
                angle = transforms.RandomRotation.get_params(degrees=(-30, 30))
                
                # Apply the same angle to all images
                # Use BICUBIC for smooth color images
                input_img = TF.rotate(input_img, angle, interpolation=TF.InterpolationMode.BICUBIC)
                target_img = TF.rotate(target_img, angle, interpolation=TF.InterpolationMode.BICUBIC)
                # Use NEAREST for the mask to preserve sharp edges and 0/1 values
                seg_mask = TF.rotate(seg_mask, angle, interpolation=TF.InterpolationMode.NEAREST)

            # --- 3. Flipping Augmentation (All three) ---
            if torch.rand(1).item() < 0.3: # 20% chance
                input_img = TF.hflip(input_img)
                target_img = TF.hflip(target_img)
                seg_mask = TF.hflip(seg_mask)

            sample['input_image'] = input_img
            sample['target_image'] = target_img
            sample['seg_diff'] = seg_mask
            return sample
    
    class ToTensor(object):
        def __init__(self, device='cuda'):
            self.device = device
        def __call__(self, sample):
            sample['input_image'] = TF.to_tensor(sample['input_image']).float().to(self.device)
            sample['target_image'] = TF.to_tensor(sample['target_image']).float().to(self.device)
            sample['seg_diff'] = TF.to_tensor(sample['seg_diff']).float().to(self.device)
            return sample
        
    class BottomCentreCrop(object):
        """Crops the given image at the bottom-center."""
        def __init__(self, size=(CROP_SIZE, CROP_SIZE)):
            self.size = size

        def __call__(self, sample):
            # Input is a tensor of shape (C, H, W) because this runs after ToTensor
            input_img = sample['input_image']
            
            # Get image dimensions from one of the tensors
            _, img_h, img_w = input_img.shape
            crop_h, crop_w = self.size

            # Calculate top and left coordinates for the crop
            # For vertical coordinate (i), it's at the bottom
            i = img_h - crop_h
            # For horizontal coordinate (j), it's centered
            j = (img_w - crop_w) // 2
            
            # h and w are just the crop dimensions
            h, w = self.size

            # Apply the same crop to all images in the sample
            sample['input_image'] = transforms.functional.crop(sample['input_image'], i, j, h, w)
            sample['target_image'] = transforms.functional.crop(sample['target_image'], i, j, h, w)
            sample['seg_diff'] = transforms.functional.crop(sample['seg_diff'], i, j, h, w)
            
            return sample

    class RandomCrop(object):
        """Sampling Random Crops of size 512"""
        def __init__(self, size=(CROP_SIZE, CROP_SIZE)):
            self.size = size
            self.crop_transform = transforms.RandomCrop(size)
        def __call__(self, sample):
            # Generate a random crop position
            i, j, h, w = transforms.RandomCrop.get_params(sample['input_image'], output_size=self.size)
            # Apply the crop to both images
            sample['input_image'] = transforms.functional.crop(sample['input_image'], i, j, h, w)
            sample['target_image'] = transforms.functional.crop(sample['target_image'], i, j, h, w)
            sample['seg_diff'] = transforms.functional.crop(sample['seg_diff'], i, j, h, w)
            return sample
    
    class RandomCropWithMaskOverlap(object):
        """
        Random crop transform that ensures a minimum overlap with the white region of seg_diff mask.
        """
        def __init__(self, size=(CROP_SIZE, CROP_SIZE), min_overlap_ratio=0.2, max_attempts=5):
            """
            Args:
                size (tuple): Crop size (height, width)
                min_overlap_ratio (float): Minimum white-pixel ratio in seg_diff (0 to 1)
                max_attempts (int): Number of times to retry for valid crop
            """
            self.size = size
            self.min_overlap_ratio = min_overlap_ratio
            self.max_attempts = max_attempts

        def __call__(self, sample):
            input_img = sample['input_image']    # shape: (C, H, W)
            target_img = sample['target_image']
            seg_mask = sample['seg_diff']        # shape: (1, H, W)

            if seg_mask.dim() == 3 and seg_mask.shape[0] == 1:
                seg_mask = seg_mask.squeeze(0)  # shape: (H, W)

            h, w = input_img.shape[1], input_img.shape[2]
            crop_h, crop_w = self.size

            for _ in range(self.max_attempts):
                top = torch.randint(0, h - crop_h + 1, (1,)).item()
                left = torch.randint(0, w - crop_w + 1, (1,)).item()

                cropped_mask = seg_mask[top:top+crop_h, left:left+crop_w]
                overlap = (cropped_mask > 0.5).float().mean().item()

                if overlap >= self.min_overlap_ratio:
                    sample['input_image'] = input_img[:, top:top+crop_h, left:left+crop_w]
                    sample['target_image'] = target_img[:, top:top+crop_h, left:left+crop_w]
                    sample['seg_diff'] = seg_mask[top:top+crop_h, left:left+crop_w].unsqueeze(0)
                    return sample

            # fallback to center crop
            center_top = (h - crop_h) // 2
            center_left = (w - crop_w) // 2
            sample['input_image'] = input_img[:, center_top:center_top+crop_h, center_left:center_left+crop_w]
            sample['target_image'] = target_img[:, center_top:center_top+crop_h, center_left:center_left+crop_w]
            sample['seg_diff'] = seg_mask[center_top:center_top+crop_h, center_left:center_left+crop_w].unsqueeze(0)

            logging.warning(f"RandomCropWithMaskOverlap: Fallback to center crop after {self.max_attempts} failed attempts.")
            return sample
    
    class ConditionalCrop(object):
        """
        Applies BottomCentreCrop 80% of the time and a random crop 20% of the time.
        """
        def __init__(self, size=(CROP_SIZE, CROP_SIZE), random_crop_prob=0.2):
            self.bottom_crop = ImagePairsDataset.BottomCentreCrop(size)
            # We use the more intelligent RandomCropWithMaskOverlap for the "random" part
            self.random_crop = ImagePairsDataset.RandomCrop(size)
            self.random_crop_prob = random_crop_prob
            print(f"Conditional cropping enabled: {int((1-random_crop_prob)*100)}% BottomCentre, {int(random_crop_prob*100)}% Random.")


        def __call__(self, sample):
            if torch.rand(1).item() < self.random_crop_prob:
                # This block runs 20% of the time
                return self.random_crop(sample)
            else:
                # This block runs 80% of the time
                return self.bottom_crop(sample)


    @staticmethod
    def get_transforms(device='cuda'):
        transform = transforms.Compose([
            # ImagePairsDataset.PairedAugmentations(),
            ImagePairsDataset.ToTensor(device=device),
            # ImagePairsDataset.RandomCrop(),
            ImagePairsDataset.RandomCropWithMaskOverlap(),
            # ImagePairsDataset.ConditionalCrop(random_crop_prob=0.5)
        ])
        return transform

def get_dataloaders(dataset_dir=DATADIR, batch_size=BATCH_SIZE, shuffle=True):
    image_datasets = {x: ImagePairsDataset(os.path.join(dataset_dir, x), transform=ImagePairsDataset.get_transforms()) for x in ['train']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=shuffle, num_workers=0) for x in ['train']}
    return dataloaders_dict


# from tqdm import tqdm
# def testdataloader():
#     loaders = get_dataloaders(shuffle=False)
#     valloader = loaders.get('val')
#     if valloader is None:
#         print("No 'val' loader found in loaders.")
#         return
#     print("Dataloader length is: ", len(valloader))
#     for batch_num, samples in enumerate(tqdm(valloader)):
#         print(f"{batch_num}: aging from {samples['input_age'][0].item()} to {samples['target_age'][0].item()}")
#     display_image_pairs()  # Uncomment and implement if needed

# testdataloader()  # Uncomment to run