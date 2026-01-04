import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as user_transforms
import random

class TurbidDataset(Dataset):
    def __init__(self, turbid_dir, clear_dir, depth_dir, transform=None, augment=False):
        """
        Custom Dataset for Paired Underwater Image Enhancement.
        Handles triplet loading: (Input Turbid, Ground Truth Clear, Depth Map).
        
        Args:
            augment (bool): If True, applies random geometric distortions (Flip/Rotate/Crop) 
                            to prevent overfitting on small datasets (~890 images).
            transform (callable): Normalization pipeline (usually ToTensor + Normalize).
        """
        self.turbid_dir = turbid_dir
        self.clear_dir = clear_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.augment = augment
        
        # 1. File Discovery
        # We only scan the source directory to avoid mismatched list lengths.
        self.image_names = [f for f in os.listdir(turbid_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 2. Sorting Strategy
        # Essential for reproducibility. Tries numerical sorting (1.jpg, 2.jpg) first,
        # falls back to string sorting if filenames are non-numeric.
        try:
            self.image_names.sort(key=lambda x: int(x.split('.')[0]))
        except ValueError:
            self.image_names.sort()
        
        # 3. Integrity Check
        # Pre-scan to ensure every Turbid image has a corresponding Clear and Depth file.
        # This prevents runtime crashes during the training loop.
        valid_pairs = []
        for name in self.image_names:
            has_clear = os.path.exists(os.path.join(clear_dir, name))
            has_depth = os.path.exists(os.path.join(depth_dir, name))
            if has_clear and has_depth:
                valid_pairs.append(name)
        
        self.image_names = valid_pairs
        print(f"Dataset Initialized. Found {len(self.image_names)} valid triplets.")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # --- I/O Operations ---
        # Load Turbid/Clear as RGB (3 channels).
        # Load Depth as 'L' (1 channel grayscale, 0-255).
        turbid = Image.open(os.path.join(self.turbid_dir, img_name)).convert("RGB")
        clear = Image.open(os.path.join(self.clear_dir, img_name)).convert("RGB")
        depth = Image.open(os.path.join(self.depth_dir, img_name)).convert("L") 

        # --- Preprocessing & Resizing ---
        # Upscale to 286px first to allow for a 256px Random Crop (Jitter).
        load_size = 286
        target_size = 256
        
        # Interpolation Physics:
        # - Images: BICUBIC (Smooths gradients, standard for photos).
        # - Depth: NEAREST (Preserves discrete depth values; prevents creating "fake" depths at edges).
        turbid = TF.resize(turbid, (load_size, load_size), interpolation=Image.BICUBIC)
        clear = TF.resize(clear, (load_size, load_size), interpolation=Image.BICUBIC)
        depth = TF.resize(depth, (load_size, load_size), interpolation=Image.NEAREST)

        # --- Synchronized Augmentation ---
        # We must apply the EXACT same geometric transform to all three images 
        # to maintain pixel-wise alignment.
        if self.augment:
            # 1. Random Crop
            i, j, h, w = user_transforms.RandomCrop.get_params(turbid, output_size=(target_size, target_size))
            turbid = TF.crop(turbid, i, j, h, w)
            clear = TF.crop(clear, i, j, h, w)
            depth = TF.crop(depth, i, j, h, w)

            # 2. Random Horizontal Flip (Standard Data Augmentation)
            if random.random() > 0.5:
                turbid = TF.hflip(turbid)
                clear = TF.hflip(clear)
                depth = TF.hflip(depth)

            # 3. Random Vertical Flip
            # Note: While physically debatable (light comes from above), this is acceptable 
            # for increasing dataset variance on small datasets like UIEB.
            if random.random() > 0.5:
                turbid = TF.vflip(turbid)
                clear = TF.vflip(clear)
                depth = TF.vflip(depth)
            
            # 4. Random 90-degree Rotations
            rotations = [0, 90, 180, 270]
            angle = random.choice(rotations)
            if angle > 0:
                turbid = TF.rotate(turbid, angle)
                clear = TF.rotate(clear, angle)
                depth = TF.rotate(depth, angle)
        
        else:
            # Validation/Test Mode:
            # Deterministic Center Crop ensures consistent evaluation metrics.
            turbid = TF.center_crop(turbid, (target_size, target_size))
            clear = TF.center_crop(clear, (target_size, target_size))
            depth = TF.center_crop(depth, (target_size, target_size))

        # --- Tensor Conversion & Normalization ---
        # RGB Images: Apply ToTensor() + Normalize (usually to [-1, 1]).
        if self.transform:
            turbid = self.transform(turbid)
            clear = self.transform(clear)
        
        # Depth Map: No Normalization applied.
        # We keep it as [0, 1] tensor for stability in the Depth-Weighted Loss.
        depth = TF.to_tensor(depth)
        
        return turbid, clear, depth