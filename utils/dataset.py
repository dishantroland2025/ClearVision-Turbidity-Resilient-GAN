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
        Args:
            augment (bool): If True, applies Physics-Safe Augmentations (Flip, Rotate, Crop).
                            If False, just Resizes and Normalizes.
            transform (callable): Optional transform to be applied (usually ToTensor + Normalize).
        """
        self.turbid_dir = turbid_dir
        self.clear_dir = clear_dir
        self.depth_dir = depth_dir
        self.transform = transform
        self.augment = augment  # <--- NEW FLAG
        
        # 1. READ ONLY ONE FOLDER (The Turbid one)
        self.image_names = [f for f in os.listdir(turbid_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 2. FORCE NUMERICAL SORTING
        try:
            self.image_names.sort(key=lambda x: int(x.split('.')[0]))
        except ValueError:
            self.image_names.sort()
        
        # 3. SAFETY CHECK
        valid_pairs = []
        for name in self.image_names:
            if os.path.exists(os.path.join(clear_dir, name)) and os.path.exists(os.path.join(depth_dir, name)):
                valid_pairs.append(name)
        self.image_names = valid_pairs

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # Load images
        turbid = Image.open(os.path.join(self.turbid_dir, img_name)).convert("RGB")
        clear = Image.open(os.path.join(self.clear_dir, img_name)).convert("RGB")
        depth = Image.open(os.path.join(self.depth_dir, img_name)).convert("L") 

        # --- SYNCHRONIZED AUGMENTATION LOGIC ---
        # Resize first to ensure we have enough pixel data
        load_size = 286
        target_size = 256
        
        # Always resize to 'load_size' first
        turbid = TF.resize(turbid, (load_size, load_size), interpolation=Image.BICUBIC)
        clear = TF.resize(clear, (load_size, load_size), interpolation=Image.BICUBIC)
        depth = TF.resize(depth, (load_size, load_size), interpolation=Image.NEAREST) # Nearest for depth to preserve values

        if self.augment:
            # 1. Random Crop (Applied identically to all)
            i, j, h, w = user_transforms.RandomCrop.get_params(turbid, output_size=(target_size, target_size))
            turbid = TF.crop(turbid, i, j, h, w)
            clear = TF.crop(clear, i, j, h, w)
            depth = TF.crop(depth, i, j, h, w)

            # 2. Random Horizontal Flip
            if random.random() > 0.5:
                turbid = TF.hflip(turbid)
                clear = TF.hflip(clear)
                depth = TF.hflip(depth)

            # 3. Random Vertical Flip (Physics-Safe for Water)
            if random.random() > 0.5:
                turbid = TF.vflip(turbid)
                clear = TF.vflip(clear)
                depth = TF.vflip(depth)
            
            # 4. Random Rotation (90, 180, 270) - Physics-Safe
            rotations = [0, 90, 180, 270]
            angle = random.choice(rotations)
            if angle > 0:
                turbid = TF.rotate(turbid, angle)
                clear = TF.rotate(clear, angle)
                depth = TF.rotate(depth, angle)
        
        else:
            # Validation Mode: Center Crop only (Deterministic)
            turbid = TF.center_crop(turbid, (target_size, target_size))
            clear = TF.center_crop(clear, (target_size, target_size))
            depth = TF.center_crop(depth, (target_size, target_size))

        # --- FINAL TRANSFORMS (ToTensor + Normalize) ---
        # We apply the passed 'transform' here, but it should only contain ToTensor/Normalize
        if self.transform:
            turbid = self.transform(turbid)
            clear = self.transform(clear)
        
        # Depth is special (0-1, single channel, no normalization)
        depth = TF.to_tensor(depth)
        
        return turbid, clear, depth

