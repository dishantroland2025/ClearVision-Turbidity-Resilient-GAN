import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class TurbidDataset(Dataset):
    def __init__(self, turbid_dir, clear_dir, depth_dir, transform=None):
        self.turbid_dir = turbid_dir
        self.clear_dir = clear_dir
        self.depth_dir = depth_dir
        self.transform = transform
        
        # 1. READ ONLY ONE FOLDER (The Turbid one)
        # We use this as the "Master List"
        self.image_names = [f for f in os.listdir(turbid_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # 2. FORCE NUMERICAL SORTING
        # Tries to turn "10.png" -> 10. Fallback to string sort if it fails.
        try:
            self.image_names.sort(key=lambda x: int(x.split('.')[0]))
        except ValueError:
            self.image_names.sort()
        
        # 3. SAFETY CHECK: Ensure the triplet exists (Turbid + Clear + Depth)
        valid_pairs = []
        for name in self.image_names:
            has_clear = os.path.exists(os.path.join(clear_dir, name))
            has_depth = os.path.exists(os.path.join(depth_dir, name))
            
            if has_clear and has_depth:
                valid_pairs.append(name)
            else:
                print(f"⚠️ Skipping {name}: Missing Clear or Depth counterpart.")
        
        self.image_names = valid_pairs
        
        # 4. Define Depth-Specific Transform
        # Depth maps are Grayscale (1 channel), so we don't normalize them with RGB mean/std
        self.depth_transform = transforms.Compose([
            transforms.Resize((256, 256)), # Must match RGB size
            transforms.ToTensor()          # Converts 0-255 pixels to 0.0-1.0 float
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        
        # Construct paths
        turbid_path = os.path.join(self.turbid_dir, img_name)
        clear_path = os.path.join(self.clear_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name)
        
        # Load images
        # Turbid/Clear -> RGB (3 channels)
        turbid_img = Image.open(turbid_path).convert("RGB")
        clear_img = Image.open(clear_path).convert("RGB")
        # Depth -> Grayscale (1 channel)
        depth_img = Image.open(depth_path).convert("L") 
        
        # Apply transforms
        if self.transform:
            turbid_img = self.transform(turbid_img)
            clear_img = self.transform(clear_img)
        
        # Apply separate transform for depth
        depth_img = self.depth_transform(depth_img)
        
        return turbid_img, clear_img, depth_img