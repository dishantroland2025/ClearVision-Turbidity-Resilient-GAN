import os
import time
import argparse
import numpy as np
from PIL import Image
from glob import glob
from os.path import join, basename
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Import your model
from models.ClearVision import ClearVisionGenerator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--turbid_path", type=str, required=True, help="Path to folder containing turbid images")
    parser.add_argument("--results_dir", type=str, default="./results", help="Where to save cleaned images")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to generator .pth file")
    
    # --- ARCHITECTURE FLAGS (Phase 2) ---
    # Default is 32 to match your Jetson optimization
    parser.add_argument("--ngf", type=int, default=32, help="Generator capacity (Must match training!)")
    
    opt = parser.parse_args()

    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opt.results_dir, exist_ok=True)
    
    # 2. Load Model
    print(f" Loading ClearVision Phase 2 (ngf={opt.ngf})...")
    print(f"  From: {opt.checkpoint_path}")
    
    # Initialize Model (Phase 2 has MSCC and Triplet built-in, so no flags needed)
    generator = ClearVisionGenerator(ngf=opt.ngf).to(device)
    
    # Load weights
    try:
        checkpoint = torch.load(opt.checkpoint_path, map_location=device)
        # Handle both raw state_dict and checkpoint dictionaries
        if 'state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['state_dict'])
        else:
            generator.load_state_dict(checkpoint)
        print("  Weights loaded successfully.")
    except Exception as e:
        print(f" Error loading checkpoint: {e}")
        print(f"  (Tip: Ensure you trained with --ngf {opt.ngf})")
        return

    generator.eval()

    # 3. Define Transforms (No Augmentation for Testing)
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 4. Process Images
    test_files = sorted(glob(join(opt.turbid_path, "*.*")))
    # Filter for valid image extensions
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    test_files = [f for f in test_files if f.lower().endswith(valid_exts)]
    
    print(f" Found {len(test_files)} images in {opt.turbid_path}")
    times = []

    with torch.no_grad():
        for i, path in enumerate(test_files):
            img_name = basename(path)
            
            # Load and Preprocess
            inp_img = Image.open(path).convert('RGB')
            inp_tensor = transform(inp_img).unsqueeze(0).to(device)
            
            # Inference & Timing
            if device.type == 'cuda':
                torch.cuda.synchronize() # Precise timing for GPU
            start_time = time.time()
            
            fake_img = generator(inp_tensor)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            # Skip first run (warmup) for timing stats
            if i > 0: 
                times.append(end_time - start_time)

            # Save Result
            save_path = join(opt.results_dir, img_name)
            save_image(fake_img, save_path, normalize=True)
            
            # Optional: Print progress every 10 images
            if (i+1) % 10 == 0:
                print(f"  Processed {i+1}/{len(test_files)}")

    # 5. Statistics
    if len(times) > 0:
        avg_time = np.mean(times)
        fps = 1.0 / avg_time
        print(f"------------------------------------------------")
        print(f" Inference Complete!")
        print(f"  Device: {device}")
        print(f"  Avg Latency: {avg_time*1000:.2f} ms")
        print(f"  FPS: {fps:.2f}")
        print(f" Saved to: {opt.results_dir}")
        print(f"------------------------------------------------\n")
    else:
        print("Warning: No images processed or only 1 image (warmup only).")

if __name__ == "__main__":
    main()