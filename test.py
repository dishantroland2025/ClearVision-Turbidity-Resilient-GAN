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
    
    # --- ARCHITECTURE FLAGS ---
    # UPDATED DEFAULT: 48 to match your "Golden" Standard Model
    parser.add_argument("--ngf", type=int, default=48, help="Generator capacity (Must match training!)")
    
    opt = parser.parse_args()

    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opt.results_dir, exist_ok=True)
    
    # 2. Load Model
    print(f" Loading ClearVision Standard (ngf={opt.ngf})...")
    print(f"  From: {opt.checkpoint_path}")
    
    generator = ClearVisionGenerator(ngf=opt.ngf).to(device)
    
    # Load weights
    try:
        checkpoint = torch.load(opt.checkpoint_path, map_location=device)
        
        # FIX: Check for 'G' (used in your training script) or raw state_dict
        if 'G' in checkpoint:
            generator.load_state_dict(checkpoint['G'])
            print("  [INFO] Loaded weights from key: 'G' (Full Checkpoint)")
        elif 'state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['state_dict'])
            print("  [INFO] Loaded weights from key: 'state_dict'")
        else:
            # Assume raw state dict
            generator.load_state_dict(checkpoint)
            print("  [INFO] Loaded raw state dictionary")
            
    except Exception as e:
        print(f"\n[ERROR] Failed to load checkpoint!")
        print(f"Details: {e}")
        print(f"Tip: Did you train with --ngf {opt.ngf}?")
        return

    generator.eval()

    # 3. Define Transforms (Matches Training)
    # Note: If your test images are already 256x256, Resize is harmless. 
    # If they are different aspect ratios, Resize((256,256)) might distort them. 
    # For benchmarking, 256x256 is standard.
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 4. Process Images
    test_files = sorted(glob(join(opt.turbid_path, "*.*")))
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
            
            # Skip first 2 runs (warmup) for timing stats
            if i > 2: 
                times.append(end_time - start_time)

            # Save Result
            save_path = join(opt.results_dir, img_name)
            save_image(fake_img, save_path, normalize=True)
            
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
        print("Warning: Not enough images for FPS calc (Need >2).")

if __name__ == "__main__":
    main()