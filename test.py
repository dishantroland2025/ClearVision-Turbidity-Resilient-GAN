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
    
    # --- ARCHITECTURE FLAGS (Must match training!) ---
    parser.add_argument("--ngf", type=int, default=80, help="Generator capacity")
    # We use string inputs for booleans to avoid argparse issues with flags
    parser.add_argument("--use_mscc", type=str, default='True', choices=['True', 'False'], help="Use MSCC module?")
    parser.add_argument("--use_triplet", type=str, default='True', choices=['True', 'False'], help="Use Triplet Attention?")
    
    opt = parser.parse_args()

    # Convert string args to boolean
    use_mscc = opt.use_mscc == 'True'
    use_triplet = opt.use_triplet == 'True'

    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opt.results_dir, exist_ok=True)
    
    # 2. Load Model
    print(f" Loading ClearVision (ngf={opt.ngf}, mscc={use_mscc}, triplet={use_triplet})...")
    print(f"  From: {opt.checkpoint_path}")
    
    # Pass the architecture flags to the Generator
    # (Make sure your models/ClearVision.py __init__ accepts these!)
    generator = ClearVisionGenerator(
        ngf=opt.ngf, 
        use_mscc=use_mscc, 
        use_triplet=use_triplet
    ).to(device)
    
    # Load weights
    try:
        checkpoint = torch.load(opt.checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['state_dict'])
        else:
            generator.load_state_dict(checkpoint)
    except Exception as e:
        print(f" Error loading checkpoint: {e}")
        print("   (Did you match the --use_mscc / --use_triplet flags to how this model was trained?)")
        return

    generator.eval()

    # 3. Define Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 4. Process Images
    test_files = sorted(glob(join(opt.turbid_path, "*.*")))
    test_files = [f for f in test_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f" Found {len(test_files)} images in {opt.turbid_path}")
    times = []

    with torch.no_grad():
        for path in test_files:
            img_name = basename(path)
            inp_img = Image.open(path).convert('RGB')
            inp_tensor = transform(inp_img).unsqueeze(0).to(device)
            
            # Inference
            start_time = time.time()
            fake_img = generator(inp_tensor)
            end_time = time.time()
            
            # Warmup logic
            if len(times) > 0: 
                times.append(end_time - start_time)
            else:
                times.append(0) 

            save_path = join(opt.results_dir, img_name)
            save_image(fake_img, save_path, normalize=True)
            # print(f"Processed: {img_name}") # Commented out to reduce spam

    # 5. Statistics
    if len(times) > 1:
        avg_time = np.mean(times[1:])
        fps = 1.0 / avg_time
        print(f"------------------------------------------------")
        print(f" Inference Complete!")
        print(f"  Avg Latency: {avg_time:.4f} sec")
        print(f"  FPS: {fps:.2f}")
        print(f" Saved to: {opt.results_dir}")
        print(f"------------------------------------------------\n")

if __name__ == "__main__":
    main()