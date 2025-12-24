import argparse
import ssl
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import json
import random

ssl._create_default_https_context = ssl._create_unverified_context

# --- IMPORTS ---
from models.ClearVision import ClearVisionGenerator, Discriminator
from utils.dataset import TurbidDataset
from utils.losses import generator_loss, PerceptualLoss

# --- CONFIGURATION ---
def get_args():
    parser = argparse.ArgumentParser()
    
    # Basic Training Args
    parser.add_argument("--name", type=str, default="experiment", help="Name of the experiment")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    
    # Optimization Args (Updated for Asymmetric LR & AMP)
    parser.add_argument("--lr", type=float, default=0.0002, help="Generator Learning rate")
    parser.add_argument("--d_lr", type=float, default=0.0002, help="Discriminator Learning rate")
    parser.add_argument("--use_amp", action='store_true', help="Use Automatic Mixed Precision (Faster)")
    
    parser.add_argument("--img_size", type=int, default=256, help="Input image resolution")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Where to save .pth files")
    parser.add_argument("--sample_dir", type=str, default="samples/", help="Where to save test images")
    
    # Model Architecture Args
    parser.add_argument("--ngf", type=int, default=64, help="Generator capacity (64=base, 80=high)")
    parser.add_argument("--ndf", type=int, default=64, help="Discriminator capacity (NEW - Default 64)") # <--- ADDED
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs with fixed learning rate")
    parser.add_argument("--n_epochs_decay", type=int, default=100, help="Number of epochs with decaying learning rate")

    # Ablation Flags
    parser.add_argument("--use_mscc", type=bool, default=True, help="Use Multi-Scale Color Correction")
    parser.add_argument("--use_triplet", type=bool, default=True, help="Use Triplet Attention")

    # --- LOSS WEIGHT FLAGS ---
    parser.add_argument('--lambda_adv', type=float, default=0.1, help='Weight for Adversarial Loss')
    parser.add_argument('--lambda_pixel', type=float, default=10.0, help='Weight for L1 Pixel Loss')
    parser.add_argument('--lambda_color', type=float, default=5.0, help='Weight for Color Consistency Loss')
    parser.add_argument('--lambda_edge', type=float, default=2.0, help='Weight for Edge/Gradient Loss')
    parser.add_argument('--lambda_perc', type=float, default=1.0, help='Weight for Perceptual Loss')
    parser.add_argument('--lambda_depth', type=float, default=5.0, help='Weight for Depth-Attention Loss')

    # Data Paths
    parser.add_argument("--turbid_path", type=str, required=True, help="Path to Turbid images")
    parser.add_argument("--clear_path", type=str, required=True, help="Path to Clear (GT) images")
    parser.add_argument("--depth_path", type=str, required=True, help="Path to Depth maps")
    
    return parser.parse_args()

def main():
    opt = get_args()
    
    # 1. Setup Directories
    if opt.checkpoint_dir == "checkpoints/":
        opt.checkpoint_dir = os.path.join("checkpoints", opt.name)
    
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    os.makedirs(opt.sample_dir, exist_ok=True)

    # 2. AUTO-SAVE CONFIGURATION
    config_path = os.path.join(opt.checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(opt), f, indent=4)
    print(f" Configuration saved to: {config_path}")

    # --- DEVICE SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Starting ClearVision Training on: {device}")
    print(f"   Experiment: {opt.name}")
    print(f"   Gen Capacity (ngf): {opt.ngf}")
    print(f"   Disc Capacity (ndf): {opt.ndf}") # <--- LOGGING
    print(f"   AMP Enabled: {opt.use_amp}")

    # --- INITIALIZE MODELS ---
    # Pass 'ndf' to Discriminator to allow balancing the architecture
    generator = ClearVisionGenerator(ngf=opt.ngf).to(device)
    discriminator = Discriminator(ndf=opt.ndf).to(device) # <--- UPDATED

    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # --- OPTIMIZERS ---
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.d_lr, betas=(0.5, 0.999))

    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - opt.n_epochs) / float(opt.n_epochs_decay + 1)
        return lr_l

    # Apply the rule to both optimizers
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    print(f"   LR Scheduler: Fixed for {opt.n_epochs}, Decay for {opt.n_epochs_decay}")

    # --- SCALER FOR AMP ---
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)

    # --- PERCEPTUAL LOSS SETUP ---
    print("Loading VGG19 for Perceptual Loss...")
    from torchvision.models import vgg19
    vgg = vgg19(pretrained=True).features.to(device)
    perceptual_fn = PerceptualLoss(vgg).to(device)

    # --- DYNAMIC LOSS WEIGHTS ---
    lambdas = {
        "adv":   opt.lambda_adv,  
        "pixel": opt.lambda_pixel, 
        "color": opt.lambda_color,  
        "edge":  opt.lambda_edge,  
        "perc":  opt.lambda_perc,  
        "depth": opt.lambda_depth   
    }

    # --- DATA LOADING (WITH AUGMENTATION) ---
    # We resize slightly larger (286) then random crop (256) to prevent overfitting
    load_size = int(opt.img_size * 1.12)  # 256 -> 286
    crop_size = opt.img_size

    transforms_ = transforms.Compose([
        # 1. Resize slightly larger so we can crop
        transforms.Resize((load_size, load_size), transforms.InterpolationMode.BICUBIC),
        
        # 2. Random Crop (Simulates different viewing distances)
        transforms.RandomCrop((crop_size, crop_size)),
        
        # 3. Flips (Valid because water has no "up/down" gravity)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),   # <--- NEW: Doubles your data
        
        # 4. Rotation (Crucial for learning textures at all angles)
        # We use 90-degree increments to avoid black borders
        transforms.RandomChoice([
            transforms.RandomRotation((0, 0)),    # 0 deg
            transforms.RandomRotation((90, 90)),  # 90 deg
            transforms.RandomRotation((180, 180)),# 180 deg
            transforms.RandomRotation((270, 270)) # 270 deg
        ]), 
        
        # 5. Tensor & Normalize
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = TurbidDataset(opt.turbid_path, opt.clear_path, opt.depth_path, transform=transforms_)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"Loaded {len(dataset)} training triplets with Data Augmentation.")

    # --- TRAINING LOOP ---
    for epoch in range(opt.epochs):
        loop = tqdm(dataloader, leave=True)
        
        for i, (turbid, clear, depth) in enumerate(loop):
            turbid = turbid.to(device)
            clear = clear.to(device)
            depth = depth.to(device)

            # ---------------------
            #  1. Train Generator
            # ---------------------
            optimizer_G.zero_grad()

            # Automatic Mixed Precision Context
            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                fake_clear = generator(turbid)
                loss_G, loss_dict = generator_loss(
                    discriminator, clear, fake_clear, turbid, 
                    depth=depth, 
                    perceptual_fn=perceptual_fn, 
                    lambdas=lambdas
                )

            # Scaled Backward Pass
            scaler.scale(loss_G).backward()

            # Unscale before clipping to ensure correct norm calculation
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)

            scaler.step(optimizer_G)
            scaler.update()

            # -------------------------
            #  2. Train Discriminator
            # -------------------------
            optimizer_D.zero_grad()

            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                pred_real = discriminator(clear, turbid)
                loss_real = torch.mean((pred_real - 1.0) ** 2)

                pred_fake = discriminator(fake_clear.detach(), turbid)
                loss_fake = torch.mean((pred_fake - 0.0) ** 2)

                loss_D = 0.5 * (loss_real + loss_fake)

            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()

            # --- LOGGING ---
            loop.set_description(f"Epoch [{epoch+1}/{opt.epochs}]")
            loop.set_postfix(
                G_Loss=f"{loss_G.item():.2f}", 
                D_Loss=f"{loss_D.item():.2f}",
                Depth_L=f"{loss_dict['Depth']:.2f}"
            )

            # --- VISUALIZATION (Every 200 batches) ---
            if i % 200 == 0:
                with torch.no_grad():
                    depth_vis = (depth.repeat(1, 3, 1, 1) * 2) - 1 
                    img_sample = torch.cat((turbid, fake_clear, clear, depth_vis), -1)
                    save_path = f"{opt.sample_dir}/{opt.name}_epoch_{epoch}_batch_{i}.png"
                    save_image(img_sample, save_path, normalize=True)

        scheduler_G.step()
        scheduler_D.step()            

        # --- SAVE CHECKPOINTS (Every 5 Epochs) ---
        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), f"{opt.checkpoint_dir}/gen_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"{opt.checkpoint_dir}/disc_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved for epoch {epoch+1}")

if __name__ == "__main__":
    main()