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

# Bypass SSL certificate errors (common on some university/corporate networks)
ssl._create_default_https_context = ssl._create_unverified_context

# --- IMPORTS ---
# We import PatchGANDiscriminator from the model file now
from models.ClearVision import ClearVisionGenerator, PatchGANDiscriminator
from utils.dataset import TurbidDataset
from utils.losses import generator_loss, PerceptualLoss, MSSSIMLoss

# --- CONFIGURATION ---
def get_args():
    parser = argparse.ArgumentParser()
    
    # Experiment Identifiers
    parser.add_argument("--name", type=str, default="ClearVision_Phase2", help="Experiment name")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Save location")
    parser.add_argument("--sample_dir", type=str, default="samples/", help="Test image location")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=200, help="Total epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (8 fits on T4)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--use_amp", action='store_true', help="Enable Mixed Precision (fp16)")
    
    # Architecture Params
    parser.add_argument("--ngf", type=int, default=32, help="Base generator filters (Speed optimized)")
    # IMPORTANT: Default to 128 filters for the 11M param Robust Discriminator
    parser.add_argument("--ndf", type=int, default=128, help="Discriminator capacity (128 for Phase 2)")
    
    # Loss Weights (The 'Teacher' Configuration)
    parser.add_argument('--lambda_adv', type=float, default=0.1, help='Adversarial realism weight')
    parser.add_argument('--lambda_pixel', type=float, default=1.0, help='L1 pixel accuracy weight')
    parser.add_argument('--lambda_color', type=float, default=0.5, help='LAB color consistency weight')
    parser.add_argument('--lambda_edge', type=float, default=0.2, help='Sobel edge preservation weight')
    parser.add_argument('--lambda_perc', type=float, default=0.2, help='VGG perceptual weight')
    parser.add_argument('--lambda_ssim', type=float, default=0.4, help='MS-SSIM structural weight')
    parser.add_argument('--lambda_depth', type=float, default=0.5, help='Physics-based depth weight')

    # Data Paths
    parser.add_argument("--turbid_path", type=str, required=True)
    parser.add_argument("--clear_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=True)
    
    return parser.parse_args()

def main():
    opt = get_args()
    
    # 1. Directory Setup
    chk_dir = os.path.join(opt.checkpoint_dir, opt.name)
    vis_dir = os.path.join(opt.sample_dir, opt.name)
    os.makedirs(chk_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(chk_dir, 'config.json'), 'w') as f:
        json.dump(vars(opt), f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Launching Experiment: {opt.name} ---")
    print(f"    Device: {device}")
    print(f"    Mixed Precision: {opt.use_amp}")
    print(f"    Discriminator Filters: {opt.ndf}")

    # 2. Model Initialization
    # Generator: The Phase 2 ClearVision (25M params)
    generator = ClearVisionGenerator(ngf=opt.ngf).to(device)
    # Discriminator: River-Optimized PatchGAN
    discriminator = PatchGANDiscriminator(ndf=opt.ndf).to(device)

    # Weight Initialization (Important for GAN stability)
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # 3. Optimizers & Schedulers
    # Beta1=0.5 is critical for GANs (prevents momentum oscillation)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # Linear Decay Policy: Keep LR constant for first half, decay linearly in second half
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - (opt.epochs // 2)) / float(opt.epochs // 2 + 1)
        return lr_l

    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    # AMP Scaler (fp16)
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)

    # 4. Losses
    print("    Loading Loss Functions...")
    from torchvision.models import vgg19
    vgg = vgg19(pretrained=True).features.to(device).eval() # Frozen VGG
    perceptual_fn = PerceptualLoss(vgg).to(device)
    ssim_fn = MSSSIMLoss().to(device)

    loss_weights = {
        "adv": opt.lambda_adv, "pixel": opt.lambda_pixel, 
        "color": opt.lambda_color, "edge": opt.lambda_edge, 
        "perc": opt.lambda_perc, "ssim": opt.lambda_ssim, 
        "depth": opt.lambda_depth
    }

    # 5. Dataset
    # NOTE: We ONLY pass ToTensor/Normalize here.
    # Geometric augmentations (Flip/Crop/Rotate) are handled INSIDE TurbidDataset
    # to guarantee RGB-Depth synchronization. If we added them here, depth would break.
    base_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = TurbidDataset(
        opt.turbid_path, opt.clear_path, opt.depth_path, 
        transform=base_transforms, 
        augment=True # Enable internal augmentation
    )
    
    dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, 
        shuffle=True, num_workers=4, pin_memory=True
    )
    print(f"    Dataset loaded: {len(dataset)} triplets.")

    # 6. Training Loop
    print("--- Starting Training ---")
    for epoch in range(opt.epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{opt.epochs}")
        
        for i, (turbid, clear, depth) in enumerate(loop):
            turbid, clear, depth = turbid.to(device), clear.to(device), depth.to(device)

            # --- TRAIN GENERATOR ---
            optimizer_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                fake_clear = generator(turbid)
                
                # Calculate composite loss
                loss_G, loss_dict = generator_loss(
                    discriminator, clear, fake_clear, turbid, 
                    depth=depth, 
                    perceptual_fn=perceptual_fn, 
                    ssim_fn=ssim_fn, # Added SSIM
                    lambdas=loss_weights
                )

            scaler.scale(loss_G).backward()
            
            # Gradient Clipping: Vital for 25M param model stability
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=10.0)
            
            scaler.step(optimizer_G)
            scaler.update()

            # --- TRAIN DISCRIMINATOR ---
            optimizer_D.zero_grad()
            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                # Real Loss
                pred_real = discriminator(clear, turbid)
                loss_real = torch.mean((pred_real - 1.0) ** 2) # LSGAN (Least Squares)
                
                # Fake Loss (Detach to stop G gradients)
                pred_fake = discriminator(fake_clear.detach(), turbid)
                loss_fake = torch.mean((pred_fake - 0.0) ** 2)
                
                loss_D = 0.5 * (loss_real + loss_fake)

            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scaler.update()

            # --- LOGGING ---
            loop.set_postfix(
                SSIM=f"{loss_dict['SSIM']:.3f}", 
                G_Loss=f"{loss_G.item():.2f}", 
                D_Loss=f"{loss_D.item():.2f}"
            )

            # --- VISUALIZATION (Every 500 steps) ---
            if i % 500 == 0:
                with torch.no_grad():
                    # Visualize Depth: Expand 1ch -> 3ch and map to [-1, 1]
                    depth_vis = (depth.repeat(1, 3, 1, 1) * 2) - 1 
                    img_grid = torch.cat((turbid, fake_clear, clear, depth_vis), -1)
                    save_image(img_grid, f"{vis_dir}/epoch_{epoch}_batch_{i}.png", normalize=True)

        # Update Learning Rates
        scheduler_G.step()
        scheduler_D.step()            

        # --- CHECKPOINTING ---
        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), f"{chk_dir}/gen_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"{chk_dir}/disc_epoch_{epoch+1}.pth")
            print(f"    Checkpoint saved: epoch {epoch+1}")

if __name__ == "__main__":
    main()