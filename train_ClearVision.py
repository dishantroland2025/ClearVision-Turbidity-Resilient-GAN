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

# Bypass SSL errors
ssl._create_default_https_context = ssl._create_unverified_context

# --- IMPORTS ---
from models.ClearVision import ClearVisionGenerator, PatchGANDiscriminator
from utils.dataset import TurbidDataset
from utils.losses import generator_loss, PerceptualLoss, MSSSIMLoss

def get_args():
    parser = argparse.ArgumentParser()
    
    # Identifiers & Paths
    parser.add_argument("--name", type=str, default="ClearVision_Phase2", help="Experiment name")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Save location")
    parser.add_argument("--sample_dir", type=str, default="samples/", help="Test image location")
    parser.add_argument("--turbid_path", type=str, required=True)
    parser.add_argument("--clear_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=True)
    
    # Training Config
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--use_amp", action='store_true')
    parser.add_argument("--resume", action='store_true', help="Resume from latest checkpoint")

    # Architecture
    parser.add_argument("--ngf", type=int, default=32)
    parser.add_argument("--ndf", type=int, default=128)
    
    # Loss Weights
    parser.add_argument('--lambda_adv', type=float, default=0.1)
    parser.add_argument('--lambda_pixel', type=float, default=10.0) # Updated to 10.0 (Anchor)
    parser.add_argument('--lambda_color', type=float, default=0.5)
    parser.add_argument('--lambda_edge', type=float, default=0.2)
    parser.add_argument('--lambda_perc', type=float, default=0.2)
    parser.add_argument('--lambda_ssim', type=float, default=0.8) # Updated to Silver Bullet
    parser.add_argument('--lambda_depth', type=float, default=0.5)

    return parser.parse_args()

def main():
    opt = get_args()
    
    # 0. Device Setup & Sanity Check
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("⚠️ WARNING: CUDA not available. Training on CPU will be extremely slow.")
        device = torch.device("cpu")

    # 1. Setup Directories
    chk_dir = os.path.join(opt.checkpoint_dir, opt.name)
    vis_dir = os.path.join(opt.sample_dir, opt.name)
    os.makedirs(chk_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Save config
    with open(os.path.join(chk_dir, 'config.json'), 'w') as f:
        json.dump(vars(opt), f, indent=4)

    print(f"--- Experiment: {opt.name} ---")
    print(f"    Device: {device}")

    # 2. Models
    generator = ClearVisionGenerator(ngf=opt.ngf).to(device)
    discriminator = PatchGANDiscriminator(ndf=opt.ndf).to(device)

    # Init Weights
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    # 3. Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    # Updated GradScaler for compatibility
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)
    start_epoch = 0

    # --- RESUME LOGIC (CRITICAL FOR COLAB) ---
    if opt.resume:
        latest_path = os.path.join(chk_dir, "latest.pth")
        if os.path.exists(latest_path):
            print(f" Found checkpoint! Resuming from {latest_path}")
            ckpt = torch.load(latest_path, map_location=device)
            generator.load_state_dict(ckpt['G'])
            discriminator.load_state_dict(ckpt['D'])
            optimizer_G.load_state_dict(ckpt['opt_G'])
            optimizer_D.load_state_dict(ckpt['opt_D'])
            start_epoch = ckpt['epoch'] + 1
        else:
            print(" No checkpoint found. Starting fresh.")
            generator.apply(weights_init)
            discriminator.apply(weights_init)
    else:
        generator.apply(weights_init)
        discriminator.apply(weights_init)

    # Scheduler (Re-create after load to sync with epoch)
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 - (opt.epochs // 2)) / float(opt.epochs // 2 + 1)
        return lr_l
    scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule, last_epoch=start_epoch-1)
    scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule, last_epoch=start_epoch-1)

    # 4. Losses
    from torchvision.models import vgg19, VGG19_Weights
    # Use the new safe syntax for weights
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    perceptual_fn = PerceptualLoss(vgg).to(device)
    ssim_fn = MSSSIMLoss().to(device)

    loss_weights = {
        "adv": opt.lambda_adv, "pixel": opt.lambda_pixel, 
        "color": opt.lambda_color, "edge": opt.lambda_edge, 
        "perc": opt.lambda_perc, "ssim": opt.lambda_ssim, 
        "depth": opt.lambda_depth
    }

    # 5. Data
    base_transforms = transforms.Compose([
        transforms.Resize((256, 256)), # Ensure size matches model
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = TurbidDataset(opt.turbid_path, opt.clear_path, opt.depth_path, transform=base_transforms, augment=True)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"    Images: {len(dataset)}")

    # 6. Loop
    for epoch in range(start_epoch, opt.epochs):
        # --- PHASED TRAINING LOGIC (NEW) ---
        # 1. Warmup Phase (Epoch 0-20): Turn off GAN loss
        if epoch < 20:
            current_lambda_adv = 0.0
            phase_status = "WARMUP (No GAN)"
        # 2. GAN Phase (Epoch 20+): Enable GAN loss
        else:
            current_lambda_adv = opt.lambda_adv  # Uses the value 0.5 or 1.0 you set
            phase_status = "GAN ENABLED"
            
        # Update the dictionary passed to the loss function
        loss_weights['adv'] = current_lambda_adv
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{opt.epochs} [{phase_status}]")
        
        for i, (turbid, clear, depth) in enumerate(loop):
            turbid, clear, depth = turbid.to(device), clear.to(device), depth.to(device)

            # --- G ---
            optimizer_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                fake_clear = generator(turbid)
                
                # --- FIXED: Use Keyword Arguments to prevent TypeError ---
                loss_G, loss_dict = generator_loss(
                    D=discriminator, 
                    real_img=clear, 
                    fake_img=fake_clear, 
                    input_img=turbid, 
                    depth=depth, 
                    perceptual_fn=perceptual_fn, 
                    ssim_fn=ssim_fn, 
                    lambdas=loss_weights
                )

            scaler.scale(loss_G).backward()
            scaler.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
            scaler.step(optimizer_G)
            scaler.update()

            # --- D ---
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

            # --- Log ---
            loop.set_postfix(SSIM=f"{loss_dict['SSIM']:.3f}", G=f"{loss_G.item():.2f}", D=f"{loss_D.item():.2f}")

            # --- Vis (Fix: Save first batch every time) ---
            if i == 0:
                with torch.no_grad():
                    depth_vis = (depth.repeat(1, 3, 1, 1) * 2) - 1 
                    img_grid = torch.cat((turbid, fake_clear, clear, depth_vis), -1)
                    save_image(img_grid, f"{vis_dir}/epoch_{epoch+1}.png", normalize=True)

        scheduler_G.step()
        scheduler_D.step()            

        # --- Save ---
        # 1. Always save latest (for resume)
        save_dict = {
            'epoch': epoch,
            'G': generator.state_dict(),
            'D': discriminator.state_dict(),
            'opt_G': optimizer_G.state_dict(),
            'opt_D': optimizer_D.state_dict()
        }
        torch.save(save_dict, f"{chk_dir}/latest.pth")

        # 2. Save history checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), f"{chk_dir}/gen_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()