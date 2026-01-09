# import argparse
# import ssl
# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from torchvision.utils import save_image
# from tqdm import tqdm
# import json
# import random

# # Bypass SSL errors
# ssl._create_default_https_context = ssl._create_unverified_context

# # --- IMPORTS ---
# from models.ClearVision import ClearVisionGenerator, PatchGANDiscriminator
# from utils.dataset import TurbidDataset
# from utils.losses import generator_loss, PerceptualLoss, SSIMLoss 

# def get_args():
#     parser = argparse.ArgumentParser()
    
#     # Identifiers & Paths
#     parser.add_argument("--name", type=str, default="ClearVision_ConfigA_Final", help="Experiment name")
#     parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Save location")
#     parser.add_argument("--sample_dir", type=str, default="samples/", help="Test image location")
#     parser.add_argument("--turbid_path", type=str, required=True)
#     parser.add_argument("--clear_path", type=str, required=True)
#     parser.add_argument("--depth_path", type=str, required=True)
    
#     # Training Config
#     parser.add_argument("--epochs", type=int, default=200)
#     parser.add_argument("--batch_size", type=int, default=4)
#     parser.add_argument("--use_amp", action='store_true')
#     parser.add_argument("--resume", action='store_true', help="Resume from latest checkpoint")

#     # Architecture
#     parser.add_argument("--ngf", type=int, default=32, help="Base channels (Model scales to 384 internal)")
#     parser.add_argument("--ndf", type=int, default=128)
    
#     # --- OPTIMIZATION CONFIG ---
#     parser.add_argument("--lr", type=float, default=1e-4, help="Standard stable LR")
    
#     # --- "CONFIG A" LOSS WEIGHTS (The Rescue Config) ---
#     # We follow Sea-Pix-GAN standards: Balanced L1 + GAN, SSIM as metric only.
#     parser.add_argument('--lambda_pixel', type=float, default=10.0)  # Reduced from 100 to prevent overfitting
#     parser.add_argument('--lambda_ssim', type=float, default=0.0)    # DISABLED for Optimization (Metric Only)
#     parser.add_argument('--lambda_adv', type=float, default=1.0)     # Increased from 0.05 to force learning
#     parser.add_argument('--lambda_perc', type=float, default=1.0)    # Standard VGG influence
#     parser.add_argument('--lambda_color', type=float, default=2.0)   # Brown water correction
#     parser.add_argument('--lambda_edge', type=float, default=0.5)    # Edge consistency
#     parser.add_argument('--lambda_depth', type=float, default=1.0)   # ENABLED: Physics guidance
    
#     return parser.parse_args()

# def main():
#     opt = get_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Directories
#     os.makedirs(os.path.join(opt.checkpoint_dir, opt.name), exist_ok=True)
#     os.makedirs(os.path.join(opt.sample_dir, opt.name), exist_ok=True)

#     # Save config
#     with open(os.path.join(opt.checkpoint_dir, opt.name, 'config.json'), 'w') as f:
#         json.dump(vars(opt), f, indent=4)

#     # Models
#     generator = ClearVisionGenerator(ngf=opt.ngf).to(device)
#     discriminator = PatchGANDiscriminator(ndf=opt.ndf).to(device)

#     def weights_init(m):
#         if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
#             torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    
#     # Optimizers
#     optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
#     optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
#     scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)
    
#     start_epoch = 0
#     if opt.resume:
#         path = os.path.join(opt.checkpoint_dir, opt.name, "latest.pth")
#         if os.path.exists(path):
#             print(f"Resuming from {path}")
#             ckpt = torch.load(path)
#             generator.load_state_dict(ckpt['G'])
#             discriminator.load_state_dict(ckpt['D'])
#             optimizer_G.load_state_dict(ckpt['opt_G'])
#             optimizer_D.load_state_dict(ckpt['opt_D'])
#             start_epoch = ckpt['epoch'] + 1
#         else:
#             print("No checkpoint found. Starting fresh.")
#             generator.apply(weights_init)
#             discriminator.apply(weights_init)
#     else:
#         generator.apply(weights_init)
#         discriminator.apply(weights_init)

#     # Losses
#     from torchvision.models import vgg19, VGG19_Weights
#     vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
#     perceptual_fn = PerceptualLoss(vgg).to(device)
    
#     # SSIM initialized for LOGGING ONLY (Loss weight is 0.0)
#     ssim_fn = SSIMLoss(window_size=11).to(device)

#     loss_weights = {
#         "adv": opt.lambda_adv, "pixel": opt.lambda_pixel, 
#         "color": opt.lambda_color, "edge": opt.lambda_edge, 
#         "perc": opt.lambda_perc, "ssim": opt.lambda_ssim, 
#         "depth": opt.lambda_depth
#     }

#     # Data
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
#     dataset = TurbidDataset(opt.turbid_path, opt.clear_path, opt.depth_path, transform=transform, augment=True)
#     dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)

#     print(f"--- Training Configured for OCEANS 2026 (Config A) ---")
#     print(f"    Strategies: L1 Anchor (10.0), Active GAN (1.0), SSIM Metric Only")
#     print(f"    Warmup Extended to 30 Epochs")

    

#     # TRAINING LOOP
#     for epoch in range(start_epoch, opt.epochs):
#         # Warmup Phase (Extended to 30 Epochs)
#         if epoch < 30:
#             current_lambda_adv = 0.0
#             phase = "WARMUP (L1 Only)"
#         else:
#             current_lambda_adv = opt.lambda_adv
#             phase = "GAN (Refinement)"
            
#         loss_weights['adv'] = current_lambda_adv
        
#         loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{opt.epochs} [{phase}]")
        
#         for i, (turbid, clear, depth) in enumerate(loop):
#             turbid, clear, depth = turbid.to(device), clear.to(device), depth.to(device)
            
#             # --- 1. TRAIN DISCRIMINATOR ---
#             # We only train D if we are out of the warmup phase
#             if epoch >= 30:
#                 optimizer_D.zero_grad()
#                 with torch.cuda.amp.autocast(enabled=opt.use_amp):
#                     fake_clear = generator(turbid)
#                     pred_real = discriminator(clear, turbid)
#                     pred_fake = discriminator(fake_clear.detach(), turbid)
                    
#                     # LSGAN Loss (MSE) - Strict 1.0/0.0 targets
#                     loss_real = torch.mean((pred_real - 1.0) ** 2)
#                     loss_fake = torch.mean((pred_fake - 0.0) ** 2)
#                     loss_D = 0.5 * (loss_real + loss_fake)

#                 scaler.scale(loss_D).backward()
#                 scaler.step(optimizer_D)
#                 d_loss_val = loss_D.item()
#             else:
#                 fake_clear = generator(turbid)
#                 d_loss_val = 0.0

#             # --- 2. TRAIN GENERATOR ---
#             optimizer_G.zero_grad()
#             with torch.cuda.amp.autocast(enabled=opt.use_amp):
#                 # FIX APPLIED HERE: Using Keyword Arguments
#                 loss_G, loss_dict = generator_loss(
#                     D=discriminator,
#                     real_img=clear,
#                     fake_img=fake_clear,
#                     input_img=turbid,
#                     depth=depth,
#                     max_depth=1.0,
#                     perceptual_fn=perceptual_fn,
#                     ssim_fn=ssim_fn,
#                     lambdas=loss_weights
#                 )

#             scaler.scale(loss_G).backward()
#             scaler.step(optimizer_G)
#             scaler.update()

#             loop.set_postfix(SSIM=f"{loss_dict.get('SSIM', 0):.3f}", D_loss=f"{d_loss_val:.3f}")

#             if i == 0:
#                 with torch.no_grad():
#                     save_image(torch.cat((turbid, fake_clear, clear), -1), 
#                              f"{opt.sample_dir}/{opt.name}/epoch_{epoch+1}.png", normalize=True)

#         # Save Checkpoint
#         save_dict = {
#             'epoch': epoch, 'G': generator.state_dict(), 'D': discriminator.state_dict(),
#             'opt_G': optimizer_G.state_dict(), 'opt_D': optimizer_D.state_dict()
#         }
#         torch.save(save_dict, os.path.join(opt.checkpoint_dir, opt.name, "latest.pth"))
        
#         if (epoch+1) % 5 == 0:
#             torch.save(generator.state_dict(), os.path.join(opt.checkpoint_dir, opt.name, f"epoch_{epoch+1}.pth"))

# if __name__ == "__main__":
#     main()

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
# Ensure you have your model and dataset files in the correct folders
from models.ClearVision import ClearVisionGenerator, PatchGANDiscriminator
from utils.dataset import TurbidDataset
from utils.losses import generator_loss, PerceptualLoss, SSIMLoss 

def get_args():
    parser = argparse.ArgumentParser()
    
    # Identifiers & Paths
    parser.add_argument("--name", type=str, default="ClearVision_Standard_Golden", help="Experiment name")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Save location")
    parser.add_argument("--sample_dir", type=str, default="samples/", help="Test image location")
    parser.add_argument("--turbid_path", type=str, required=True)
    parser.add_argument("--clear_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=True)
    
    # Training Config
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--use_amp", action='store_true')
    
    # --- RESUME vs FINETUNE ---
    parser.add_argument("--resume", action='store_true', help="Continue training from latest.pth (loads optimizer)")
    parser.add_argument("--finetune", type=str, default=None, help="Path to checkpoint to initialize weights from (resets optimizer & epoch)")

    # Architecture
    parser.add_argument("--ngf", type=int, default=48, help="Generator capacity")
    parser.add_argument("--ndf", type=int, default=64)
    
    # --- OPTIMIZATION CONFIG ---
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning Rate")
    
    # --- LOSS WEIGHTS ---
    parser.add_argument('--lambda_pixel', type=float, default=50.0) 
    parser.add_argument('--lambda_ssim', type=float, default=0.0)    
    parser.add_argument('--lambda_adv', type=float, default=0.5)    
    parser.add_argument('--lambda_perc', type=float, default=1.0)    
    parser.add_argument('--lambda_color', type=float, default=1.0)   
    parser.add_argument('--lambda_edge', type=float, default=0.5)    
    parser.add_argument('--lambda_depth', type=float, default=0.5)   
    
    return parser.parse_args()

def main():
    opt = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Directories
    os.makedirs(os.path.join(opt.checkpoint_dir, opt.name), exist_ok=True)
    os.makedirs(os.path.join(opt.sample_dir, opt.name), exist_ok=True)

    # Save config
    with open(os.path.join(opt.checkpoint_dir, opt.name, 'config.json'), 'w') as f:
        json.dump(vars(opt), f, indent=4)

    # Models
    generator = ClearVisionGenerator(ngf=opt.ngf).to(device)
    discriminator = PatchGANDiscriminator(ndf=opt.ndf).to(device)

    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)
    
    start_epoch = 0
    
    # --- INITIALIZATION LOGIC ---
    if opt.resume:
        # RESUME: Loads everything to continue exactly where you left off
        path = os.path.join(opt.checkpoint_dir, opt.name, "latest.pth")
        if os.path.exists(path):
            print(f"Resuming from {path}")
            ckpt = torch.load(path)
            generator.load_state_dict(ckpt['G'])
            discriminator.load_state_dict(ckpt['D'])
            optimizer_G.load_state_dict(ckpt['opt_G'])
            optimizer_D.load_state_dict(ckpt['opt_D'])
            start_epoch = ckpt['epoch'] + 1
        else:
            print(f"Warning: No checkpoint found at {path}. Starting fresh.")
            generator.apply(weights_init)
            discriminator.apply(weights_init)
            
    elif opt.finetune:
        # FINETUNE: Loads weights only, resets optimizer and epoch
        print(f"Finetuning from checkpoint: {opt.finetune}")
        if os.path.exists(opt.finetune):
            ckpt = torch.load(opt.finetune)
            
            # Robust loading (handles 'G' key or raw state_dict)
            if 'G' in ckpt:
                generator.load_state_dict(ckpt['G'])
            else:
                generator.load_state_dict(ckpt)
            
            # We usually reset Discriminator for a new phase to avoid overpowering G immediately
            discriminator.apply(weights_init)
            print("Generator weights loaded. Discriminator and Optimizers reset for Finetuning.")
        else:
            print(f"Error: Finetune checkpoint {opt.finetune} not found.")
            return
    else:
        # FRESH START
        generator.apply(weights_init)
        discriminator.apply(weights_init)

    # Losses
    from torchvision.models import vgg19, VGG19_Weights
    vgg = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
    perceptual_fn = PerceptualLoss(vgg).to(device)
    ssim_fn = SSIMLoss(window_size=11).to(device)

    loss_weights = {
        "adv": opt.lambda_adv, "pixel": opt.lambda_pixel, 
        "color": opt.lambda_color, "edge": opt.lambda_edge, 
        "perc": opt.lambda_perc, "ssim": opt.lambda_ssim, 
        "depth": opt.lambda_depth
    }

    # Data
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = TurbidDataset(opt.turbid_path, opt.clear_path, opt.depth_path, transform=transform, augment=True)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    print(f"--- Training Configured ---")
    print(f"    Mode: {'Finetune' if opt.finetune else 'Resume' if opt.resume else 'Fresh'}")
    print(f"    Learning Rate: {opt.lr}")

    # TRAINING LOOP
    for epoch in range(start_epoch, opt.epochs):
        
        # --- WARMUP LOGIC ---
        # Skip warmup if we are Finetuning or Resuming (unless resuming into a warmup)
        # We only want L1-only warmup for a fresh start.
        if epoch < 30 and not opt.finetune and not opt.resume:
            current_lambda_adv = 0.0
            phase = "WARMUP (L1 Only)"
        else:
            current_lambda_adv = opt.lambda_adv
            phase = "TRAIN"
            
        loss_weights['adv'] = current_lambda_adv
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{opt.epochs} [{phase}]")
        
        for i, (turbid, clear, depth) in enumerate(loop):
            turbid, clear, depth = turbid.to(device), clear.to(device), depth.to(device)
            
            # --- 1. TRAIN DISCRIMINATOR ---
            # Only train D if adv loss is active
            if current_lambda_adv > 0:
                optimizer_D.zero_grad()
                with torch.cuda.amp.autocast(enabled=opt.use_amp):
                    fake_clear = generator(turbid)
                    pred_real = discriminator(clear, turbid)
                    pred_fake = discriminator(fake_clear.detach(), turbid)
                    
                    # LSGAN Loss
                    loss_real = torch.mean((pred_real - 1.0) ** 2)
                    loss_fake = torch.mean((pred_fake - 0.0) ** 2)
                    loss_D = 0.5 * (loss_real + loss_fake)

                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
                d_loss_val = loss_D.item()
            else:
                fake_clear = generator(turbid)
                d_loss_val = 0.0

            # --- 2. TRAIN GENERATOR ---
            optimizer_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                loss_G, loss_dict = generator_loss(
                    D=discriminator,
                    real_img=clear,
                    fake_img=fake_clear,
                    input_img=turbid,
                    depth=depth,
                    max_depth=1.0,
                    perceptual_fn=perceptual_fn,
                    ssim_fn=ssim_fn,
                    lambdas=loss_weights
                )

            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scaler.update()

            loop.set_postfix(SSIM=f"{loss_dict.get('SSIM', 0):.3f}", D_loss=f"{d_loss_val:.3f}")

            if i == 0:
                with torch.no_grad():
                    save_image(torch.cat((turbid, fake_clear, clear), -1), 
                             f"{opt.sample_dir}/{opt.name}/epoch_{epoch+1}.png", normalize=True)

        # Save Checkpoint
        save_dict = {
            'epoch': epoch, 'G': generator.state_dict(), 'D': discriminator.state_dict(),
            'opt_G': optimizer_G.state_dict(), 'opt_D': optimizer_D.state_dict()
        }
        torch.save(save_dict, os.path.join(opt.checkpoint_dir, opt.name, "latest.pth"))
        
        if (epoch+1) % 10 == 0:
            torch.save(save_dict, os.path.join(opt.checkpoint_dir, opt.name, f"epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    main()