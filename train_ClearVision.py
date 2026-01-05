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
import math
import copy

# Bypass SSL errors
ssl._create_default_https_context = ssl._create_unverified_context

# --- IMPORTS ---
from models.ClearVision import ClearVisionGenerator, PatchGANDiscriminator
from utils.dataset import TurbidDataset
from utils.losses import generator_loss, PerceptualLoss, MSSSIMLoss

# --- EMA HELPER CLASS (The New Fix) ---
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # new_average = decay * old_average + (1 - decay) * current_param
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA weights to the model (for evaluation/saving)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights (for continuing training)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]

def get_args():
    parser = argparse.ArgumentParser()
    
    # Identifiers & Paths
    parser.add_argument("--name", type=str, default="ClearVision_Rescue_Edge", help="Experiment name")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Save location")
    parser.add_argument("--sample_dir", type=str, default="samples/", help="Test image location")
    parser.add_argument("--turbid_path", type=str, required=True)
    parser.add_argument("--clear_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=True)
    
    # Training Config
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_amp", action='store_true')
    parser.add_argument("--resume", action='store_true', help="Resume from latest checkpoint")

    # Architecture (LOCKED at 32 for Jetson Speed)
    parser.add_argument("--ngf", type=int, default=32, help="Kept at 32 for >30 FPS on Edge")
    parser.add_argument("--ndf", type=int, default=128)
    
    # --- OPTIMIZATION (THE RESCUE FIX) ---
    parser.add_argument("--lr_g", type=float, default=3e-4)
    parser.add_argument("--lr_d", type=float, default=3e-5)
    parser.add_argument("--ema_decay", type=float, default=0.999, help="Decay rate for Exponential Moving Average")
    
    # Loss Weights
    parser.add_argument('--lambda_adv', type=float, default=0.1)
    parser.add_argument('--lambda_pixel', type=float, default=10.0)
    parser.add_argument('--lambda_color', type=float, default=2.0)
    parser.add_argument('--lambda_edge', type=float, default=0.5)
    parser.add_argument('--lambda_perc', type=float, default=5.0)
    parser.add_argument('--lambda_ssim', type=float, default=5.0)
    parser.add_argument('--lambda_depth', type=float, default=1.0)
    
    # Tuning Integration
    parser.add_argument("--tuning_file", type=str, default="best_hyperparameters.json", help="Path to tuned params")

    return parser.parse_args()

# --- INSTANCE NOISE FUNCTION ---
def get_instance_noise(epoch, total_epochs, std=0.05):
    current_std = std * (1 - epoch / total_epochs)
    return max(0, current_std)

def main():
    opt = get_args()
    
    # 0. Device Setup
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

    # --- AUTO-LOAD TUNED HYPERPARAMETERS ---
    if os.path.exists(opt.tuning_file):
        print(f"🔥 FOUND TUNED PARAMETERS: {opt.tuning_file}")
        with open(opt.tuning_file, 'r') as f:
            tuned = json.load(f)
            if 'lambda_pixel' in tuned: opt.lambda_pixel = tuned['lambda_pixel']
            if 'lambda_ssim' in tuned: opt.lambda_ssim = tuned['lambda_ssim']
            if 'lambda_adv' in tuned: opt.lambda_adv = tuned['lambda_adv']
            if 'lambda_color' in tuned: opt.lambda_color = tuned['lambda_color']
            if 'lr_g' in tuned: opt.lr_g = tuned['lr_g']
            if 'lr_d_ratio' in tuned: opt.lr_d = opt.lr_g * tuned['lr_d_ratio']

    # Save final config
    with open(os.path.join(chk_dir, 'config.json'), 'w') as f:
        json.dump(vars(opt), f, indent=4)

    print(f"--- Experiment: {opt.name} ---")
    print(f"    LR_G: {opt.lr_g} | LR_D: {opt.lr_d} | EMA Decay: {opt.ema_decay}")

    # 2. Models
    generator = ClearVisionGenerator(ngf=opt.ngf).to(device)
    discriminator = PatchGANDiscriminator(ndf=opt.ndf).to(device)

    # Init Weights
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    # 3. Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(0.5, 0.999))
    
    scaler = torch.cuda.amp.GradScaler(enabled=opt.use_amp)
    start_epoch = 0

    # --- EMA INITIALIZATION ---
    ema_generator = EMA(generator, decay=opt.ema_decay)

    # --- RESUME LOGIC ---
    if opt.resume:
        latest_path = os.path.join(chk_dir, "latest.pth")
        if os.path.exists(latest_path):
            print(f" Found checkpoint! Resuming from {latest_path}")
            ckpt = torch.load(latest_path, map_location=device)
            generator.load_state_dict(ckpt['G'])
            discriminator.load_state_dict(ckpt['D'])
            optimizer_G.load_state_dict(ckpt['opt_G'])
            optimizer_D.load_state_dict(ckpt['opt_D'])
            # Restore EMA state if available
            if 'G_ema' in ckpt:
                ema_generator.shadow = ckpt['G_ema']
            else:
                print("Warning: No EMA state found in checkpoint. Starting EMA from current weights.")
                ema_generator = EMA(generator, decay=opt.ema_decay)
            start_epoch = ckpt['epoch'] + 1
        else:
            print(" No checkpoint found. Starting fresh.")
            generator.apply(weights_init)
            discriminator.apply(weights_init)
            # Re-init EMA with fresh weights
            ema_generator = EMA(generator, decay=opt.ema_decay)
    else:
        generator.apply(weights_init)
        discriminator.apply(weights_init)
        # Re-init EMA
        ema_generator = EMA(generator, decay=opt.ema_decay)

    # --- SCHEDULER: COSINE ANNEALING ---
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=opt.epochs, eta_min=opt.lr_g * 0.01)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=opt.epochs, eta_min=opt.lr_d * 0.01)

    # 4. Losses
    from torchvision.models import vgg19, VGG19_Weights
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
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = TurbidDataset(opt.turbid_path, opt.clear_path, opt.depth_path, transform=base_transforms, augment=True)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"    Images: {len(dataset)}")

    # 6. Loop
    for epoch in range(start_epoch, opt.epochs):
        # Warmup for 30 epochs
        if epoch < 30:
            current_lambda_adv = 0.0
            phase_status = "WARMUP"
        else:
            current_lambda_adv = opt.lambda_adv
            phase_status = "GAN"
            
        loss_weights['adv'] = current_lambda_adv
        noise_std = get_instance_noise(epoch, opt.epochs)
        
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{opt.epochs} [{phase_status}] Noise:{noise_std:.3f}")
        
        for i, (turbid, clear, depth) in enumerate(loop):
            turbid, clear, depth = turbid.to(device), clear.to(device), depth.to(device)

            # --- G ---
            optimizer_G.zero_grad()
            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                fake_clear = generator(turbid)
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
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            scaler.step(optimizer_G)
            
            # --- UPDATE EMA ---
            ema_generator.update()
            
            scaler.update()

            # --- D ---
            optimizer_D.zero_grad()
            with torch.cuda.amp.autocast(enabled=opt.use_amp):
                noise = torch.randn_like(turbid) * noise_std
                pred_real = discriminator(clear + noise, turbid + noise)
                loss_real = torch.mean((pred_real - 0.9) ** 2)
                pred_fake = discriminator(fake_clear.detach() + noise, turbid + noise)
                loss_fake = torch.mean((pred_fake - 0.1) ** 2)
                loss_D = 0.5 * (loss_real + loss_fake)

            scaler.scale(loss_D).backward()
            scaler.unscale_(optimizer_D)
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 0.5)
            scaler.step(optimizer_D)
            scaler.update()

            loop.set_postfix(SSIM=f"{loss_dict['SSIM']:.3f}", G=f"{loss_G.item():.2f}", D=f"{loss_D.item():.2f}")

            # --- Vis (USING EMA WEIGHTS) ---
            if i == 0:
                with torch.no_grad():
                    # Temporarily swap weights to EMA for visualization
                    ema_generator.apply_shadow()
                    
                    # Generate with EMA weights
                    fake_ema = generator(turbid)
                    depth_vis = (depth.repeat(1, 3, 1, 1) * 2) - 1 
                    img_grid = torch.cat((turbid, fake_ema, clear, depth_vis), -1)
                    save_image(img_grid, f"{vis_dir}/epoch_{epoch+1}.png", normalize=True)
                    
                    # Restore original training weights
                    ema_generator.restore()

        scheduler_G.step()
        scheduler_D.step()            

        # --- Save ---
        save_dict = {
            'epoch': epoch,
            'G': generator.state_dict(),
            'G_ema': ema_generator.shadow, # Save stable EMA weights
            'D': discriminator.state_dict(),
            'opt_G': optimizer_G.state_dict(),
            'opt_D': optimizer_D.state_dict()
        }
        torch.save(save_dict, f"{chk_dir}/latest.pth")

        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), f"{chk_dir}/gen_epoch_{epoch+1}.pth")
            
            # OPTIONAL: Save a specific EMA checkpoint occasionally
            # We create a temp dict to save purely the EMA model state
            ema_generator.apply_shadow()
            torch.save(generator.state_dict(), f"{chk_dir}/gen_ema_epoch_{epoch+1}.pth")
            ema_generator.restore()

if __name__ == "__main__":
    main()