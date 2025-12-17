import argparse
import ssl
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

ssl._create_default_https_context = ssl._create_unverified_context

# --- IMPORTS ---
# Ensure your folder structure matches these imports!
from models.ClearVision import ClearVisionGenerator, Discriminator
from utils.dataset import TurbidDataset
from utils.Loss_Funtion import generator_loss, PerceptualLoss

# --- CONFIGURATION ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size (Keep small for Jetson/Laptop)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=256, help="Input image resolution")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/", help="Where to save .pth files")
    parser.add_argument("--sample_dir", type=str, default="samples/", help="Where to save test images")
    
    # DATA PATHS (Update these defaults to your actual paths)
    parser.add_argument("--turbid_path", type=str, required=True, help="Path to Turbid images")
    parser.add_argument("--clear_path", type=str, required=True, help="Path to Clear (GT) images")
    parser.add_argument("--depth_path", type=str, required=True, help="Path to Depth maps")
    
    return parser.parse_args()

def main():
    opt = get_args()
    
    # Create directories if they don't exist
    os.makedirs(opt.checkpoint_dir, exist_ok=True)
    os.makedirs(opt.sample_dir, exist_ok=True)

    # AUTO-SAVE CONFIGURATION
    import json
    config_path = os.path.join(opt.checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        # We use vars(opt) because your variable is named 'opt'
        json.dump(vars(opt), f, indent=4)
    print(f"Configuration saved to: {config_path}")

    # --- DEVICE SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting ClearVision Training on: {device}")

    # --- INITIALIZE MODELS ---
    generator = ClearVisionGenerator().to(device)
    discriminator = Discriminator().to(device)

    # Initialize weights
    def weights_init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    
    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # --- OPTIMIZERS ---
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    # --- PERCEPTUAL LOSS SETUP ---
    # We initialize VGG once here to save memory
    print("Loading VGG19 for Perceptual Loss...")
    from torchvision.models import vgg19
    vgg = vgg19(pretrained=True).features.to(device)
    perceptual_fn = PerceptualLoss(vgg).to(device)

    # --- LOSS WEIGHTS (Hyperparameters from your Proposal) ---
    lambdas = {
        "adv":   0.1,  
        "pixel": 10.0, 
        "color": 5.0,  
        "edge":  2.0,  
        "perc":  1.0,  
        "depth": 5.0   
    }

    # --- DATA LOADING ---
    transforms_ = transforms.Compose([
        transforms.Resize((opt.img_size, opt.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Normalize to [-1, 1]
    ])

    dataset = TurbidDataset(opt.turbid_path, opt.clear_path, opt.depth_path, transform=transforms_)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=2)

    print(f"Loaded {len(dataset)} training triplets.")

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

            # Generate Fake Image
            fake_clear = generator(turbid)

            # Calculate Combined Loss
            # This calls the function in utils/losses.py
            loss_G, loss_dict = generator_loss(
                discriminator, clear, fake_clear, turbid, 
                depth=depth, 
                perceptual_fn=perceptual_fn, 
                lambdas=lambdas
            )

            loss_G.backward()
            optimizer_G.step()

            # -------------------------
            #  2. Train Discriminator
            # -------------------------
            optimizer_D.zero_grad()

            # Real Loss (Discriminator should predict Real=1)
            # Input is (Condition + Real)
            pred_real = discriminator(clear, turbid)
            loss_real = torch.mean((pred_real - 1.0) ** 2) # LSGAN

            # Fake Loss (Discriminator should predict Fake=0)
            # Input is (Condition + Fake). Detach to stop G from updating
            pred_fake = discriminator(fake_clear.detach(), turbid)
            loss_fake = torch.mean((pred_fake - 0.0) ** 2) # LSGAN

            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()

            # --- LOGGING ---
            loop.set_description(f"Epoch [{epoch+1}/{opt.epochs}]")
            loop.set_postfix(
                G_Loss=f"{loss_G.item():.2f}", 
                D_Loss=f"{loss_D.item():.2f}",
                Depth_L=f"{loss_dict['Depth']:.2f}" # Monitor Physics Loss
            )

            # --- VISUALIZATION (Every 200 batches) ---
            if i % 200 == 0:
                with torch.no_grad():
                    # Format depth for display (1 channel -> 3 channels, scale -1 to 1)
                    depth_vis = (depth.repeat(1, 3, 1, 1) * 2) - 1 
                    
                    # Stack images: Turbid | Generated | Ground Truth | Depth
                    img_sample = torch.cat((turbid, fake_clear, clear, depth_vis), -1)
                    save_image(img_sample, f"{opt.sample_dir}/epoch_{epoch}_batch_{i}.png", normalize=True)

        # --- SAVE CHECKPOINTS (Every 5 Epochs) ---
        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), f"{opt.checkpoint_dir}/gen_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"{opt.checkpoint_dir}/disc_epoch_{epoch+1}.pth")
            print(f"Checkpoint saved for epoch {epoch+1}")

if __name__ == "__main__":
    main()