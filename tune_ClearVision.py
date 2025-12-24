import optuna
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
import json
import gc 
import traceback

# --- IMPORTS ---
from models.ClearVision import ClearVisionGenerator, Discriminator
from utils.dataset import TurbidDataset
from utils.losses import generator_loss, PerceptualLoss
from utils.metrics import calculate_psnr

# --- CONFIG ---
SEARCH_EPOCHS = 20  
TOTAL_TRIALS = 70   
# ----------------

def objective(trial):
    # 1. Search Space
    lr_g = trial.suggest_float("lr_g", 5e-5, 5e-4, log=True)
    lr_d = trial.suggest_float("lr_d", 5e-5, 5e-4, log=True) 
    batch_size = trial.suggest_categorical("batch_size", [4, 8])
    ngf = trial.suggest_categorical("ngf", [64, 80])
    
    # Loss Weights
    lambda_pixel = trial.suggest_float("lambda_pixel", 5.0, 15.0)
    lambda_color = trial.suggest_float("lambda_color", 1.0, 10.0)
    lambda_perc = trial.suggest_float("lambda_perc", 0.1, 5.0)
    lambda_edge = trial.suggest_float("lambda_edge", 0.1, 5.0)
    lambda_depth = trial.suggest_float("lambda_depth", 0.1, 10.0)

    # 2. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # PATHS (Verify these!)
    TURBID_PATH = "/Users/dishantdas/River-UIEB/train/raw"
    CLEAR_PATH = "/Users/dishantdas/River-UIEB/train/GT"
    DEPTH_PATH = "/Users/dishantdas/River-UIEB/train/depths"
    
    try:
        # Init Models
        generator = ClearVisionGenerator(ngf=ngf).to(device)
        discriminator = Discriminator().to(device) 
        
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
        
        scaler = torch.cuda.amp.GradScaler()
        
        from torchvision.models import vgg19
        vgg = vgg19(weights='DEFAULT').features.to(device).eval()
        perceptual_fn = PerceptualLoss(vgg).to(device)
        
        lambdas = {
            "adv": 0.1, 
            "pixel": lambda_pixel, "color": lambda_color, "edge": lambda_edge, 
            "perc": lambda_perc, "depth": lambda_depth 
        }

        # 3. DATA LOADING (The Safe Way)
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        # Load Datasets explicitly
        # Train gets Augmentation, Val gets None
        train_dataset = TurbidDataset(TURBID_PATH, CLEAR_PATH, DEPTH_PATH, transform=base_transform, augment=True)
        # We re-instantiate for validation to force augment=False without messing with split references
        val_dataset_raw = TurbidDataset(TURBID_PATH, CLEAR_PATH, DEPTH_PATH, transform=base_transform, augment=False)
        
        # Create indices for splitting (ensuring we use different images for train/val)
        total_len = len(train_dataset)
        val_len = int(0.1 * total_len)
        train_len = total_len - val_len
        
        # Use a fixed generator for reproducibility of the split
        split_gen = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_len, generator=split_gen).tolist()
        
        train_idx = indices[:train_len]
        val_idx = indices[train_len:]
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_idx), num_workers=4, pin_memory=True)
        # Validate on the Clean (non-augmented) dataset using the validation indices
        val_loader = DataLoader(val_dataset_raw, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_idx), num_workers=4, pin_memory=True)

        # 4. LOOP
        for epoch in range(SEARCH_EPOCHS):
            generator.train()
            
            for turbid, clear, depth in train_loader:
                turbid, clear, depth = turbid.to(device), clear.to(device), depth.to(device)
                
                # Train G
                optimizer_G.zero_grad()
                with torch.cuda.amp.autocast():
                    fake_clear = generator(turbid)
                    loss_G, _ = generator_loss(
                                    discriminator, 
                                    clear, 
                                    fake_clear, 
                                    turbid, 
                                    depth=depth, 
                                    perceptual_fn=perceptual_fn, 
                                    lambdas=lambdas
)
                scaler.scale(loss_G).backward()
                scaler.step(optimizer_G)
                scaler.update()
                
                # Train D
                optimizer_D.zero_grad()
                with torch.cuda.amp.autocast():
                    pred_real = discriminator(clear, turbid)
                    pred_fake = discriminator(fake_clear.detach(), turbid)
                    loss_D = 0.5 * (torch.mean((pred_real - 1)**2) + torch.mean(pred_fake**2))
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
                scaler.update()

            # 5. VALIDATION (PSNR Only)
            generator.eval()
            total_psnr = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for v_turbid, v_clear, _ in val_loader:
                    v_turbid, v_clear = v_turbid.to(device), v_clear.to(device)
                    v_fake = generator(v_turbid)
                    total_psnr += calculate_psnr(v_clear, v_fake)
                    num_batches += 1
            
            avg_val_psnr = total_psnr / max(num_batches, 1)
            trial.report(avg_val_psnr, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return avg_val_psnr

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f" OOM: Batch {batch_size}, NGF {ngf}. Pruning.")
            torch.cuda.empty_cache()
            return float("-inf")
        else:
            raise e
    except Exception as e:
        print("----------------------------------------------------------------")
        traceback.print_exc()  # <--- ADD THIS LINE
        print("----------------------------------------------------------------")
        return float('-inf')
    finally:
        del generator
        del discriminator
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    storage_url = "sqlite:///optuna_clearvision.db"
    study_name = "clearvision_final_v1"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction="maximize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        load_if_exists=True 
    )
    
    print(f" Tuning Started. Target: {TOTAL_TRIALS}")
    trials_left = TOTAL_TRIALS - len(study.trials)
    if trials_left > 0:
        study.optimize(objective, n_trials=trials_left)
    
    print(" Done!")
    print(f"   Best PSNR: {study.best_value}")
    with open("best_params_final.json", "w") as f:
        json.dump(study.best_params, f, indent=4)