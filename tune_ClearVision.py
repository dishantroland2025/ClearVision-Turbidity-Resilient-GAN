# import optuna
# import argparse
# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from tqdm import tqdm
# import json
# import gc 
# import traceback

# # --- IMPORTS ---
# from models.ClearVision import ClearVisionGenerator, PatchGANDiscriminator
# from utils.dataset import TurbidDataset
# from utils.losses import generator_loss, PerceptualLoss, MSSSIMLoss

# # --- CONFIG ---
# SEARCH_EPOCHS = 15     
# TOTAL_TRIALS = 100     
# IMG_SIZE = 256

# # --- HELPER: PSNR ---
# def calculate_psnr(img1, img2):
#     mse = torch.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return 100
#     return 20 * torch.log10(1.0 / torch.sqrt(mse))

# def objective(trial):
#     # 1. Search Space
#     batch_size = trial.suggest_categorical("batch_size", [4, 8])
#     lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
    
#     # Loss Weights
#     lambda_ssim  = trial.suggest_float("lambda_ssim", 0.1, 1.0)
#     lambda_depth = trial.suggest_float("lambda_depth", 0.1, 5.0)
#     lambda_perc  = trial.suggest_float("lambda_perc", 0.01, 0.5)
#     lambda_edge  = trial.suggest_float("lambda_edge", 0.1, 2.0)
#     lambda_adv   = trial.suggest_float("lambda_adv", 0.01, 0.2)

#     # 2. Setup
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # ⚠️ UPDATE THESE PATHS BEFORE RUNNING ⚠️
#     TURBID_PATH = "/path/to/Sorted-UIEB/Raw"
#     CLEAR_PATH = "/path/to/Sorted-UIEB/GT"
#     DEPTH_PATH = "/path/to/Sorted-UIEB/depths"
    
#     try:
#         # Init Models
#         generator = ClearVisionGenerator(ngf=32).to(device)
#         discriminator = PatchGANDiscriminator(ndf=128).to(device) 
        
#         # Init Optimizers
#         optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
#         optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
#         scaler = torch.cuda.amp.GradScaler()
        
#         # Init Losses
#         from torchvision.models import vgg19
#         vgg = vgg19(weights='DEFAULT').features.to(device).eval()
#         perceptual_fn = PerceptualLoss(vgg).to(device)
#         ssim_fn = MSSSIMLoss().to(device)
        
#         lambdas = {
#             "adv": lambda_adv, 
#             "pixel": 10.0,       
#             "color": 0.5,        
#             "edge": lambda_edge, 
#             "perc": lambda_perc, 
#             "ssim": lambda_ssim, 
#             "depth": lambda_depth 
#         }

#         # 3. Data Loading
#         base_transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#         ])
        
#         train_dataset = TurbidDataset(TURBID_PATH, CLEAR_PATH, DEPTH_PATH, transform=base_transform, augment=True)
#         val_dataset_raw = TurbidDataset(TURBID_PATH, CLEAR_PATH, DEPTH_PATH, transform=base_transform, augment=False)
        
#         total_len = len(train_dataset)
#         val_len = int(0.1 * total_len)
#         train_len = total_len - val_len
        
#         split_gen = torch.Generator().manual_seed(42)
#         indices = torch.randperm(total_len, generator=split_gen).tolist()
        
#         train_loader = DataLoader(
#             train_dataset, batch_size=batch_size, 
#             sampler=torch.utils.data.SubsetRandomSampler(indices[:train_len]), 
#             num_workers=4, pin_memory=True
#         )
#         val_loader = DataLoader(
#             val_dataset_raw, batch_size=batch_size, 
#             sampler=torch.utils.data.SubsetRandomSampler(indices[train_len:]), 
#             num_workers=4, pin_memory=True
#         )

#         # 4. Loop
#         for epoch in range(SEARCH_EPOCHS):
#             generator.train()
            
#             for turbid, clear, depth in train_loader:
#                 turbid, clear, depth = turbid.to(device), clear.to(device), depth.to(device)
                
#                 # --- Train G ---
#                 optimizer_G.zero_grad()
#                 with torch.cuda.amp.autocast():
#                     fake_clear = generator(turbid)
                    
#                     # [MODIFICATION]: KEYWORD ARGUMENTS ONLY (Prevents Positional Crashes)
#                     loss_G, _ = generator_loss(
#                         D=discriminator, 
#                         real_img=clear, 
#                         fake_img=fake_clear, 
#                         input_img=turbid, 
#                         depth=depth, 
#                         perceptual_fn=perceptual_fn, 
#                         ssim_fn=ssim_fn, 
#                         lambdas=lambdas
#                     )
                
#                 scaler.scale(loss_G).backward()
#                 scaler.unscale_(optimizer_G)
#                 torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
#                 scaler.step(optimizer_G)
#                 scaler.update()
                
#                 # --- Train D ---
#                 optimizer_D.zero_grad()
#                 with torch.cuda.amp.autocast():
#                     pred_real = discriminator(clear, turbid)
#                     pred_fake = discriminator(fake_clear.detach(), turbid)
#                     loss_D = 0.5 * (torch.mean((pred_real - 1)**2) + torch.mean(pred_fake**2))
                    
#                 scaler.scale(loss_D).backward()
#                 scaler.step(optimizer_D)
#                 scaler.update()

#             # 5. Validation
#             generator.eval()
#             total_psnr = 0.0
#             count = 0
            
#             with torch.no_grad():
#                 for v_turbid, v_clear, _ in val_loader:
#                     v_turbid, v_clear = v_turbid.to(device), v_clear.to(device)
#                     v_fake = (generator(v_turbid) + 1) / 2.0
#                     v_clear_dn = (v_clear + 1) / 2.0
#                     total_psnr += calculate_psnr(v_clear_dn, v_fake)
#                     count += 1
            
#             avg_psnr = total_psnr / max(count, 1)
            
#             trial.report(avg_psnr, epoch)
#             if trial.should_prune():
#                 raise optuna.exceptions.TrialPruned()

#         return avg_psnr.item()

#     except optuna.exceptions.TrialPruned as e:
#         raise e
#     except RuntimeError as e:
#         if "out of memory" in str(e):
#             print(f" OOM: Batch {batch_size}. Pruning.")
#             torch.cuda.empty_cache()
#             return float("-inf")
#         else:
#             traceback.print_exc()
#             return float("-inf")
#     except Exception as e:
#         traceback.print_exc()
#         return float('-inf')
#     finally:
#         del generator, discriminator, optimizer_G, optimizer_D
#         torch.cuda.empty_cache()
#         gc.collect()

# if __name__ == "__main__":
#     db_path = os.path.abspath("optuna_clearvision_phase2.db")
#     storage_url = f"sqlite:///{db_path}"
#     study_name = "cv_phase2_anchored"
    
#     print(f"--- 5D Anchored Tuning (100 Trials) ---")
#     print(f" Database: {storage_url}")
    
#     study = optuna.create_study(
#         study_name=study_name, storage=storage_url, direction="maximize",
#         pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
#         load_if_exists=True 
#     )
    
#     trials_completed = len(study.trials)
#     trials_left = TOTAL_TRIALS - trials_completed
    
#     if trials_left > 0:
#         print(f" Resuming from Trial {trials_completed + 1}. Runs left: {trials_left}")
#         study.optimize(objective, n_trials=trials_left, gc_after_trial=True)
#         print("--- Target Reached! ---")
#     else:
#         print(f" Tuning Complete ({trials_completed} trials done).")
    
#     print(f" Best PSNR: {study.best_value:.4f}")
#     print(" Best Params:")
#     print(json.dumps(study.best_params, indent=4))
    
#     with open("best_hyperparameters.json", "w") as f:
#         json.dump(study.best_params, f, indent=4)












import optuna
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import json
import gc 
import traceback

# --- IMPORTS ---
from models.ClearVision import ClearVisionGenerator, PatchGANDiscriminator
from utils.dataset import TurbidDataset
from utils.losses import generator_loss, PerceptualLoss, MSSSIMLoss

# --- CONFIG ---
SEARCH_EPOCHS = 30     
TOTAL_TRIALS = 50     
IMG_SIZE = 256

# --- HELPER: PSNR ---
def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def objective(trial):
    # 1. Search Space (Anchored around your best known params)
    batch_size = 16
    lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
    
    # Loss Weights
    lambda_pixel = trial.suggest_float("lambda_pixel", 30.0, 70.0)
    lambda_adv   = trial.suggest_float("lambda_adv", 0.1, 1.0)
    lambda_ssim  = trial.suggest_float("lambda_ssim", 0.0, 0.2)
    lambda_color = trial.suggest_float("lambda_color", 0.5, 2.0)
    lambda_edge  = trial.suggest_float("lambda_edge", 0.1, 1.0)
    lambda_perc  = trial.suggest_float("lambda_perc", 0.5, 2.0)
    lambda_depth = trial.suggest_float("lambda_depth", 0.1, 1.0)

    # 2. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths updated to match your bash script
    TURBID_PATH = "/content/dataset/Turbid_UIEB_Split/train/Raw"
    CLEAR_PATH = "/content/dataset/Turbid_UIEB_Split/train/GT"
    DEPTH_PATH = "/content/dataset/Turbid_UIEB_Split/train/depths"
    
    try:
        # Init Models with your optimized ngf=48
        generator = ClearVisionGenerator(ngf=48).to(device)
        discriminator = PatchGANDiscriminator(ndf=128).to(device) 
        
        # Init Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        scaler = torch.cuda.amp.GradScaler()
        
        # Init Losses
        from torchvision.models import vgg19
        vgg = vgg19(weights='DEFAULT').features.to(device).eval()
        perceptual_fn = PerceptualLoss(vgg).to(device)
        ssim_fn = MSSSIMLoss().to(device)
        
        lambdas = {
            "adv": lambda_adv, 
            "pixel": lambda_pixel,       
            "color": lambda_color,        
            "edge": lambda_edge, 
            "perc": lambda_perc, 
            "ssim": lambda_ssim, 
            "depth": lambda_depth 
        }

        # 3. Data Loading
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        train_dataset = TurbidDataset(TURBID_PATH, CLEAR_PATH, DEPTH_PATH, transform=base_transform, augment=True)
        val_dataset_raw = TurbidDataset(TURBID_PATH, CLEAR_PATH, DEPTH_PATH, transform=base_transform, augment=False)
        
        total_len = len(train_dataset)
        val_len = int(0.1 * total_len)
        train_len = total_len - val_len
        
        split_gen = torch.Generator().manual_seed(42)
        indices = torch.randperm(total_len, generator=split_gen).tolist()
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, 
            sampler=torch.utils.data.SubsetRandomSampler(indices[:train_len]), 
            num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset_raw, batch_size=batch_size, 
            sampler=torch.utils.data.SubsetRandomSampler(indices[train_len:]), 
            num_workers=4, pin_memory=True
        )

        # 4. Loop
        for epoch in range(SEARCH_EPOCHS):
            generator.train()
            
            for turbid, clear, depth in train_loader:
                turbid, clear, depth = turbid.to(device), clear.to(device), depth.to(device)
                
                # --- Train G ---
                optimizer_G.zero_grad()
                with torch.cuda.amp.autocast():
                    fake_clear = generator(turbid)
                    
                    loss_G, _ = generator_loss(
                        D=discriminator, 
                        real_img=clear, 
                        fake_img=fake_clear, 
                        input_img=turbid, 
                        depth=depth, 
                        perceptual_fn=perceptual_fn, 
                        ssim_fn=ssim_fn, 
                        lambdas=lambdas
                    )
                
                scaler.scale(loss_G).backward()
                scaler.unscale_(optimizer_G)
                torch.nn.utils.clip_grad_norm_(generator.parameters(), 10.0)
                scaler.step(optimizer_G)
                scaler.update()
                
                # --- Train D ---
                optimizer_D.zero_grad()
                with torch.cuda.amp.autocast():
                    pred_real = discriminator(clear, turbid)
                    pred_fake = discriminator(fake_clear.detach(), turbid)
                    loss_D = 0.5 * (torch.mean((pred_real - 1)**2) + torch.mean(pred_fake**2))
                    
                scaler.scale(loss_D).backward()
                scaler.step(optimizer_D)
                scaler.update()

            # 5. Validation
            generator.eval()
            total_psnr = 0.0
            count = 0
            
            with torch.no_grad():
                for v_turbid, v_clear, _ in val_loader:
                    v_turbid, v_clear = v_turbid.to(device), v_clear.to(device)
                    v_fake = (generator(v_turbid) + 1) / 2.0
                    v_clear_dn = (v_clear + 1) / 2.0
                    total_psnr += calculate_psnr(v_clear_dn, v_fake)
                    count += 1
            
            avg_psnr = total_psnr / max(count, 1)
            
            trial.report(avg_psnr, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return avg_psnr.item()

    except optuna.exceptions.TrialPruned as e:
        raise e
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f" OOM: Batch {batch_size}. Pruning.")
            torch.cuda.empty_cache()
            return float("-inf")
        else:
            traceback.print_exc()
            return float("-inf")
    except Exception as e:
        traceback.print_exc()
        return float('-inf')
    finally:
        try:
            del generator, discriminator, optimizer_G, optimizer_D
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    db_path = os.path.abspath("optuna_clearvision_phase1.db")
    storage_url = f"sqlite:///{db_path}"
    study_name = "cv_phase1_anchored"
    
    print(f"--- Phase 1 Anchored Tuning ({TOTAL_TRIALS} Trials) ---")
    
    study = optuna.create_study(
        study_name=study_name, storage=storage_url, direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        load_if_exists=True 
    )
    
    # Enqueue your known best parameters as the very first trial
    study.enqueue_trial({
        "batch_size": 16,
        "lr": 0.0002,
        "lambda_pixel": 50.0,
        "lambda_adv": 0.5,
        "lambda_ssim": 0.0,
        "lambda_color": 1.0,
        "lambda_edge": 0.5,
        "lambda_perc": 1.0,
        "lambda_depth": 0.5
    })
    
    trials_completed = len(study.trials)
    trials_left = TOTAL_TRIALS - trials_completed
    
    if trials_left > 0:
        print(f" Resuming from Trial {trials_completed + 1}. Runs left: {trials_left}")
        study.optimize(objective, n_trials=trials_left, gc_after_trial=True)
        print("--- Target Reached! ---")
    else:
        print(f" Tuning Complete ({trials_completed} trials done).")
    
    print(f" Best PSNR: {study.best_value:.4f}")
    with open("best_hyperparameters_phase1.json", "w") as f:
        json.dump(study.best_params, f, indent=4)