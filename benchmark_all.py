import os
import subprocess
import pandas as pd
import glob
import re
import json

# --- CONFIGURATION ---
# IMPORTANT: Update these paths to match your supercomputer/laptop folders!
TEST_TURBID = "/Users/dishantdas/River-UIEB/test/raw" 
TEST_CLEAR = "/Users/dishantdas/River-UIEB/test/GT"   
TEST_DEPTH = "/Users/dishantdas/River-UIEB/test/depths"  # Not used for generation, but kept for reference
CHECKPOINT_ROOT = "checkpoints"
RESULTS_ROOT = "results"
# ---------------------

def parse_metric_output(output_str):
    """Helper to extract numbers from the print output of evaluation scripts"""
    psnr = re.search(r'PSNR:\s*([\d\.]+)', output_str)
    ssim = re.search(r'SSIM:\s*([\d\.]+)', output_str)
    lpips = re.search(r'LPIPS:\s*([\d\.]+)', output_str)
    uiqm = re.search(r'UIQM:\s*([\d\.]+)', output_str)
    
    return {
        'PSNR': float(psnr.group(1)) if psnr else 0,
        'SSIM': float(ssim.group(1)) if ssim else 0,
        'LPIPS': float(lpips.group(1)) if lpips else 0,
        'UIQM': float(uiqm.group(1)) if uiqm else 0
    }

def get_experiment_config(exp_folder):
    """Reads the auto-saved config.json to find the correct architecture settings."""
    config_path = os.path.join(exp_folder, "config.json")
    
    # Default values if config is missing (Safe fallback)
    defaults = {
        "ngf": 80, 
        "use_mscc": True, 
        "use_triplet": True
    }
    
    if not os.path.exists(config_path):
        print(f" Warning: No config.json found in {exp_folder}. Using defaults.")
        return defaults
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return {
        "ngf": config.get("ngf", 80),
        "use_mscc": config.get("use_mscc", True),
        "use_triplet": config.get("use_triplet", True)
    }

def main():
    experiments = []

    # Loop through every experiment folder in checkpoints/
    for exp_folder in glob.glob(os.path.join(CHECKPOINT_ROOT, "*")):
        if not os.path.isdir(exp_folder):
            continue
            
        exp_name = os.path.basename(exp_folder)
        
        # 1. Find the latest checkpoint
        ckpts = glob.glob(os.path.join(exp_folder, "*.pth"))
        if not ckpts:
            print(f" Skipping {exp_name}: No checkpoints found.")
            continue
            
        # Get the latest file by modification time
        latest_ckpt = max(ckpts, key=os.path.getctime)
        
        # 2. Get the correct Hyperparameters from config.json
        config = get_experiment_config(exp_folder)
        ngf_val = config["ngf"]
        
        # Convert booleans to strings 'True'/'False' for argparse compatibility
        use_mscc = str(config["use_mscc"])
        use_triplet = str(config["use_triplet"])
        
        print(f"\n Benchmarking: {exp_name}")
        print(f"   Settings: NGF={ngf_val}, MSCC={use_mscc}, Triplet={use_triplet}")
        
        # 3. GENERATE IMAGES
        # We invoke test.py using the specific settings for THIS experiment
        output_dir = os.path.join(RESULTS_ROOT, exp_name)
        
        cmd_gen = [
            "python3", "test.py",
            "--turbid_path", TEST_TURBID,
            "--checkpoint_path", latest_ckpt,
            "--results_dir", output_dir,
            "--ngf", str(ngf_val),
            "--use_mscc", use_mscc,       
            "--use_triplet", use_triplet  
        ]
        
        # Run generation
        try:
            subprocess.run(cmd_gen, check=True)
        except subprocess.CalledProcessError:
            print(f" Error generating images for {exp_name}")
            continue
        
        # 4. RUN EVALUATION
        # Calls the new evaluate_folder.py script
        cmd_eval = [
            "python3", "evaluate_folder.py",
            "--generated_path", output_dir,
            "--ground_truth_path", TEST_CLEAR
        ]
        
        result = subprocess.run(cmd_eval, capture_output=True, text=True)
        scores = parse_metric_output(result.stdout)
        
        # Add to list
        entry = {
            'Experiment': exp_name, 
            'NGF': ngf_val,
            'MSCC': use_mscc,
            'Triplet': use_triplet
        }
        entry.update(scores)
        experiments.append(entry)

    # 5. SAVE REPORT
    if experiments:
        df = pd.DataFrame(experiments)
        # Sort by PSNR descending so the winner is at the top
        df = df.sort_values(by="PSNR", ascending=False)
        df.to_csv("Final_Benchmark_Results.csv", index=False)
        print("\n Benchmarking Complete. Results saved to Final_Benchmark_Results.csv")
        print(df)
    else:
        print("\n No experiments were successfully benchmarked.")

if __name__ == "__main__":
    main()