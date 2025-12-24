import argparse
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch # <--- Added import
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from utils.metrics import calculate_psnr, calculate_ssim, calculate_uiqm, LPIPSMetric

def evaluate_folders(generated_path, gt_path, output_csv="metrics.csv"):
    print(f" Evaluating:\n  Gen: {generated_path}\n  GT:  {gt_path}")

    # --- AUTO-DETECT DEVICE (Fix for Mac M1) ---
    if torch.cuda.is_available():
        device = 'cuda'
        print("   Device: CUDA (Supercomputer)")
    elif torch.backends.mps.is_available():
        device = 'mps' # Mac M1/M2 acceleration
        print("   Device: MPS (Mac M1)")
    else:
        device = 'cpu'
        print("   Device: CPU (Slow but working)")
    # -------------------------------------------

    # Initialize LPIPS with the detected device
    lpips_calc = LPIPSMetric(device=device)

    gen_files = sorted(glob.glob(os.path.join(generated_path, "*.*")))
    results = []

    print("   Calculating PSNR, SSIM, LPIPS, UIQM...")
    
    for gen_file in tqdm(gen_files):
        name = os.path.basename(gen_file)
        gt_file = os.path.join(gt_path, name)

        if not os.path.exists(gt_file):
            continue

        # Load images as [0, 255] arrays
        img_gen = np.array(Image.open(gen_file).convert('RGB'))
        img_gt = np.array(Image.open(gt_file).convert('RGB'))

        # Resize GT if necessary to match Gen
        if img_gen.shape != img_gt.shape:
            img_gt = np.array(Image.open(gt_file).convert('RGB').resize((img_gen.shape[1], img_gen.shape[0])))

        # Calculate Metrics
        psnr = calculate_psnr(img_gt, img_gen)
        ssim = calculate_ssim(img_gt, img_gen)
        lpips = lpips_calc.calculate(img_gt, img_gen)
        uiqm = calculate_uiqm(img_gen) 

        results.append({
            "File": name,
            "PSNR": psnr,
            "SSIM": ssim,
            "LPIPS": lpips,
            "UIQM": uiqm
        })

    # Save Results
    if len(results) > 0:
        df = pd.DataFrame(results)
        print("\n Average Scores:")
        print(f"   PSNR:  {df['PSNR'].mean():.4f}")
        print(f"   SSIM:  {df['SSIM'].mean():.4f}")
        print(f"   LPIPS: {df['LPIPS'].mean():.4f}")
        print(f"   UIQM:  {df['UIQM'].mean():.4f}")
        df.to_csv(output_csv, index=False)
        print(f"   Saved detailed results to {output_csv}")
    else:
        print(" No matching image pairs found.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_path', type=str, required=True)
    parser.add_argument('--ground_truth_path', type=str, required=True)
    parser.add_argument('--output', type=str, default="metrics_results.csv")
    args = parser.parse_args()

    evaluate_folders(args.generated_path, args.ground_truth_path, args.output)