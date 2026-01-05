import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
import lpips
import math

# =========================================
# 1. PSNR Metric
# =========================================
def calculate_psnr(img_gt, img_gen):
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR).
    Expects numpy arrays in [0, 255].
    """
    # Ensure they are the same size
    if img_gen.shape != img_gt.shape:
        return 0.0
    
    return psnr_func(img_gt, img_gen, data_range=255)

# =========================================
# 2. SSIM Metric
# =========================================
def calculate_ssim(img_gt, img_gen):
    """
    Calculates Structural Similarity Index (SSIM).
    Expects numpy arrays in [0, 255].
    """
    # Ensure they are the same size
    if img_gen.shape != img_gt.shape:
        return 0.0
    
    # Multichannel=True handles RGB images correctly
    return ssim_func(img_gt, img_gen, data_range=255, channel_axis=2)

# =========================================
# 3. LPIPS Metric (THE FIX IS HERE)
# =========================================
class LPIPSMetric:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        # Initialize LPIPS model (AlexNet is standard for speed/accuracy balance)
        self.loss_fn = lpips.LPIPS(net='alex').to(self.device).eval()

    def calculate(self, img_gt, img_gen):
        """
        Calculates LPIPS distance.
        Inputs: Numpy arrays (H, W, C) in [0, 255]
        """
        with torch.no_grad():
            # 1. Convert Numpy -> Tensor
            t_gt = torch.from_numpy(img_gt).float().to(self.device)
            t_gen = torch.from_numpy(img_gen).float().to(self.device)

            # 2. Permute Dimensions: (H, W, C) -> (1, C, H, W)
            # This fixes the "Tensor size 256 vs 3" error
            t_gt = t_gt.permute(2, 0, 1).unsqueeze(0)
            t_gen = t_gen.permute(2, 0, 1).unsqueeze(0)

            # 3. Normalize: [0, 255] -> [-1, 1] (Required by LPIPS)
            t_gt = (t_gt / 255.0) * 2.0 - 1.0
            t_gen = (t_gen / 255.0) * 2.0 - 1.0

            # 4. Calculate Distance
            dist = self.loss_fn(t_gen, t_gt)
            
        return dist.item()

# =========================================
# 4. UIQM Metric (Standard Implementation)
# =========================================
class UIQMMetric:
    def __init__(self):
        pass

    def calculate(self, img):
        """
        Calculates UIQM (Underwater Image Quality Measure).
        Input: Numpy array (H, W, C) in [0, 255] (RGB)
        """
        # Simple/Fast implementation of UIQM components
        # (UICM, UISM, UIConM)
        try:
            return self.getUIQM(img)
        except Exception as e:
            print(f"UIQM Error: {e}")
            return 0.0

    def getUIQM(self, img):
        # Weights for the 3 components (standard paper values)
        c1, c2, c3 = 0.0282, 0.2953, 3.5753
        uicm = self.getUICM(img)
        uism = self.getUISM(img)
        uiconm = self.getUIConM(img)
        return c1 * uicm + c2 * uism + c3 * uiconm

    def getUICM(self, img):
        # Underwater Image Colorfulness Measure
        img = img.astype(float)
        R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
        rg = R - G
        yb = 0.5 * (R + G) - B
        mu_rg, sig_rg = np.mean(rg), np.std(rg)
        mu_yb, sig_yb = np.mean(yb), np.std(yb)
        return math.sqrt(mu_rg**2 + mu_yb**2) + math.sqrt(sig_rg**2 + sig_yb**2)

    def getUISM(self, img):
        # Underwater Image Sharpness Measure (Simplified)
        # Using Sobel gradient magnitude as proxy for sharpness
        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        gy, gx = np.gradient(gray)
        gnorm = np.sqrt(gx**2 + gy**2)
        # EME (Enhancement Measure by Entropy) logic approx
        return np.mean(gnorm) 

    def getUIConM(self, img):
        # Underwater Image Contrast Measure (Log Michelson Contrast)
        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        # Divide into blocks (e.g., 32x32)
        h, w = gray.shape
        block_size = 32
        k_h = h // block_size
        k_w = w // block_size
        val = 0.0
        if k_h == 0 or k_w == 0: return 0.0
        
        for i in range(k_h):
            for j in range(k_w):
                block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                mn, mx = np.min(block), np.max(block)
                if mx > mn and (mx+mn) > 0:
                    val += math.log((mx-mn)/(mx+mn))
        return -1.0 / (k_h * k_w) * val # Log contrast is usually negative, we invert