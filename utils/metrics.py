import torch
import torch.nn as nn
import numpy as np
import math
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
import cv2

# --- 1. Standard Reference Metrics (PSNR, SSIM, LPIPS) ---

def calculate_psnr(img1, img2):
    """Calculates PSNR (Peak Signal-to-Noise Ratio). Inputs: [0, 255] numpy arrays."""
    return peak_signal_noise_ratio(img1, img2, data_range=255)

def calculate_ssim(img1, img2):
    """Calculates SSIM. Inputs: [0, 255] numpy arrays."""
    # multichannel=True is critical for RGB images
    return structural_similarity(img1, img2, data_range=255, channel_axis=2, multichannel=True)

# LPIPS requires a model download, initialized once to save time
class LPIPSMetric:
    def __init__(self, device='cuda'):
        self.device = device
        # AlexNet is the standard backbone for LPIPS comparison
        self.loss_fn = lpips.LPIPS(net='alex').to(device).eval()

    def calculate(self, img1, img2):
        """
        Inputs: Numpy arrays [H, W, C] in range [0, 255].
        Converts to Tensor [-1, 1] for LPIPS model.
        """
        # Normalize to [-1, 1]
        t1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 127.5 - 1.0
        t2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 127.5 - 1.0
        
        t1 = t1.unsqueeze(0).to(self.device)
        t2 = t2.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            dist = self.loss_fn(t1, t2)
        return dist.item()


# --- 2. Underwater Specific Metric (UIQM) ---
# This is a Python port of the standard UIQM logic (Panetta et al.)

def calculate_uiqm(img):
    """
    Underwater Image Quality Measure (UIQM).
    Inputs: RGB numpy array [0, 255].
    """
    # Convert to standard weights used in underwater papers
    # UIQM = c1*UICM + c2*UISM + c3*UIConM
    c1, c2, c3 = 0.0282, 0.2953, 3.5753 
    
    uicm = _uicm(img)
    uism = _uism(img)
    uiconm = _uiconm(img)
    
    return c1 * uicm + c2 * uism + c3 * uiconm

def _uicm(img):
    """Underwater Image Colorfulness Measure"""
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    
    # Standard deviation and mean calculation
    mu_rg, sig_rg = np.mean(rg), np.std(rg)
    mu_yb, sig_yb = np.mean(yb), np.std(yb)
    
    return -0.0268 * np.sqrt(mu_rg**2 + mu_yb**2) + 0.1586 * np.sqrt(sig_rg**2 + sig_yb**2)

def _uism(img):
    """Underwater Image Sharpness Measure"""
    # Use standard Sobel-based sharpness
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    # Contrast at each pixel
    edge_map = np.sqrt(gx**2 + gy**2)
    return np.mean(edge_map)

def _uiconm(img):
    """Underwater Image Contrast Measure (LogAMEE)"""
    # Simplified implementation using entropy-like contrast
    # A full LogAMEE implementation is very slow; this is a standard approximation
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Split into blocks (e.g., 32x32)
    h, w = gray.shape
    k = 32
    entropy_vals = []
    
    for i in range(0, h-k, k):
        for j in range(0, w-k, k):
            block = gray[i:i+k, j:j+k]
            max_val = np.max(block)
            min_val = np.min(block)
            if max_val > min_val:
                val = (max_val - min_val) / (max_val + min_val)
                entropy_vals.append(val)
    
    if len(entropy_vals) == 0: return 0
    return np.mean(entropy_vals)