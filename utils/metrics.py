import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func
import lpips
import math
from scipy import ndimage  # Required for MuLA-GAN UIQM

# =========================================
# 1. PSNR Metric
# =========================================
def calculate_psnr(img_gt, img_gen):
    if img_gen.shape != img_gt.shape: return 0.0
    return psnr_func(img_gt, img_gen, data_range=255)

# =========================================
# 2. SSIM Metric
# =========================================
def calculate_ssim(img_gt, img_gen):
    if img_gen.shape != img_gt.shape: return 0.0
    return ssim_func(img_gt, img_gen, data_range=255, channel_axis=2)

# =========================================
# 3. LPIPS Metric
# =========================================
class LPIPSMetric:
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.loss_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device).eval()

    def calculate(self, img_gt, img_gen):
        with torch.no_grad():
            t_gt = torch.from_numpy(img_gt).float().to(self.device)
            t_gen = torch.from_numpy(img_gen).float().to(self.device)
            t_gt = (t_gt.permute(2, 0, 1).unsqueeze(0) / 255.0) * 2.0 - 1.0
            t_gen = (t_gen.permute(2, 0, 1).unsqueeze(0) / 255.0) * 2.0 - 1.0
            return self.loss_fn(t_gen, t_gt).item()

# =========================================
# 4. UIQM Metric (Official MuLA-GAN Implementation)
# =========================================
class UIQMMetric:
    def __init__(self):
        pass

    def calculate(self, img):
        try:
            return self.getUIQM(img)
        except Exception as e:
            # print(f"UIQM Error: {e}")
            return 0.0

    def mu_a(self, x, alpha_L=0.1, alpha_R=0.1):
        """ Calculates the asymmetric alpha-trimmed mean """
        x = sorted(x)
        K = len(x)
        T_a_L = math.ceil(alpha_L*K)
        T_a_R = math.floor(alpha_R*K)
        weight = (1/(K-T_a_L-T_a_R))
        s = int(T_a_L+1)
        e = int(K-T_a_R)
        val = sum(x[s:e])
        val = weight*val
        return val

    def s_a(self, x, mu):
        val = 0
        for pixel in x:
            val += math.pow((pixel-mu), 2)
        return val/len(x)

    def _uicm(self, x):
        R = x[:,:,0].flatten()
        G = x[:,:,1].flatten()
        B = x[:,:,2].flatten()
        RG = R-G
        YB = ((R+G)/2)-B
        mu_a_RG = self.mu_a(RG)
        mu_a_YB = self.mu_a(YB)
        s_a_RG = self.s_a(RG, mu_a_RG)
        s_a_YB = self.s_a(YB, mu_a_YB)
        l = math.sqrt( (math.pow(mu_a_RG,2)+math.pow(mu_a_YB,2)) )
        r = math.sqrt(s_a_RG+s_a_YB)
        return (-0.0268*l)+(0.1586*r)

    def sobel(self, x):
        dx = ndimage.sobel(x, 0)
        dy = ndimage.sobel(x, 1)
        mag = np.hypot(dx, dy)
        max_val = np.max(mag)
        if max_val == 0: return mag
        mag *= 255.0 / max_val
        return mag

    def eme(self, x, window_size):
        if x.shape[0] < window_size or x.shape[1] < window_size:
            return 0.0
            
        k1 = x.shape[1]/window_size
        k2 = x.shape[0]/window_size
        w = 2./(k1*k2)
        blocksize_x = window_size
        blocksize_y = window_size
        
        x = x[:int(blocksize_y*k2), :int(blocksize_x*k1)]
        val = 0
        for l in range(int(k1)):
            for k in range(int(k2)):
                block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1)]
                max_ = np.max(block)
                min_ = np.min(block)
                if min_ == 0.0: val += 0
                elif max_ == 0.0: val += 0
                else: val += math.log(max_/min_)
        return w*val

    def _uism(self, x):
        R = x[:,:,0]
        G = x[:,:,1]
        B = x[:,:,2]
        
        Rs = self.sobel(R)
        Gs = self.sobel(G)
        Bs = self.sobel(B)
        
        R_edge_map = np.multiply(Rs, R)
        G_edge_map = np.multiply(Gs, G)
        B_edge_map = np.multiply(Bs, B)
        
        r_eme = self.eme(R_edge_map, 10)
        g_eme = self.eme(G_edge_map, 10)
        b_eme = self.eme(B_edge_map, 10)
        
        lambda_r = 0.299
        lambda_g = 0.587
        lambda_b = 0.144
        
        return (lambda_r*r_eme) + (lambda_g*g_eme) + (lambda_b*b_eme)

    def _uiconm(self, x, window_size):
        if x.shape[0] < window_size or x.shape[1] < window_size:
            return 0.0

        k1 = x.shape[1]/window_size
        k2 = x.shape[0]/window_size
        w = -1./(k1*k2)
        blocksize_x = window_size
        blocksize_y = window_size
        
        x = x[:int(blocksize_y*k2), :int(blocksize_x*k1)]
        alpha = 1
        val = 0
        for l in range(int(k1)):
            for k in range(int(k2)):
                block = x[k*window_size:window_size*(k+1), l*window_size:window_size*(l+1), :]
                max_ = np.max(block)
                min_ = np.min(block)
                top = max_-min_
                bot = max_+min_
                
                if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0: 
                    val += 0.0
                else: 
                    val += alpha*math.pow((top/bot),alpha) * math.log(top/bot)
        return w*val

    def getUIQM(self, x):
        x = x.astype(np.float32)
        c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
        uicm   = self._uicm(x)
        uism   = self._uism(x)
        uiconm = self._uiconm(x, 10)
        uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
        return uiqm

# =========================================
# 5. WRAPPER FUNCTION
# =========================================
def calculate_uiqm(img):
    metric = UIQMMetric()
    return metric.calculate(img)