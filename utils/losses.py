import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.color as K
import kornia.filters as KF
import math
from math import exp

# ==========================================
# 1. ADVERSARIAL LOSS (LSGAN)
# ==========================================
def adversarial_loss(pred, target_is_real=True):
    """
    Least Squares GAN Loss (Mao et al.).
    """
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return torch.mean((pred - target) ** 2)


# ==========================================
# 2. PIXEL LOSS (L1)
# ==========================================
def pixel_loss(fake, real):
    """
    Standard L1 Loss. Enforces pixel-wise accuracy.
    """
    return torch.mean(torch.abs(fake - real))


# ==========================================
# 3. LAB COLOR LOSS
# ==========================================
def lab_color_loss(fake, real):
    """
    Computes L1 loss only on 'a' and 'b' channels of CIELAB space.
    """
    # Convert [-1, 1] RGB -> [0, 1] RGB -> LAB
    fake_lab = K.rgb_to_lab((fake + 1) / 2)
    real_lab = K.rgb_to_lab((real + 1) / 2)

    return (
        torch.mean(torch.abs(fake_lab[:, 1] - real_lab[:, 1])) +
        torch.mean(torch.abs(fake_lab[:, 2] - real_lab[:, 2]))
    )


# ==========================================
# 4. EDGE LOSS (Sobel)
# ==========================================
def edge_loss(fake, real):
    """
    Computes gradients (edges) using Sobel filters.
    """
    fake_edges = KF.sobel(fake)
    real_edges = KF.sobel(real)
    return torch.mean(torch.abs(fake_edges - real_edges))


# ==========================================
# 5. DEPTH-WEIGHTED LOSS (Physics-Based)
# ==========================================
def depth_weighted_loss(fake, real, depth, max_depth=1.0):
    """
    Weights the pixel loss based on water depth.
    """
    if depth.dim() == 3:
        depth = depth.unsqueeze(1) # Ensure [B, 1, H, W]

    # Weight map: 1.0 (shallow) -> 5.0 (deep)
    weights = 1.0 + 4.0 * (depth / max_depth)
    return torch.mean(weights * torch.abs(fake - real))


# ==========================================
# 6. STANDARD SSIM LOSS (Replaces MS-SSIM)
# ==========================================
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIMLoss(torch.nn.Module):
    """
    Standard SSIM Loss.
    We prefer this over MS-SSIM for Turbid River data because downsampling (in MS-SSIM)
    destroys the fine grain of sediment, leading to blurry results.
    """
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        # Return 1 - SSIM (Minimize distance)
        return 1.0 - _ssim(img1, img2, window, self.window_size, channel, self.size_average)


# ==========================================
# 7. PERCEPTUAL LOSS (VGG-19)
# ==========================================
class PerceptualLoss(nn.Module):
    """
    Computes distance between feature maps of VGG-19.
    """
    def __init__(self, vgg):
        super().__init__()
        self.vgg = vgg
        self.layers = [2, 7, 16, 25]  # relu1_2, relu2_2, relu3_4, relu4_4

        for p in self.vgg.parameters():
            p.requires_grad = False

    def _normalize(self, x):
        # ImageNet Normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x + 1) / 2  # [-1, 1] → [0, 1]
        return (x - mean) / std

    def forward(self, fake, real):
        fake = self._normalize(fake)
        real = self._normalize(real)

        loss = torch.tensor(0.0, device=fake.device)
        x_f, x_r = fake, real

        for i, layer in enumerate(self.vgg):
            x_f = layer(x_f)
            x_r = layer(x_r)
            if i in self.layers:
                loss += torch.mean(torch.abs(x_f - x_r))

        return loss


# ==========================================
# 8. GENERATOR LOSS AGGREGATOR
# ==========================================
def generator_loss(
    D,
    real_img,
    fake_img,
    input_img,
    depth=None,
    max_depth=1.0,
    perceptual_fn=None,
    ssim_fn=None, 
    lambdas=None
):
    """
    Aggregates all 7 loss components into a single scalar.
    """
    # 1. Adversarial Loss (Fool the Discriminator)
    pred_fake = D(fake_img, input_img)
    loss_adv = adversarial_loss(pred_fake, True)

    # 2. Pixel & Color Losses
    loss_pix = pixel_loss(fake_img, real_img)
    loss_color = lab_color_loss(fake_img, real_img)
    loss_edge = edge_loss(fake_img, real_img)
    
    # 3. Advanced Metrics
    loss_perc = perceptual_fn(fake_img, real_img) if perceptual_fn else 0
    loss_ssim = ssim_fn(fake_img, real_img) if ssim_fn else 0
    
    # 4. Physics Loss
    loss_depth = 0
    if depth is not None:
        loss_depth = depth_weighted_loss(fake_img, real_img, depth, max_depth)

    # Weighted Sum
    total = (
        lambdas["adv"] * loss_adv +
        lambdas["pixel"] * loss_pix +
        lambdas["color"] * loss_color +
        lambdas["edge"] * loss_edge +
        lambdas["perc"] * loss_perc +
        lambdas["ssim"] * loss_ssim +
        lambdas["depth"] * loss_depth
    )

    # Logging Dictionary
    # FIX: Convert 'loss_ssim' back to 'SSIM Score' for display consistency
    # ssim_fn returns (1 - SSIM). So we do (1 - loss) to get the Score back.
    
    current_ssim_score = 0
    if ssim_fn:
        current_ssim_score = 1.0 - loss_ssim.item()

    loss_dict = {
        "Total": total.item(),
        "Adv": loss_adv.item(),
        "Pixel": loss_pix.item(),
        "Color": loss_color.item(),
        "Edge": loss_edge.item(),
        "SSIM": current_ssim_score, # Log the Score (e.g., 0.82), not the Loss (0.18)
        "Depth": loss_depth.item() if depth is not None else 0
    }

    return total, loss_dict