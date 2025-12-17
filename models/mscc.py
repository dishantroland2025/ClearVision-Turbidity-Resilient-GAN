import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleColorCorrection(nn.Module):
    def __init__(self, channels=512):
        super(MultiScaleColorCorrection, self).__init__()
        
        # Branch 1: Global (Scene-level color adjustment)
        # Input: (B, 512, H, W) -> Global Avg Pool -> (B, 512, 1, 1)
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Branch 2: Regional (Grid-level color gradients)
        # Input: (B, 512, H, W) -> Pool to 8x8 grid -> Conv -> Upsample back
        self.regional_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Branch 3: Local (Pixel-wise refinement)
        # Input: (B, 512, H, W) -> Standard Conv -> (B, 512, H, W)
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. Global Branch
        global_attn = self.global_branch(x)
        
        # 2. Regional Branch (Requires Interpolation to match size)
        regional_map = self.regional_branch(x)
        regional_attn = F.interpolate(regional_map, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 3. Local Branch
        local_attn = self.local_branch(x)

        # 4. Fusion: Average the attention maps
        fused_attention = (global_attn + regional_attn + local_attn) / 3.0
        
        # Apply to input features
        return x * fused_attention