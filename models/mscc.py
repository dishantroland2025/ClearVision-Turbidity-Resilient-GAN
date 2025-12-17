import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleColorCorrection(nn.Module):
    """
    Multi-Scale Color Correction (MSCC) Module
    
    Addresses severe color casts in turbid underwater images through three parallel branches:
    
    1. Global Branch: Scene-level color adjustment
       - Uses global average pooling and fully-connected layers
       - Learns overall color tone correction (e.g., removing brown-yellow cast)
    
    2. Regional Branch: Depth-gradient color correction
       - Pools to 8x8 grid to capture regional variations
       - Handles depth-dependent color changes (distant regions need more correction)
    
    3. Local Branch: Pixel-wise refinement
       - Uses standard convolutions for fine-grained adjustments
       - Addresses local color distortions and texture details
    
    The three branches are fused via element-wise multiplication to create
    adaptive color correction that works at multiple spatial scales.
    
    Args:
        channels: Number of input/output channels (default: 512)
    """
    def __init__(self, channels=512):
        super(MultiScaleColorCorrection, self).__init__()
        
        # Branch 1: Global (Scene-level color adjustment)
        # Architecture: GAP -> FC(512->256) -> ReLU -> FC(256->512) -> Sigmoid
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.ReLU(True),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid()
        )

        # Branch 2: Regional (Depth-gradient color correction)
        # Architecture: Pool(8x8) -> Conv(512->256) -> ReLU -> Conv(256->512) -> Sigmoid -> Upsample
        self.regional_pool = nn.AdaptiveAvgPool2d(8)
        self.regional_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(channels // 2, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Branch 3: Local (Pixel-wise refinement)
        # Architecture: Conv(512->256, 3x3) -> ReLU -> Conv(256->512, 3x3) -> Sigmoid
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 1. Global Branch - Scene-level attention
        global_attn = self.global_branch(x).view(b, c, 1, 1)
        
        # 2. Regional Branch - Depth-gradient attention
        regional_map = self.regional_pool(x)  # Pool to 8x8
        regional_attn = self.regional_conv(regional_map)
        # Upsample back to original spatial dimensions
        regional_attn = F.interpolate(regional_attn, size=(h, w), 
                                      mode='bilinear', align_corners=False)
        
        # 3. Local Branch - Pixel-wise attention
        local_attn = self.local_branch(x)

        # 4. Fusion: Average the three attention maps
        # Each branch contributes equally to the final color correction
        fused_attention = (global_attn + regional_attn + local_attn) / 3.0
        
        # Apply fused attention to input features via element-wise multiplication
        return x * fused_attention