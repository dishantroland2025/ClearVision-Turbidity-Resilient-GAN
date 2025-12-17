import torch
import torch.nn as nn
import math

class GhostModule(nn.Module):
    """
    Ghost Module: Generates more features from cheap operations.
    Reduces parameters while maintaining capacity.
    
    Paper: GhostNet: More Features from Cheap Operations
    Standard ratio=2 means half the features are intrinsic, half are ghost.
    
    Args:
        inp: Number of input channels
        oup: Number of output channels
        kernel_size: Kernel size for primary convolution (default: 3)
        ratio: Ratio for splitting intrinsic and ghost features (default: 2, standard)
        dw_size: Kernel size for depthwise convolution (default: 3)
        stride: Stride for convolution (default: 1)
        relu: Whether to use ReLU activation (default: True)
    """
    def __init__(self, inp, oup, kernel_size=3, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        # Calculate intrinsic and ghost feature channels
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # 1. Primary Convolution (Standard, but with fewer filters)
        # Generates intrinsic features
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # 2. Cheap Operation (Depthwise Conv)
        # Generates "Ghost" features from the intrinsic ones
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        # Generate intrinsic features
        x1 = self.primary_conv(x)
        
        # Generate ghost features from intrinsic features
        x2 = self.cheap_operation(x1)
        
        # Concatenate intrinsic and ghost features
        out = torch.cat([x1, x2], dim=1)
        
        # Slice to exact output channels (handles rounding in channel calculation)
        return out[:, :self.oup, :, :]