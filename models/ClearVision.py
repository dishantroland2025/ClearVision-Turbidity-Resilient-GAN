import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ghost_module import GhostModule
from models.triplet_attention import TripletAttention
from models.mscc import MultiScaleColorCorrection

# -----------------------------------------------------------------------------
# Ghost Residual Block
# -----------------------------------------------------------------------------
class GhostResBlock(nn.Module):
    """
    Ghost Residual Block with Triplet Attention.
    
    Structure:
    - Two Ghost convolution layers
    - Residual shortcut connection
    - Triplet Attention after second convolution
    - BatchNorm and ReLU activations
    """
    def __init__(self, in_channels, out_channels, stride=1, use_triplet=True):
        super(GhostResBlock, self).__init__()
        
        # First Ghost convolution
        self.conv1 = GhostModule(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Second Ghost convolution
        self.conv2 = GhostModule(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Triplet Attention
        self.use_triplet = use_triplet
        if use_triplet:
            self.triplet = TripletAttention()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_triplet:
            out = self.triplet(out)
        
        out += residual
        out = self.relu(out)
        return out

# -----------------------------------------------------------------------------
# ClearVision Generator (FIXED - Correct Skip Connections)
# -----------------------------------------------------------------------------
class ClearVisionGenerator(nn.Module):
    """
    ClearVision Generator: U-Net architecture with Ghost modules,
    Triplet Attention, and Multi-Scale Color Correction.
    
    Architecture follows the research proposal exactly:
    - 4 Encoder blocks: 64 → 128 → 256 → 512 (scaled by ngf/64)
    - Bottleneck: MSCC + 4 Ghost Residual Blocks
    - 4 Decoder blocks with skip connections
    
    Performance vs Quality Trade-off (ngf parameter):
    - ngf=64:  ~15M params, Fast (~100 FPS on Jetson NX), Good quality
    - ngf=80:  ~26M params, Real-time (~80 FPS), High quality ← RECOMMENDED
    - ngf=96:  ~37M params, Slower (~50 FPS), Best quality
    - ngf=112: ~52M params, Too slow for real-time
    
    Args:
        input_nc: Number of input channels (default: 3 for RGB)
        output_nc: Number of output channels (default: 3 for RGB)
        ngf: Base number of filters (default: 80 for ~26M params, optimal for Jetson NX)
             Use ngf=64 for standard, ngf=80 for high capacity, ngf=96 for very high
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=80):
        super(ClearVisionGenerator, self).__init__()

        # Calculate channel dimensions based on ngf
        mult = ngf / 64.0  # Scaling factor
        
        c1 = int(64 * mult)   # First encoder output
        c2 = int(128 * mult)  # Second encoder output
        c3 = int(256 * mult)  # Third encoder output
        c4 = int(512 * mult)  # Fourth encoder output (bottleneck)

        # --- Initial Processing ---
        # Input: 256×256×3 -> 256×256×c1
        self.initial = nn.Sequential(
            nn.Conv2d(input_nc, c1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )

        # --- Encoder Path (4 Blocks) ---
        # Resolution: 256×256 → 256×256 → 128×128 → 64×64 → 32×32
        self.enc1 = GhostResBlock(c1, c1, stride=1)      # 256×256, c1
        self.enc2 = GhostResBlock(c1, c2, stride=2)      # 128×128, c2
        self.enc3 = GhostResBlock(c2, c3, stride=2)      # 64×64, c3
        self.enc4 = GhostResBlock(c3, c4, stride=2)      # 32×32, c4

        # --- Bottleneck Section ---
        # 1. Multi-Scale Color Correction Module
        self.mscc = MultiScaleColorCorrection(channels=c4)
        
        # 2. Four Ghost Residual Blocks
        self.bottleneck_res = nn.Sequential(
            GhostResBlock(c4, c4),
            GhostResBlock(c4, c4),
            GhostResBlock(c4, c4),
            GhostResBlock(c4, c4)
        )

        # --- Decoder Path (4 Blocks with CORRECT Skip Connections) ---
        
        # Dec 1: bottleneck(c4, 32×32) upsampled to 64×64 + skip e3(c3, 64×64)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = GhostResBlock(c4 + c3, c3)  # Input: c4+c3, Output: c3
        
        # Dec 2: dec1(c3, 64×64) upsampled to 128×128 + skip e2(c2, 128×128)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = GhostResBlock(c3 + c2, c2)  # Input: c3+c2, Output: c2
        
        # Dec 3: dec2(c2, 128×128) upsampled to 256×256 + skip e1(c1, 256×256)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = GhostResBlock(c2 + c1, c1)  # Input: c2+c1, Output: c1
        
        # Dec 4: dec3(c1, 256×256) + skip x0(c1, 256×256) - Final refinement
        self.dec4 = GhostResBlock(c1 + c1, c1)  # Input: c1+c1, Output: c1

        # --- Output Layer ---
        self.final = nn.Sequential(
            nn.Conv2d(c1, output_nc, kernel_size=7, stride=1, padding=3),
            nn.Tanh()  # Normalize to [-1, 1]
        )

    def forward(self, x):
        # Input Processing
        x0 = self.initial(x)  # c1, 256×256
        
        # Encoder with skip connections
        e1 = self.enc1(x0)    # c1, 256×256 (Skip 1)
        e2 = self.enc2(e1)    # c2, 128×128 (Skip 2)
        e3 = self.enc3(e2)    # c3, 64×64   (Skip 3)
        e4 = self.enc4(e3)    # c4, 32×32   (Skip 4)

        # Bottleneck
        b = self.mscc(e4)           # Multi-Scale Color Correction
        b = self.bottleneck_res(b)  # 4 Residual Blocks (c4, 32×32)

        # Decoder with CORRECTED skip connections
        # up1: 32×32 → 64×64, concat with e3 (64×64) ✓
        d1 = self.up1(b)                    # c4, 64×64
        d1 = torch.cat([d1, e3], dim=1)     # c4+c3, 64×64
        d1 = self.dec1(d1)                  # c3, 64×64

        # up2: 64×64 → 128×128, concat with e2 (128×128) ✓
        d2 = self.up2(d1)                   # c3, 128×128
        d2 = torch.cat([d2, e2], dim=1)     # c3+c2, 128×128
        d2 = self.dec2(d2)                  # c2, 128×128

        # up3: 128×128 → 256×256, concat with e1 (256×256) ✓
        d3 = self.up3(d2)                   # c2, 256×256
        d3 = torch.cat([d3, e1], dim=1)     # c2+c1, 256×256
        d3 = self.dec3(d3)                  # c1, 256×256
        
        # Final refinement: concat with initial features x0 (256×256) ✓
        d4 = torch.cat([d3, x0], dim=1)     # c1+c1, 256×256
        d4 = self.dec4(d4)                  # c1, 256×256

        # Output
        return self.final(d4)


# -----------------------------------------------------------------------------
# Discriminator (PatchGAN)
# -----------------------------------------------------------------------------
class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for ClearVision.
    
    Evaluates whether patches of the image are real or fake.
    Conditional discriminator - takes both generated/real image and input image.
    
    Args:
        input_nc: Number of input channels (default: 3)
        ndf: Base number of discriminator filters (default: 64)
    """
    def __init__(self, input_nc=3, ndf=64):
        super(Discriminator, self).__init__()
        
        # Input: Real/Fake (3) + Condition (3) = 6 channels
        self.model = nn.Sequential(
            # Layer 1: 6 → 64
            nn.Conv2d(input_nc * 2, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 64 → 128
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 128 → 256
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 256 → 512
            nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: 512 → 1
            nn.Conv2d(ndf * 8, 1, 4, 1, 1)
        )

    def forward(self, x, condition):
        """
        Args:
            x: Generated or real image
            condition: Input degraded image
        Returns:
            Patch-wise predictions (real/fake)
        """
        x = torch.cat([x, condition], dim=1)
        return self.model(x)
