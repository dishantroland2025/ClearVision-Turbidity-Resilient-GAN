import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. HELPER BLOCKS
# ==========================================

class StandardResBlock(nn.Module):
    """
    Standard ResNet block. 
    Structure: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add Residual -> ReLU
    Bias is False because BN follows convolution.
    """
    def __init__(self, channels):
        super(StandardResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    Used in skip connections to gate high-frequency noise before it hits the decoder.
    Reduction ratio 16 is standard.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        # Global Average Pooling -> MLP -> Scale Weights
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y

class ECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA-Net).
    Replaces fully connected layers with 1D convolution for speed (0.0ms latency cost).
    Kernel size 'k' is calculated dynamically based on channel depth.
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        # Calculate adaptive kernel size
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        # (B, C, 1, 1) -> (B, 1, C) for Conv1d
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    Combines Channel Attention (MLP) and Spatial Attention (7x7 Conv).
    Used in Enc4 to refine high-level features.
    """
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        # Channel Branch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        # Spatial Branch (7x7 to capture wider context)
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        # Apply Channel Attn
        mc = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        x = x * mc
        # Apply Spatial Attn
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        ms = self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * ms

class SimplifiedTripletAttention(nn.Module):
    """
    Physics-aware attention for underwater depth/color correlation.
    Rotates the tensor to mix (Channel, Height) and (Channel, Width).
    Critical for depth-dependent attenuation correction.
    """
    def __init__(self):
        super(SimplifiedTripletAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Channel-Height Interaction
        x_perm1 = x.permute(0, 3, 2, 1) # (B, W, H, C)
        max1, _ = torch.max(x_perm1, dim=1, keepdim=True)
        avg1 = torch.mean(x_perm1, dim=1, keepdim=True)
        att1 = self.sigmoid(self.conv(torch.cat([max1, avg1], dim=1)))
        out1 = (x_perm1 * att1).permute(0, 3, 2, 1)

        # 2. Channel-Width Interaction
        x_perm2 = x.permute(0, 2, 1, 3) # (B, H, W, C)
        max2, _ = torch.max(x_perm2, dim=1, keepdim=True)
        avg2 = torch.mean(x_perm2, dim=1, keepdim=True)
        att2 = self.sigmoid(self.conv(torch.cat([max2, avg2], dim=1)))
        out2 = (x_perm2 * att2).permute(0, 2, 1, 3)

        return (out1 + out2) / 2.0

class MultiScaleColorCorrection(nn.Module):
    """
    Dual-branch color corrector placed after the bottleneck.
    Global Branch: 1024-dim MLP acts as a 'lookup table' for water types.
    Local Branch: Spatial convolution for depth-dependent adjustment.
    """
    def __init__(self, channels):
        super(MultiScaleColorCorrection, self).__init__()
        # Expanded capacity (1024) + Dropout to prevent overfitting on small datasets
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, 1024), nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, channels), nn.Sigmoid()
        )
        self.local_branch = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, h, w = x.size()
        g = self.global_branch(x).view(b, c, 1, 1)
        l = self.local_branch(x)
        # Multiplicative correction (Physics: attenuation is multiplicative)
        return x * ((g + l) / 2.0)

# ==========================================
# 2. MAIN GENERATOR (Phase 2 Fixed Architecture)
# ==========================================

class ClearVisionGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32):
        super(ClearVisionGenerator, self).__init__()

        # --- ENCODER ---
        # Level 0: Input 256x256 -> 32ch. 7x7 conv for large initial receptive field.
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Level 1: 32 -> 64 channels (Downsample 256 -> 128)
        self.enc1 = StandardResBlock(ngf)
        self.down1 = nn.Conv2d(ngf, ngf*2, 3, 2, 1, bias=False)

        # Level 2: 64 -> 128 channels (Downsample 128 -> 64)
        self.enc2 = StandardResBlock(ngf*2)
        self.down2 = nn.Conv2d(ngf*2, ngf*4, 3, 2, 1, bias=False)

        # Level 3: 128 -> 256 channels (Intermediate Stage)
        # Phase 2 Fix: Added this stage to bridge the gap between 128 and 384 channels.
        # Keeps resolution at 64x64.
        self.enc3 = StandardResBlock(ngf*4)
        self.expand4 = nn.Conv2d(ngf*4, ngf*8, 1, 1, 0, bias=False) # 1x1 Expansion
        self.enc4 = nn.Sequential(StandardResBlock(ngf*8), CBAM(ngf*8)) # CBAM here on deep features
        
        # Level 4: 256 -> 384 channels (Downsample 64 -> 32)
        # This feeds the heavy bottleneck.
        self.down4 = nn.Conv2d(ngf*8, 384, 3, 2, 1, bias=False)

        # --- BOTTLENECK (High Capacity) ---
        # Resolution: 32x32. Channels: 384.
        # Split into Pre/Post with Triplet Attention in the middle.
        self.bottleneck_pre = nn.Sequential(
            StandardResBlock(384), StandardResBlock(384), StandardResBlock(384)
        )
        self.triplet = SimplifiedTripletAttention()
        self.bn_triplet = nn.BatchNorm2d(384) # Added BN to stabilize attention gradients
        self.bottleneck_post = nn.Sequential(
            StandardResBlock(384), StandardResBlock(384), StandardResBlock(384), StandardResBlock(384)
        )
        self.mscc = MultiScaleColorCorrection(channels=384)

        # --- DECODER ---
        # Standard U-Net decoder with ECA blocks added for channel calibration.
        # Note: Decoder shapes must match Encoder outputs (r4, r3, r2, r1).

        # Dec 1: Upsample 32 -> 64. Channels 384 -> 256.
        # Concats with r4 (256ch).
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bn_up1 = nn.BatchNorm2d(384) # Norm before concat is crucial
        self.skip_att1 = SEBlock(ngf*8)   # SE Gate
        self.reduce1 = nn.Conv2d(384 + ngf*8, ngf*8, 1, 1, 0, bias=False)
        self.dec1 = nn.Sequential(StandardResBlock(ngf*8), ECABlock(ngf*8))

        # Dec 2: Upsample 64 -> 128. Channels 256 -> 128.
        # Concats with r2 (Note: r2 is 64ch, r3 was processed in intermediate stage).
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.skip_att2 = SEBlock(ngf*2)
        # 256 (from up) + 64 (from skip) -> Reduce to 64
        self.reduce2 = nn.Conv2d(ngf*8 + ngf*2, ngf*2, 1, 1, 0, bias=False)
        self.dec2 = nn.Sequential(StandardResBlock(ngf*2), ECABlock(ngf*2))

        # Dec 3: Upsample 128 -> 256. Channels 64 -> 32.
        # Concats with r1 (32ch).
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.skip_att3 = SEBlock(ngf)
        self.reduce3 = nn.Conv2d(ngf*2 + ngf, ngf, 1, 1, 0, bias=False)
        self.dec3 = nn.Sequential(StandardResBlock(ngf), ECABlock(ngf))

        self.final = nn.Sequential(nn.Conv2d(ngf, output_nc, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        # --- ENCODER PASS ---
        x0 = self.initial_conv(x)       # [B, 32, 256, 256]
        r1 = self.enc1(x0)
        
        x1 = self.down1(r1)             # [B, 64, 128, 128]
        r2 = self.enc2(x1)
        
        x2 = self.down2(r2)             # [B, 128, 64, 64]
        r3 = self.enc3(x2)
        
        # Intermediate Stage (Phase 2)
        x3_exp = self.expand4(r3)       # [B, 256, 64, 64]
        r4 = self.enc4(x3_exp)          # Clean features for skip connection
        
        x4 = self.down4(r4)             # [B, 384, 32, 32] - To Bottleneck

        # --- BOTTLENECK PASS ---
        b = self.bottleneck_pre(x4)
        b = self.triplet(b)
        b = self.bn_triplet(b)
        b = self.bottleneck_post(b)
        b = self.mscc(b)                # Color correction applied here

        # --- DECODER PASS ---
        # D1: Recover to 64x64
        u1 = self.bn_up1(self.up1(b))
        s1 = self.skip_att1(r4)         # Gate high-level features
        d1 = self.dec1(self.reduce1(torch.cat([u1, s1], dim=1)))

        # D2: Recover to 128x128
        # We skip r3 (intermediate) and jump to r2 for spatial alignment
        u2 = self.up2(d1)
        s2 = self.skip_att2(r2)
        d2 = self.dec2(self.reduce2(torch.cat([u2, s2], dim=1)))

        # D3: Recover to 256x256
        u3 = self.up3(d2)
        s3 = self.skip_att3(r1)
        d3 = self.dec3(self.reduce3(torch.cat([u3, s3], dim=1)))

        return self.final(d3)
    

# ==========================================
# 3. DISCRIMINATOR
# ==========================================

class PatchGANDiscriminator(nn.Module):
    """
    River-Optimized 70x70 PatchGAN Discriminator.
    
    Configuration:
    - Normalization: BATCH NORM (Specific for Single-Domain River Data).
      Captures the global distribution of river sediment/turbidity.
    - Stabilization: SPECTRAL NORM (Critical for 25M+ Generator).
      Prevents gradient explosion during training.
    - Capacity: High (ndf=128 -> ~11M Params).
    """
    def __init__(self, input_nc=3, ndf=128):
        super(PatchGANDiscriminator, self).__init__()
        
        # Helper to create a Spectrally Normalized block with Batch Norm
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1, bias=False))]
            if normalization:
                # Batch Norm is superior for single-domain (River) datasets 
                # because it learns the domain-specific statistics (brown cast, sediment density).
                layers.append(nn.BatchNorm2d(out_filters)) 
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Input: Real/Fake (3) + Turbid (3) = 6 channels
            # Layer 1: 6 -> 128 (No Norm in first layer)
            *discriminator_block(input_nc * 2, ndf, normalization=False),
            
            # Layer 2: 128 -> 256
            *discriminator_block(ndf, ndf * 2),
            
            # Layer 3: 256 -> 512
            *discriminator_block(ndf * 2, ndf * 4),
            
            # Layer 4: 512 -> 1024 (Stride 1)
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output: 1024 -> 1 (Patch Map)
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)
        )

    def forward(self, x, condition):
        # Concatenate: [Clear/Fake, Turbid]
        img_input = torch.cat((x, condition), 1)
        return self.model(img_input)
