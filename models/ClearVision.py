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
    OCEANS FIX: Only used in the deepest skip connection (Enc4).
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
        y = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * y

class ECABlock(nn.Module):
    """
    Efficient Channel Attention (ECA-Net).
    """
    def __init__(self, channels, gamma=2, b=1):
        super(ECABlock, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    """
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        mc = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        x = x * mc
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        ms = self.sigmoid(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        return x * ms

class SimplifiedTripletAttention(nn.Module):
    """
    Physics-aware attention. Kept only in Bottleneck.
    """
    def __init__(self):
        super(SimplifiedTripletAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Channel-Height Interaction
        x_perm1 = x.permute(0, 3, 2, 1) 
        max1, _ = torch.max(x_perm1, dim=1, keepdim=True)
        avg1 = torch.mean(x_perm1, dim=1, keepdim=True)
        att1 = self.sigmoid(self.conv(torch.cat([max1, avg1], dim=1)))
        out1 = (x_perm1 * att1).permute(0, 3, 2, 1)

        # 2. Channel-Width Interaction
        x_perm2 = x.permute(0, 2, 1, 3) 
        max2, _ = torch.max(x_perm2, dim=1, keepdim=True)
        avg2 = torch.mean(x_perm2, dim=1, keepdim=True)
        att2 = self.sigmoid(self.conv(torch.cat([max2, avg2], dim=1)))
        out2 = (x_perm2 * att2).permute(0, 2, 1, 3)

        return (out1 + out2) / 2.0

class MultiScaleColorCorrection(nn.Module):
    """
    Physics-Informed Novelty Block.
    """
    def __init__(self, channels):
        super(MultiScaleColorCorrection, self).__init__()
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
        return x * ((g + l) / 2.0)

# ==========================================
# 2. MAIN GENERATOR (OCEANS OPTIMIZED)
# ==========================================

class ClearVisionGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=32):
        super(ClearVisionGenerator, self).__init__()

        # --- ENCODER ---
        self.initial_conv = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 7, 1, 3, bias=False), 
            nn.BatchNorm2d(ngf), nn.ReLU(True)
        )

        # 32 -> 64
        self.enc1 = StandardResBlock(ngf)
        self.down1 = nn.Conv2d(ngf, ngf*2, 3, 2, 1, bias=False)

        # 64 -> 128
        self.enc2 = StandardResBlock(ngf*2)
        self.down2 = nn.Conv2d(ngf*2, ngf*4, 3, 2, 1, bias=False)

        # 128 -> 256
        self.enc3 = StandardResBlock(ngf*4)
        self.expand4 = nn.Conv2d(ngf*4, ngf*8, 1, 1, 0, bias=False) 
        self.enc4 = nn.Sequential(StandardResBlock(ngf*8), CBAM(ngf*8)) 
        
        # 256 -> 384 (Hardcoded High Capacity Bottleneck)
        self.down4 = nn.Conv2d(ngf*8, 384, 3, 2, 1, bias=False)

        # --- BOTTLENECK (High Capacity) ---
        self.bottleneck_pre = nn.Sequential(
            StandardResBlock(384), StandardResBlock(384), StandardResBlock(384)
        )
        self.triplet = SimplifiedTripletAttention()
        self.bn_triplet = nn.BatchNorm2d(384) 
        self.bottleneck_post = nn.Sequential(
            StandardResBlock(384), StandardResBlock(384), StandardResBlock(384), StandardResBlock(384)
        )
        self.mscc = MultiScaleColorCorrection(channels=384)

        # --- DECODER ---
        
        # D1: Upsample 32 -> 64. Channels 384 -> 256.
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.bn_up1 = nn.BatchNorm2d(384) 
        
        # [KEPT] SE Gate on Deepest Skip (Filters Semantic Noise)
        self.skip_att1 = SEBlock(ngf*8)   
        
        self.reduce1 = nn.Conv2d(384 + ngf*8, ngf*8, 1, 1, 0, bias=False)
        self.dec1 = nn.Sequential(StandardResBlock(ngf*8), ECABlock(ngf*8))

        # D2: Upsample 64 -> 128. Channels 256 -> 128.
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # [REMOVED] SE Block from Shallow Skip (Unlocks Texture Flow)
        # self.skip_att2 = SEBlock(ngf*2) 
        
        self.reduce2 = nn.Conv2d(ngf*8 + ngf*2, ngf*2, 1, 1, 0, bias=False)
        self.dec2 = nn.Sequential(StandardResBlock(ngf*2), ECABlock(ngf*2))

        # D3: Upsample 128 -> 256. Channels 64 -> 32.
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # [REMOVED] SE Block from Shallow Skip (Unlocks Texture Flow)
        # self.skip_att3 = SEBlock(ngf)
        
        self.reduce3 = nn.Conv2d(ngf*2 + ngf, ngf, 1, 1, 0, bias=False)
        self.dec3 = nn.Sequential(StandardResBlock(ngf), ECABlock(ngf))

        self.final = nn.Sequential(nn.Conv2d(ngf, output_nc, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        # --- ENCODER ---
        x0 = self.initial_conv(x)
        r1 = self.enc1(x0)
        
        x1 = self.down1(r1)
        r2 = self.enc2(x1)
        
        x2 = self.down2(r2)
        r3 = self.enc3(x2)
        
        x3_exp = self.expand4(r3)
        r4 = self.enc4(x3_exp)
        
        x4 = self.down4(r4)

        # --- BOTTLENECK ---
        b = self.bottleneck_pre(x4)
        b = self.triplet(b)
        b = self.bn_triplet(b)
        b = self.bottleneck_post(b)
        b = self.mscc(b)

        # --- DECODER ---
        # D1
        u1 = self.bn_up1(self.up1(b))
        s1 = self.skip_att1(r4) # [KEPT] Deepest SE Gate
        d1 = self.dec1(self.reduce1(torch.cat([u1, s1], dim=1)))

        # D2
        u2 = self.up2(d1)
        # s2 = self.skip_att2(r2) [REMOVED]
        d2 = self.dec2(self.reduce2(torch.cat([u2, r2], dim=1))) # Direct Skip

        # D3
        u3 = self.up3(d2)
        # s3 = self.skip_att3(r1) [REMOVED]
        d3 = self.dec3(self.reduce3(torch.cat([u3, r1], dim=1))) # Direct Skip

        return self.final(d3)
    

# ==========================================
# 3. DISCRIMINATOR
# ==========================================

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=128):
        super(PatchGANDiscriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.utils.spectral_norm(nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1, bias=False))]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters)) 
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_nc * 2, ndf, normalization=False),
            *discriminator_block(ndf, ndf * 2),
            *discriminator_block(ndf * 2, ndf * 4),
            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 4, stride=1, padding=1, bias=False)),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=1)
        )

    def forward(self, x, condition):
        img_input = torch.cat((x, condition), 1)
        return self.model(img_input)