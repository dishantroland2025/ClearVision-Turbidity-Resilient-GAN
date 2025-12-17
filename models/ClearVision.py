import torch
import torch.nn as nn
import torch.nn.functional as F

# Import your custom modules
from .ghost_module import GhostModule
from .triplet_attention import TripletAttention
from .mscc import MultiScaleColorCorrection

# --- BUILDING BLOCKS ---

class GhostEncoderBlock(nn.Module):
    """ Downsampling: Ghost Conv (Stride 2) + Triplet Attention """
    def __init__(self, in_channels, out_channels):
        super(GhostEncoderBlock, self).__init__()
        # 1. Downsample using Stride=2 Ghost Module
        self.down = GhostModule(in_channels, out_channels, kernel_size=3, stride=2, relu=True)
        # 2. Refine features with Triplet Attention
        self.attn = TripletAttention()

    def forward(self, x):
        x = self.down(x)
        x = self.attn(x)
        return x

class GhostDecoderBlock(nn.Module):
    """ Upsampling: Bilinear Up + Ghost Conv + Triplet Attention """
    def __init__(self, in_channels, out_channels):
        super(GhostDecoderBlock, self).__init__()
        # 1. Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # 2. Reduce channels (input is current + skip connection)
        self.conv = GhostModule(in_channels, out_channels, kernel_size=3, stride=1, relu=True)
        # 3. Refine
        self.attn = TripletAttention()

    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Handle slight shape mismatches (common in U-Nets)
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = F.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=True)
        
        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.attn(x)
        return x

# --- MAIN GENERATOR ---

class ClearVisionGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(ClearVisionGenerator, self).__init__()

        # Initial Feature Extraction (No downsampling yet)
        self.initial = GhostModule(in_channels, 32, kernel_size=3, stride=1)

        # Encoder Path (Downsampling)
        self.enc1 = GhostEncoderBlock(32, 64)    # 256 -> 128
        self.enc2 = GhostEncoderBlock(64, 128)   # 128 -> 64
        self.enc3 = GhostEncoderBlock(128, 256)  # 64 -> 32
        self.enc4 = GhostEncoderBlock(256, 512)  # 32 -> 16

        # Bottleneck (The "Thinking" Phase)
        # 1. Color Correction
        self.color_correct = MultiScaleColorCorrection(512)
        # 2. Feature Processing
        self.bottleneck_conv = GhostModule(512, 512, kernel_size=3)

        # Decoder Path (Upsampling + Skips)
        # Input channels = Previous Decoder Output + Skip Connection
        self.dec4 = GhostDecoderBlock(512 + 256, 256) # 16 -> 32
        self.dec3 = GhostDecoderBlock(256 + 128, 128) # 32 -> 64
        self.dec2 = GhostDecoderBlock(128 + 64, 64)   # 64 -> 128
        self.dec1 = GhostDecoderBlock(64 + 32, 32)    # 128 -> 256

        # Output Layer
        self.final = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh() # Forces output to [-1, 1] range
        )

    def forward(self, x):
        # Initial
        x1 = self.initial(x) # Save for skip connection 1
        
        # Encoder
        e1 = self.enc1(x1)   # Save for skip connection 2
        e2 = self.enc2(e1)   # Save for skip connection 3
        e3 = self.enc3(e2)   # Save for skip connection 4
        e4 = self.enc4(e3)   # Bottleneck input

        # Bottleneck
        b = self.color_correct(e4)
        b = self.bottleneck_conv(b)

        # Decoder
        d4 = self.dec4(b, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, x1)

        return self.final(d1)

# --- DISCRIMINATOR (Standard PatchGAN) ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def disc_block(in_f, out_f, normalize=True):
            layers = [nn.Conv2d(in_f, out_f, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_f))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # Input: Concatenated (Real/Fake, Condition)
            *disc_block(in_channels * 2, 64, normalize=False),
            *disc_block(64, 128),
            *disc_block(128, 256),
            *disc_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img_A, img_B):
        # Concatenate Input (Turbid) and Target/Generated (Clear)
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)