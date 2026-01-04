import torch
import torch.nn as nn
# Updated Import for Phase 2
from models.ClearVision import ClearVisionGenerator, PatchGANDiscriminator

def test_full_model():
    print("\n-------------------------------------------")
    print(" STARTING PRE-FLIGHT SYSTEM CHECK")
    print("-------------------------------------------")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Testing on Device: {device}")

    # 1. Dummy Input (Batch=2, Channels=3, 256x256)
    # Testing Batch=2 ensures Batch Norm doesn't crash (it fails on Batch=1 in training mode)
    img = torch.randn(2, 3, 256, 256).to(device)
    print(f"   Input Tensor Shape: {img.shape}")

    # 2. Test Generator (Phase 2 Spec)
    print("\n   [1/3] Initializing Generator (ngf=32)...")
    try:
        gen = ClearVisionGenerator(ngf=32).to(device)
        print(f"       Parameter Count: {sum(p.numel() for p in gen.parameters())/1e6:.2f} Million")
        
        fake_img = gen(img)
        print(f"       Output Shape:    {fake_img.shape}")
        
        if fake_img.shape == img.shape:
            print("       Generator Pass: Shape Match")
        else:
            print("       Generator Fail: Shape Mismatch")
            return
            
    except Exception as e:
        print(f"       Generator CRASHED: {e}")
        return 

    # 3. Test Discriminator (PatchGAN)
    print("\n   [2/3] Initializing Discriminator (ndf=128)...")
    try:
        disc = PatchGANDiscriminator(ndf=128).to(device)
        print(f"       Parameter Count: {sum(p.numel() for p in disc.parameters())/1e6:.2f} Million")
        
        # Discriminator takes Concatenated input (Real + Turbid)
        # We simulate the forward pass
        validity = disc(img, img) # (Real, Condition)
        
        print(f"       Output Shape:    {validity.shape}")
        # PatchGAN output should be (Batch, 1, 30, 30) or similar, NOT scalar
        if validity.dim() == 4:
             print("       Discriminator Pass: Patch Output Correct")
        else:
             print("       Discriminator Warning: Expected 4D Patch Output")
        
    except Exception as e:
        print(f"       Discriminator CRASHED: {e}")
        return

    print("\n-------------------------------------------")
    print(" SYSTEM CHECK COMPLETE: READY FOR LAUNCH 🚀")
    print("-------------------------------------------\n")

if __name__ == "__main__":
    test_full_model()