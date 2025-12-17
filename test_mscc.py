import torch
from models.mscc import MultiScaleColorCorrection

def test_mscc():
    print("\n STARTING MSCC MODULE TEST...")
    
    # Input: Batch=2, Channels=512, Height=16, Width=16 
    # (Simulating the Bottleneck feature map size)
    x = torch.randn(2, 512, 16, 16)
    print(f"   Input Shape: {x.shape}")
    
    # Initialize MSCC
    mscc = MultiScaleColorCorrection(channels=512)
    
    try:
        out = mscc(x)
        print(f"   Output Shape: {out.shape}")
        
        # Verification
        if out.shape == x.shape:
            print("SUCCESS: Multi-Scale Color Correction fused correctly.")
        else:
            print(f"FAILURE: Output shape mismatch! Got {out.shape}")
            
    except Exception as e:
        print(f"CRASHED: {e}")

if __name__ == "__main__":
    test_mscc()