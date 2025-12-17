import torch
from models.ClearVision import ClearVisionGenerator, Discriminator

def test_full_model():
    print("\nSTARTING FULL SYSTEM TEST...")
    
    # 1. Dummy Input (Batch=1, Channels=3, 256x256)
    img = torch.randn(1, 3, 256, 256)
    print(f"   Input Image Shape: {img.shape}")

    # 2. Test Generator
    print("   Initializing Generator...")
    gen = ClearVisionGenerator()
    try:
        fake_img = gen(img)
        print(f"   Generator Output:  {fake_img.shape}")
        
        if fake_img.shape == img.shape:
            print("   Generator Pass: SUCCESS")
        else:
            print("   Generator Pass: FAIL (Shape mismatch)")
            
    except Exception as e:
        print(f"   Generator CRASHED: {e}")
        return # Stop if generator fails

    # 3. Test Discriminator
    print("   Initializing Discriminator...")
    disc = Discriminator()
    try:
        # Discriminator takes PAIRS (Turbid, Clear)
        validity = disc(fake_img, img)
        print(f"   Discriminator Out: {validity.shape}")
        print("   Discriminator Pass: SUCCESS")
        
    except Exception as e:
        print(f"   Discriminator CRASHED: {e}")

if __name__ == "__main__":
    test_full_model()