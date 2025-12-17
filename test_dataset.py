import torch
from torchvision import transforms
from utils.dataset import TurbidDataset

def test_data_loading():
    print("\nSTARTING DATASET TEST (WITH DEPTH)...")
    
    # --- UPDATE PATHS ---
    turbid_path = "/Users/dishantdas/River-UIEB/train/raw"
    clear_path = "/Users/dishantdas/River-UIEB/train/GT"
    depth_path = "/Users/dishantdas/River-UIEB/train/depths"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    try:
        # Initialize Dataset
        dataset = TurbidDataset(turbid_path, clear_path, depth_path, transform=transform)
        print(f"   Found {len(dataset)} valid triplets.")
        
        if len(dataset) == 0:
            print("   ERROR: No images found! Check your paths.")
            return

        # Get the first triplet
        turbid, clear, depth = dataset[0]
        
        print(f"   Turbid Shape: {turbid.shape} (Should be 3, 256, 256)")
        print(f"   Depth Shape:  {depth.shape}  (Should be 1, 256, 256)")
        
        if turbid.shape[0] == 3 and depth.shape[0] == 1:
            print("   Data Loading: SUCCESS")
        else:
            print("   Data Loading: FAILED (Channel mismatch)")

    except Exception as e:
        print(f"   CRASHED: {e}")

if __name__ == "__main__":
    test_data_loading()