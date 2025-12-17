import torch
from models.ClearVision import ClearVisionGenerator, Discriminator

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Initialize models
gen = ClearVisionGenerator()
disc = Discriminator()

# Count
gen_params = count_parameters(gen)
disc_params = count_parameters(disc)

print(f"\nMODEL STATISTICS:")
print(f"   --------------------------------")
print(f"   Generator Parameters:     {gen_params:,}")
print(f"   Discriminator Parameters: {disc_params:,}")
print(f"   --------------------------------")
print(f"   Total Trainable Params:   {gen_params + disc_params:,}\n")

# Comparison context (approximate)
print("   (For Reference: Standard pix2pix is ~54,000,000 params)")
if gen_params < 10000000:
    print("   VERDICT: Your model is highly efficient (Lightweight).")