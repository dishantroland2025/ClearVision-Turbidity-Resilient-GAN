import os
import shutil
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# The "Master" folder (The one with the correct number of images)
RAW_DIR = r"/Users/dishantdas/Sorted-UIEB/raw"

# The "Target" folder (The one with the extra image to clean)
GT_DIR = r"/Users/dishantdas/Sorted-UIEB/GT"

# Where to move the extra files (Safe Trash)
TRASH_DIR = r"/Users/dishantdas/Sorted-UIEB/_removed_files"
# ---------------------

def remove_extras():
    # 1. Create a Safe Trash folder
    os.makedirs(TRASH_DIR, exist_ok=True)

    # 2. Build the Master List of valid IDs (using stems to ignore .jpg vs .png)
    # set() is faster for searching than list()
    valid_stems = {Path(f).stem for f in os.listdir(RAW_DIR) if not f.startswith('.')}
    
    print(f"✅ Found {len(valid_stems)} valid images in Raw.")

    removed_count = 0
    
    # 3. Check every file in the GT folder
    gt_files = [f for f in os.listdir(GT_DIR) if not f.startswith('.')]
    
    for filename in tqdm(gt_files):
        stem = Path(filename).stem
        
        # If this GT image is NOT in the Raw folder...
        if stem not in valid_stems:
            src_path = os.path.join(GT_DIR, filename)
            dest_path = os.path.join(TRASH_DIR, filename)
            
            # ...Move it to trash
            shutil.move(src_path, dest_path)
            print(f"🗑️ REMOVED EXTRA: {filename}")
            removed_count += 1

    print("-" * 30)
    if removed_count == 0:
        print("🎉 No extra files found. Folders match perfectly!")
    else:
        print(f"🧹 Done. Moved {removed_count} extra files to: {TRASH_DIR}")
        print("Check that folder. If happy, you can delete it manually.")

if __name__ == "__main__":
    remove_extras()