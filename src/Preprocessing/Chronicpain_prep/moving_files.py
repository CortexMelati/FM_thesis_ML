"""
=============================================================================
📂 BIDS STANDARDIZATION: CHRONIC PAIN DATASET
=============================================================================
Objective:
    Reorganize the directory structure of the external 'Chronicpainset' 
    to match the BIDS-like hierarchy used by the TDBrain dataset.

    Target Structure:
    /derivatives/sub-XX/ses-1/eeg/

    This ensures that the main preprocessing pipeline can iterate through 
    both datasets uniformly without requiring custom path logic for each.

Operations:
    1. Scans the source directory for subject folders (sub-*).
    2. Creates a parallel 'derivatives' directory.
    3. MOVES relevant EEG files (.vhdr, .eeg, .vmrk, .tsv, .json, .csv, .fif)
       into the standardized session folder (ses-1).
    4. Cleans up the old empty source directories to save space.

Execution:
    python ./FM_thesis_ML/src/Preprocessing/Chronicpain_prep/moving_files.py
=============================================================================
"""

import os
import shutil
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
current_dir = Path(__file__).resolve()
sys.path.append(str(current_dir.parents[2]))

from config import CHRONIC_PAIN_DIR

# =============================================================================
# CONFIGURATION
# =============================================================================
SOURCE_ROOT = CHRONIC_PAIN_DIR
TARGET_ROOT = CHRONIC_PAIN_DIR / "derivatives"

# Allowed file extensions to move
EXTENSIONS = (".csv", ".json", ".vhdr", ".vmrk", ".eeg", ".tsv", ".fif")

def reorganize_dataset():
    print(f"🚀 Starting Dataset Reorganization (SAFE MOVE for sub-001 to sub-074)...")
    print(f"   Source: {SOURCE_ROOT}")
    print(f"   Target: {TARGET_ROOT}")
    
    os.makedirs(TARGET_ROOT, exist_ok=True)

    subjects = []
    for d in os.listdir(SOURCE_ROOT):
        if d.startswith("sub-") and os.path.isdir(os.path.join(SOURCE_ROOT, d)):
            try:
                sub_num = int(d.split('-')[1])
                if 1 <= sub_num <= 74:
                    subjects.append(d)
            except ValueError:
                pass 
                
    subjects.sort()
    
    print(f"\nFound {len(subjects)} targeted subjects to process.")

    for sub in subjects:
            original_sub_dir = os.path.join(SOURCE_ROOT, sub)
            source_eeg_dir = os.path.join(original_sub_dir, "eeg")
            
            target_sub_dir = os.path.join(TARGET_ROOT, sub)
            target_eeg_dir = os.path.join(target_sub_dir, "ses-1", "eeg")
            os.makedirs(target_eeg_dir, exist_ok=True)

            if os.path.exists(source_eeg_dir):
                eeg_files = [f for f in os.listdir(source_eeg_dir) if f.endswith(EXTENSIONS)]
                count_eeg = 0
                for f in eeg_files:
                    src_path = os.path.join(source_eeg_dir, f)
                    dst_path = os.path.join(target_eeg_dir, f)
                    shutil.move(src_path, dst_path)
                    count_eeg += 1
                if count_eeg > 0:
                    print(f"✅ {sub}: Moved {count_eeg} EEG files.")

            root_files = [f for f in os.listdir(original_sub_dir) 
                        if os.path.isfile(os.path.join(original_sub_dir, f)) and f.endswith(EXTENSIONS)]
            count_root = 0
            for f in root_files:
                src_path = os.path.join(original_sub_dir, f)
                dst_path = os.path.join(target_sub_dir, f) 
                shutil.move(src_path, dst_path)
                count_root += 1
            if count_root > 0:
                print(f"✅ {sub}: Moved {count_root} root metadata files (e.g., .tsv).")

            try:
                if os.path.exists(source_eeg_dir):
                    os.rmdir(source_eeg_dir) 
                os.rmdir(original_sub_dir)   
            except OSError:
                pass

    print("\n" + "="*60)
    print("🎉 SAFE MOVE COMPLETE")
    print("   Targeted EEG files (sub-001 to sub-074) are moved.")
    print("   Other files and folders remain untouched.")
    print("="*60)

if __name__ == "__main__":
    reorganize_dataset()