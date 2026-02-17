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
    3. Copies relevant EEG files (.vhdr, .eeg, .vmrk, .tsv, .json, .csv, .fif)
       into the standardized session folder (ses-1).

Execution:
    python ./FM_thesis_ML/src/Preprocessing/Chronicpain prep/moving_files.py
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

# Allowed file extensions to copy
EXTENSIONS = (".csv", ".json", ".vhdr", ".vmrk", ".eeg", ".tsv", ".fif")

def reorganize_dataset():
    print(f"🚀 Starting Dataset Reorganization...")
    print(f"   Source: {SOURCE_ROOT}")
    print(f"   Target: {TARGET_ROOT}")
    
    # 1. Create main derivatives directory
    os.makedirs(TARGET_ROOT, exist_ok=True)

    # 2. Identify Subjects
    # Note: Iterate over SOURCE_ROOT, filtering for directories starting with 'sub-'
    subjects = [d for d in os.listdir(SOURCE_ROOT) if d.startswith("sub-")]
    print(f"\nFound {len(subjects)} subjects to process.")

    for sub in subjects:
        # Define source path
        source_eeg_dir = os.path.join(SOURCE_ROOT, sub, "eeg")
        
        if not os.path.exists(source_eeg_dir):
            print(f"[SKIP] No EEG directory found for {sub}")
            continue

        # Define target path (BIDS structure: sub -> ses-1 -> eeg)
        target_eeg_dir = os.path.join(TARGET_ROOT, sub, "ses-1", "eeg")
        os.makedirs(target_eeg_dir, exist_ok=True)

        # Find relevant files
        eeg_files = [
            f for f in os.listdir(source_eeg_dir)
            if f.endswith(EXTENSIONS)
        ]

        if not eeg_files:
            print(f"[SKIP] No valid EEG files found for {sub}")
            continue

        # Copy files
        count = 0
        for f in eeg_files:
            src_path = os.path.join(source_eeg_dir, f)
            dst_path = os.path.join(target_eeg_dir, f)
            
            # Use copy2 to preserve metadata (timestamps)
            shutil.copy2(src_path, dst_path)
            count += 1
            
        print(f"✅ {sub}: Copied {count} files to standardized folder.")

    print("\n" + "="*60)
    print("🎉 REORGANIZATION COMPLETE")
    print("   All Chronicpainset files (incl. .fif & .tsv) are now in TDBrain structure.")
    print("   Note: 'TDBRAIN_participants_V2.xlsx' remains in the root folder.")
    print("="*60)

if __name__ == "__main__":
    reorganize_dataset()