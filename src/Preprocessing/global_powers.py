"""
=============================================================================
VALIDATION PREPARATION: CALCULATE GLOBAL POWERS
=============================================================================
Objective:
    1. Load the master dataset ('final_dataset.csv').
    2. Calculate the Global Average Power across all 20 EEG channels for 
       each frequency band (Delta through Gamma).
    3. Ensure metadata (specifically Age) is attached, attempting a fallback 
       merge with external metadata if missing.
    4. Save the result as 'validation_global_powers.csv' for use in 
       physiological validation scripts (e.g., Berger Effect, Age Correlation).

Input:
    - results/final_dataset.csv
    - (Optional) Data/Chronicpainset/participants.tsv (Fallback for metadata)

Output:
    - results/validation_global_powers.csv

Execution:
    python ./FM_thesis_ML/src/Preprocessing/global_powers.py
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import RESULTS_DIR, CHRONIC_PAIN_DIR

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

# INPUT: The combined master dataset (TDBrain + External)
INPUT_FILE = RESULTS_DIR / "final_dataset.csv"

# OUTPUT: The compact validation file
OUTPUT_FILE = RESULTS_DIR / "validation_global_powers.csv"

# Metadata backup (Fallback for External set if Age is missing)
META_EXTERNAL = CHRONIC_PAIN_DIR / "participants.tsv"

def calculate_global_bands():
    print("🧠 Calculating Global Band Powers for validation...")
    print("-" * 50)
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        print("   -> Have you run 'final_prep.py' yet?")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"📄 Data loaded: {len(df)} epochs.")
    
    # --- 1. METADATA CHECK (Add Age if missing) ---
    if 'Age' not in df.columns:
        print("   ⚠️ 'Age' column missing. Attempting to merge from participants.tsv...")
        if os.path.exists(META_EXTERNAL):
            try:
                meta_df = pd.read_csv(META_EXTERNAL, sep='\t')
                # Create Dictionary: {'sub-001': 25.0, '001': 25.0}
                age_map = dict(zip(meta_df['participant_id'], meta_df['age']))
                
                def get_age(sub_id):
                    sid = str(sub_id).strip()
                    # Ensure format matches (handle 'sub-X' vs 'X')
                    if not sid.startswith('sub-'): 
                        return age_map.get(f"sub-{sid}", age_map.get(sid, np.nan))
                    return age_map.get(sid, np.nan)
                
                df['Age'] = df['Subject'].apply(get_age)
                print("   ✅ Ages successfully mapped.")
            except Exception as e:
                print(f"   ❌ Error loading metadata: {e}")
    
    # --- 2. SELECT METADATA ---
    # Keep relevant metadata columns
    desired_meta = ['Subject', 'Condition', 'Age', 'Gender', 'Group', 'Group_Detailed', 'Dataset_Source', 'Label']
    existing_meta = [c for c in desired_meta if c in df.columns]
    val_df = df[existing_meta].copy()
    
    # --- 3. CALCULATE BANDS (GLOBAL AVERAGE) ---
    
    bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
    found_any = False
    
    print("⚙️  Averaging bands across all channels...")
    for band in bands:
        # Find columns containing the band name (e.g., 'Fp1_Alpha') but exclude existing 'Global' columns
        cols = [c for c in df.columns if band in c and 'Global' not in c]
        
        if len(cols) > 0:
            # Calculate row-wise mean (axis=1)
            val_df[f'Global_{band}'] = df[cols].mean(axis=1)
            found_any = True
        else:
            print(f"   ⚠️ No channels found for band: {band}")

    # --- 4. SAVE ---
    if found_any:
        val_df.to_csv(OUTPUT_FILE, index=False)
        print("-" * 50)
        print(f"✅ SUCCESS! File saved: {OUTPUT_FILE}")
        print("   You can now run 'src/Visualizations_ML/validate_physiology.py'.")
    else:
        print("❌ No frequency data found to average. Check your input file.")

if __name__ == "__main__":
    calculate_global_bands()