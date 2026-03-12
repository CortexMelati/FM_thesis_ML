"""
=============================================================================
🧠 THESIS EEG PIPELINE: FINAL DATA MERGE
=============================================================================
Objective:
    Consolidate processed EEG features and participant metadata into a single
    MASTER dataset for statistical analysis and machine learning.

Data Sources:
1.  METADATA (Participant Info):
    -   Retrieves Age, Gender, and Diagnosis from Excel (TDBrain) and TSV (Chronic Pain).
2.  FEATURES (EEG Metrics):
    -   Scans the 'results' directory for all '*_features.csv' files generated
        by the preprocessing pipeline.

Inclusion Criteria:
1.  AGE: Minimum 18 years.
2.  COMPLETENESS: Subject MUST have data for BOTH Eyes Closed (EC) and Eyes Open (EO).
3.  EPOCHS: Subject MUST have exactly 10 epochs available per condition (0-11). Amend REQUIRED_EPOCHS as needed.

Labels (Y-variable):
     0 = Healthy
     1 = Chronic Pain (TDBrain + External)
    -1 = Unknown (No Indication / NaN) -> SEPARATE LABEL

Output:
    -   'results/final_dataset.csv': A comprehensive CSV where each row represents
        one epoch, enriched with subject metadata.

Execution:
    python ./FM_thesis_ML/src/Preprocessing/final_prep.py
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import (
    RESULTS_DIR, 
    TDBRAIN_DIR, 
    CHRONIC_PAIN_DIR,
    MIN_AGE
)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Define Paths using Config
OUTPUT_FILE = RESULTS_DIR / "final_dataset.csv"
TSV_FILE = CHRONIC_PAIN_DIR / "participants.tsv"

# Metadata Sources & Label Definitions
# Note: Using Path objects (/) instead of os.path.join
META_CONFIG = {
    'Healthy': (
        TDBRAIN_DIR / "TDBRAIN_participants_HEALTHY.xlsx", 
        0, 
        "TDBrain_Healthy"
    ),
    'Pain': (
        TDBRAIN_DIR / "TDBRAIN_participants_CHRONIC_PAIN.xlsx", 
        1, 
        "TDBrain_ChronicPain"
    ),
    # Separate labels for Unknown categories (-1 and -2)
    # 'Unknown_Informal': (
    #     TDBRAIN_DIR / "TDBRAIN_participants_UNKNOWN.xlsx", 
    #     -1, 
    #     "TDBrain_Unknown_Informal"
    # ),
    'Unknown_NaN': (
        TDBRAIN_DIR / "TDBRAIN_participants_UNKNOWN_NaNs.xlsx", 
        -1, 
        "TDBrain_Unknown_NoIndication"
    )
}

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def normalize_gender(val):
    """Standardizes gender strings to numeric format (0=F, 1=M)."""
    s = str(val).lower().strip()
    if s.endswith('.0'): s = s[:-2]
    if s in ['0', 'f', 'female', 'vrouw', 'woman', '2']: return 0  # Female
    if s in ['1', 'm', 'male', 'man']: return 1    # Male
    return np.nan

def load_metadata_map():
    """
    Builds a dictionary of ALL eligible subjects (18+) with their specific group labels.
    Reads from both TDBrain Excel files and External TSV files.
    """
    print("📖 Building Metadata Index (Filter: 18+ & Labels)...")
    meta = {}
    
    # 1. Process TDBRAIN Excel Files
    for key, (path, label, detailed_name) in META_CONFIG.items():
        if os.path.exists(path):
            df = pd.read_excel(path)
            # Dynamic column search for flexibility
            id_col = next((c for c in df.columns if 'ID' in c or 'sub' in c), 'participants_ID')
            age_col = next((c for c in df.columns if 'age' in c.lower()), 'age')
            sex_col = next((c for c in df.columns if 'gender' in c.lower() or 'sex' in c.lower()), 'gender')

            count = 0
            for _, row in df.iterrows():
                try:
                    age = float(row[age_col])
                    if age >= MIN_AGE: 
                        sub_id = str(row[id_col]).strip()
                        meta[sub_id] = {
                            'Age': age,
                            'Gender': normalize_gender(row[sex_col]),
                            'Label': label, 
                            'Group_Detailed': detailed_name,
                            'Dataset': 'TDBRAIN'
                        }
                        count += 1
                except: continue
            print(f"   -> {key}: Loaded {count} subjects (Label: {label}).")

    # 2. Process External CP (TSV)
    if os.path.exists(TSV_FILE):
        df = pd.read_csv(TSV_FILE, sep='\t')
        count = 0
        for _, row in df.iterrows():
            try:
                age = float(row['age'])
                if age >= MIN_AGE:
                    sub_id = str(row['participant_id']).strip()
                    meta[sub_id] = {
                        'Age': age,
                        'Gender': normalize_gender(row['sex']),
                        'Label': 1, 
                        'Group_Detailed': "External_CP",
                        'Dataset': 'EXTERNAL'
                    }
                    count += 1
            except: continue
        print(f"   -> External CP: Loaded {count} subjects (Label: 1).")

    return meta

def get_valid_subjects():
    """
    Scans the results directory and groups files by Subject ID.
    Returns a dictionary ONLY for subjects who have BOTH Eyes Closed (EC) and Eyes Open (EO).
    """
    print(f"\n📂 Scanning for Completeness (EC + EO Required)...")
    
    # Use config RESULTS_DIR, convert to string for glob compatibility
    search_path = os.path.join(str(RESULTS_DIR), "**", "*_features.csv")
    all_files = glob.glob(search_path, recursive=True)
    
    # Group files by Subject ID
    subject_files = {}
    for f in all_files:
        filename = os.path.basename(f)
        sub_id = filename.split('_')[0]
        if sub_id not in subject_files:
            subject_files[sub_id] = []
        subject_files[sub_id].append(f)
        
    valid_subjects = {}
    incomplete_count = 0
    
    for sub_id, files in subject_files.items():
        # Check if both conditions exist in the file list
        has_ec = any('EC' in os.path.basename(f) for f in files)
        has_eo = any('EO' in os.path.basename(f) for f in files)
        
        if has_ec and has_eo:
            valid_subjects[sub_id] = files
        else:
            incomplete_count += 1

    print(f"   ✅ Complete Subjects (EC+EO): {len(valid_subjects)}")
    print(f"   ✂️  Removed Incomplete Subjects: {incomplete_count}")
    
    return valid_subjects

def create_dataset():
    # Step 1: Load Metadata
    meta_map = load_metadata_map()
    
    # Step 2: Retrieve Valid Subjects (Must have both EC and EO files)
    valid_subjects = get_valid_subjects()
    
    all_data = []
    skipped_age_count = 0
    skipped_epoch_count = 0 
    
    # Minimum epochs required per condition (0-11 means 12 total, but we will truncate to 10 for consistency)
    REQUIRED_EPOCHS = 9 
    
    # Step 3: Merge Metadata and Features
    for sub_id, files in tqdm(valid_subjects.items(), desc="Merging Data"):
        
        # Filter 1: Check against metadata map (Effectively filters for Age >= 18)
        if sub_id not in meta_map:
            skipped_age_count += 1
            continue
            
        subject_dfs = []
        valid_epoch_count = True
        
        # Process both EC and EO files for this subject
        for f in files:
            try:
                df = pd.read_csv(f)
                
                # Filter 2: Minimum X epochs needed after autoreject
                if len(df) < REQUIRED_EPOCHS:
                    valid_epoch_count = False
                    break  # Instantly fail this subject
                
                # Truncate to EXACTLY the required amount to ensure equal voting power
                df = df.head(REQUIRED_EPOCHS)
                
                info = meta_map[sub_id]
                
                # Append Metadata
                df['Age'] = info['Age']
                df['Gender'] = info['Gender']
                df['Label'] = info['Label']
                df['Group_Detailed'] = info['Group_Detailed']
                df['Dataset'] = info['Dataset']
                
                # Tag Condition
                filename = os.path.basename(f)
                if 'EC' in filename: df['Condition'] = 'EC'
                elif 'EO' in filename: df['Condition'] = 'EO'
                else: df['Condition'] = 'UNK'
                
                subject_dfs.append(df)
            except Exception as e:
                print(f"\nError reading {f}: {e}")
                valid_epoch_count = False
                break
                
        # If the subject passed the check for BOTH files, add them
        if valid_epoch_count and len(subject_dfs) >= 2:
            all_data.extend(subject_dfs)
        elif not valid_epoch_count:
            skipped_epoch_count += 1

    # Step 4: Save Final Dataset
    if all_data:
        print("\n💾 Concatenating DataFrame...")
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        
        print("\n" + "="*60)
        print("✅ FINAL DATASET CREATED SUCCESSFULLY")
        print(f"📁 Location: {OUTPUT_FILE}")
        print("-" * 60)
        print(f"✂️  Excluded (Age < 18 or No Metadata): {skipped_age_count}")
        print(f"✂️  Excluded (< {REQUIRED_EPOCHS} clean epochs):      {skipped_epoch_count}")
        print("-" * 60)
        print(f"🔢 Total Epochs in Dataset:   {len(final_df)}")
        print(f"👤 Unique Subjects Included:  {final_df['Subject'].nunique()}")
        print("-" * 60)
        print("📊 DISTRIBUTION PER GROUP (Unique Subjects):")
        counts = final_df.groupby(['Group_Detailed', 'Label'])['Subject'].nunique()
        print(counts)
        print("="*60)
    else:
        print("\n❌ Error: No valid data found to merge!")

if __name__ == "__main__":
    create_dataset()