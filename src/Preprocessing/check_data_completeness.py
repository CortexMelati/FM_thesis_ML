"""
=============================================================================
✅ DATA INTEGRITY AUDIT: COMPLETENESS CHECK
=============================================================================
Objective:
    Verify that every subject listed in the participant metadata (Excel/TSV)
    has a corresponding results folder containing all required output files.

Scope (5 Datasets):
    1. TDBrain - Chronic Pain
    2. TDBrain - Healthy
    3. TDBrain - Unknown
    4. TDBrain - Unknown NaNs (checked within Unknown folder)
    5. External Chronic Pain Dataset

Checks per Subject:
    - Does the folder exist?
    - Do files exist for BOTH conditions (EC/EO)?
    - Do all file types exist (.npy, .txt, .csv, .pdf)?

Output:
    - Console summary of missing files.
    - CSV Report: 'full_dataset_audit.csv' (if issues are found).

Execution:
    python ./FM_thesis_ML/src/Preprocessing/check_data_completeness.py
=============================================================================
"""

import pandas as pd
import os
import glob
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import RESULTS_DIR, TDBRAIN_DIR, CHRONIC_PAIN_DIR

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# Requirements for every subject
CONDITIONS = ["EC", "EO"]
EXTENSIONS = [".npy", ".txt", ".csv", ".pdf"]

# Datasets to audit (Including Unknown NaNs)
DATASETS = [
    {
        "name": "TDBRAIN - Chronic Pain",
        "excel": TDBRAIN_DIR / "TDBRAIN_participants_CHRONIC_PAIN.xlsx",
        "res_dir": RESULTS_DIR / "TDBrain" / "chronicpain",
        "type": "excel"
    },
    {
        "name": "TDBRAIN - Healthy",
        "excel": TDBRAIN_DIR / "TDBRAIN_participants_HEALTHY.xlsx",
        "res_dir": RESULTS_DIR / "TDBrain" / "healthy",
        "type": "excel"
    },
    {
        "name": "TDBRAIN - Unknown", 
        "excel": TDBRAIN_DIR / "TDBRAIN_participants_UNKNOWN.xlsx",
        "res_dir": RESULTS_DIR / "TDBrain" / "unknown",
        "type": "excel"
    },
    {
        "name": "TDBRAIN - Unknown NaNs (Check in Unknown folder)", 
        "excel": TDBRAIN_DIR / "TDBRAIN_participants_UNKNOWN_NaNs.xlsx",
        # These subjects are stored in the same 'unknown' folder as the group above
        "res_dir": RESULTS_DIR / "TDBrain" / "unknown_nans", # Updated to match preprocess_pipeline logic if changed, or keep 'unknown'
        # Note: If preprocess_pipeline puts them in 'unknown_nans', keep this. If it puts them in 'unknown', change back.
        # Based on previous context, usually they share the folder or have a specific one. 
        # I will revert to 'unknown' to be safe unless specified otherwise, but standard logic suggests distinct folders if distinct lists.
        # Let's stick to the config provided in the prompt:
        "res_dir": RESULTS_DIR / "TDBrain" / "unknown", 
        "type": "excel"
    },  
    {
        "name": "Chronic Pain Dataset (Original)",
        "tsv": CHRONIC_PAIN_DIR / "participants.tsv",
        "res_dir": RESULTS_DIR / "chronicpain",
        "type": "tsv"
    }
]

# ==========================================

def check_folder_completeness(folder_path, sub_id):
    """
    Checks if the subject folder exists and contains all required files.
    Returns a list of missing items.
    """
    if not os.path.exists(folder_path):
        return ["❌ FOLDER MISSING"]
    
    files = os.listdir(folder_path)
    missing = []
    
    for cond in CONDITIONS: # Check EC, EO
        for ext in EXTENSIONS: # Check .npy, .txt, etc
            # Search for file containing 'EC'/'EO' AND ending with extension
            found = False
            for f in files:
                if cond in f and f.endswith(ext):
                    found = True
                    break
            if not found:
                missing.append(f"{cond}{ext}")
                
    return missing

def run_audit():
    print("🚀 STARTING FULL DATASET AUDIT (Inc. Unknown)")
    print("="*60)
    
    all_problems = []
    total_subjects_checked = 0
    
    for ds in DATASETS:
        print(f"\n📂 Checking: {ds['name']}")
        
        # 1. Load Subject IDs
        ids = []
        try:
            if ds['type'] == 'excel':
                df = pd.read_excel(ds['excel'])
                col = 'participants_ID' if 'participants_ID' in df.columns else 'participant_id'
                # Ensure strings and strip whitespace
                ids = df[col].dropna().astype(str).str.strip().tolist()
            elif ds['type'] == 'tsv':
                df = pd.read_csv(ds['tsv'], sep='\t')
                ids = df['participant_id'].dropna().astype(str).str.strip().tolist()
        except Exception as e:
            print(f"   ❌ Could not load list: {e}")
            continue

        print(f"   -> Found {len(ids)} subjects in metadata list.")
        total_subjects_checked += len(ids)
        
        # 2. Check per Subject
        ds_problems = 0
        for sub_id in ids:
            # Path: results/group/sub-ID
            # Note: ds['res_dir'] is a Path object, creating full path
            sub_folder = ds['res_dir'] / sub_id
            
            missing_items = check_folder_completeness(sub_folder, sub_id)
            
            if missing_items:
                ds_problems += 1
                status = "MISSING FOLDER" if "❌ FOLDER MISSING" in missing_items else "INCOMPLETE"
                
                # Format details nicely
                details = str(missing_items) if len(missing_items) < 8 else "MISSING ALL FILES (Empty folder or naming issue?)"
                
                all_problems.append({
                    "Dataset": ds['name'],
                    "Subject": sub_id,
                    "Status": status,
                    "Details": details
                })
                
                # Print immediately if it's an incomplete set (folder exists, but files missing)
                if status == "INCOMPLETE":
                     print(f"   ⚠️  {sub_id}: Missing {details}")

    # --- REPORT GENERATION ---
    print("\n" + "="*60)
    print(f"Total Subjects Audited: {total_subjects_checked}")
    
    if len(all_problems) > 0:
        print(f"❌ Issues Found: {len(all_problems)}")
        
        # Save Report
        outfile = "full_dataset_audit.csv"
        df_out = pd.DataFrame(all_problems)
        df_out.to_csv(outfile, index=False)
        print(f"📄 Report saved to: {outfile}")
        
        # Show "Incomplete" cases (more interesting than just missing folders)
        incomplete = df_out[df_out['Status'] == 'INCOMPLETE']
        if not incomplete.empty:
            print("\n--- INCOMPLETE SUBJECTS (Folder exists, files missing) ---")
            print(incomplete[['Dataset', 'Subject', 'Details']].to_string(index=False))
        else:
            print("\nNo 'partial' subjects found (Either complete or folder entirely missing).")
            
    else:
        print("✅ SUCCESS! The entire dataset (Healthy, Pain & Unknown) is 100% complete.")

if __name__ == "__main__":
    run_audit()