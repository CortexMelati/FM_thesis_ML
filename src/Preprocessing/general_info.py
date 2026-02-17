"""
=============================================================================
📊 GENERATE TABLE 1: DEMOGRAPHICS & CLINICAL CHARACTERISTICS
=============================================================================
Objective:
    Generate the demographic summary table for the thesis (Table 1).
    This script applies STRICT filtering to ensure the reported N matches
    the final dataset exactly.

Filtering Criteria:
    1. UNIQUENESS: Remove duplicate IDs immediately.
    2. AGE: Exclude subjects < 18.0 years.
    3. COMPLETENESS: Subjects MUST have processed 'EC' and 'EO' feature files 
       physically present in the 'results' folder.

Input:
    - Raw Metadata (Excel/TSV)
    - Processed Results (Verification of existence)

Output:
    - Console Table showing N, Age (Mean ± SD), Gender distribution, and Diagnosis status.

Execution:
    python ./FM_thesis_ML/src/Preprocessing/general_info.py
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

from config import (
    RESULTS_DIR, 
    TDBRAIN_DIR, 
    CHRONIC_PAIN_DIR, 
    MIN_AGE
)

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Data Paths (Using Config Variables)
META_DIR = TDBRAIN_DIR
CP_EXT_DIR = CHRONIC_PAIN_DIR

PATH_HEALTHY      = META_DIR / "TDBRAIN_participants_HEALTHY.xlsx"
PATH_PAIN         = META_DIR / "TDBRAIN_participants_CHRONIC_PAIN.xlsx"
PATH_UNKNOWN      = META_DIR / "TDBRAIN_participants_UNKNOWN.xlsx"
PATH_UNKNOWN_NANS = META_DIR / "TDBRAIN_participants_UNKNOWN_NaNs.xlsx"

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def normalize_gender(val):
    """Standardize gender to 'F' or 'M'."""
    s = str(val).lower().strip()
    if s.endswith('.0'): s = s[:-2]
    if s in ['0', 'f', 'female', 'vrouw', 'woman', '2']: return 'F'
    if s in ['1', 'm', 'male', 'man']: return 'M'
    return np.nan

def check_files_exist(subject_id, relative_folder):
    """
    Verifies if the processed EC and EO feature files actually exist on disk.
    This ensures we only count subjects that successfully passed the preprocessing pipeline.
    """
    # Using Path objects for robust joining
    base_path = RESULTS_DIR / relative_folder / subject_id
    
    ec_file = base_path / f"{subject_id}_EC_features.csv"
    eo_file = base_path / f"{subject_id}_EO_features.csv"
    
    return ec_file.exists() and eo_file.exists()

def load_tdbrain_group(filepath, group_name, formal_status, results_subfolder):
    """
    Loads TDBrain metadata, deduplicates, and filters for file existence.
    """
    if not filepath.exists(): return pd.DataFrame()
    df = pd.read_excel(filepath)
    
    # Dynamic column finding
    id_col = next((c for c in df.columns if 'ID' in c or 'sub' in c), 'participants_ID')
    age_col = next((c for c in df.columns if 'age' in c.lower()), None)
    sex_col = next((c for c in df.columns if 'gender' in c.lower() or 'sex' in c.lower()), None)
    
    if not age_col or not sex_col: return pd.DataFrame()
    
    data = pd.DataFrame()
    data['ID'] = df[id_col].astype(str).str.strip()
    data['Age'] = pd.to_numeric(df[age_col], errors='coerce')
    data['Gender'] = df[sex_col].apply(normalize_gender)
    data['Group'] = group_name
    data['Formal_Diag'] = formal_status
    
    # 1. DEDUPLICATE (Critical step!)
    data = data.drop_duplicates(subset='ID')
    
    # 2. FILE COMPLETENESS CHECK
    data['Complete'] = data['ID'].apply(lambda x: check_files_exist(x, results_subfolder))
    data = data[data['Complete'] == True].copy()
    
    return data

def load_external_cp(data_dir):
    """
    Loads External Chronic Pain metadata (TSV), deduplicates, and filters.
    """
    tsv_path = data_dir / "participants.tsv"
    
    if tsv_path.exists():
        df = pd.read_csv(tsv_path, sep='\t')
        age_col = next((c for c in df.columns if 'age' in c.lower()), 'age')
        sex_col = next((c for c in df.columns if 'sex' in c.lower()), 'sex')
        
        data = pd.DataFrame()
        data['ID'] = df['participant_id'].astype(str).str.strip()
        data['Age'] = pd.to_numeric(df[age_col], errors='coerce')
        data['Gender'] = df[sex_col].apply(normalize_gender)
        data['Group'] = 'External CP'
        data['Formal_Diag'] = 'Yes'
        
        # Deduplicate and Check Files
        data = data.drop_duplicates(subset='ID')
        # Note: External CP results are stored in 'chronicpain' folder
        data['Complete'] = data['ID'].apply(lambda x: check_files_exist(x, "chronicpain"))
        data = data[data['Complete'] == True].copy()
        
        return data
    return pd.DataFrame()

# =============================================================================
# 3. DATA COLLECTION
# =============================================================================

print(f"📊 Calculating exact statistics (Unique, 18+, Complete)...")

# Load Groups (Including deduplication and file check)
# TDBrain Healthy results are in 'TDBrain/healthy'
df_healthy = load_tdbrain_group(PATH_HEALTHY, "TDBRAIN Healthy (A)", "Yes", os.path.join("TDBrain", "healthy"))

# TDBrain Pain results are in 'TDBrain/chronicpain'
df_pain    = load_tdbrain_group(PATH_PAIN, "TDBRAIN Chronic Pain (B)", "Yes", os.path.join("TDBrain", "chronicpain"))

# TDBrain Unknown/NaNs are in 'TDBrain/unknown'
df_unk     = load_tdbrain_group(PATH_UNKNOWN, "TDBRAIN Informal Indication (C)", "No", os.path.join("TDBrain", "unknown"))
df_nans    = load_tdbrain_group(PATH_UNKNOWN_NANS, "TDBRAIN No indication (D)", "No", os.path.join("TDBrain", "unknown"))

# External CP
df_ext_cp  = load_external_cp(CP_EXT_DIR)

# Merge All
df_all = pd.concat([df_ext_cp, df_healthy, df_pain, df_unk, df_nans], ignore_index=True)

# Apply Age Filter (>= 18.0)
df_filtered = df_all[df_all['Age'] >= MIN_AGE].copy()

# =============================================================================
# 4. GENERATE TABLE OUTPUT
# =============================================================================

summary = df_filtered.groupby('Group', sort=False).agg(
    N=('Age', 'count'), 
    Age_Mean=('Age', 'mean'),
    Age_SD=('Age', 'std'),
    Gender_Total=('Gender', 'count'),
    Female_Count=('Gender', lambda x: (x == 'F').sum()),
    Formally_Diagnosed=('Formal_Diag', 'first')
).reset_index()

# Format Columns for Display
summary['Age (Mean ± SD)'] = summary.apply(lambda row: f"{row['Age_Mean']:.1f} ± {row['Age_SD']:.1f}", axis=1)
summary['Gender (% Female)'] = summary.apply(
    lambda row: f"{(row['Female_Count'] / row['Gender_Total'] * 100):.1f}%" if row['Gender_Total'] > 0 else "0%", axis=1
)

final_table = summary[['Group', 'N', 'Age (Mean ± SD)', 'Gender (% Female)', 'Formally_Diagnosed']]
final_table.columns = ['Diagnosis / Sample', 'N', 'Age (Mean ± SD)', 'Gender (% Female)', 'Diagnosed']

print("\n" + "="*90)
print(final_table.to_string(index=False))
print("="*90)