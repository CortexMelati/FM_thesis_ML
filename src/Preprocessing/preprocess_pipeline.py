"""
=============================================================================
🧠 EEG PREPROCESSING PIPELINE
=============================================================================
Overview:
    This script performs the standard preprocessing pipeline for the Thesis EEG data.
    It transforms raw EEG files (.vhdr) into cleaned epochs and extracted features.

Key Steps:
    1. SCALING:    Detects if data is in Volts and rescales to Microvolts (uV).
    2. BASELINE:   Removes DC offset (centers data around 0).
    3. RENAMING:   Standardizes channel names to the 10-20 system.
    4. RANSAC:     Robustly detects and interpolates bad channels.
    5. AUTOREJECT: Cleans data by interpolating bad segments instead of discarding full subjects.
    6. SPLITTING:  Separates output into Eyes Closed (EC) and Eyes Open (EO) conditions.

Outputs:
    1. .npy files (3D Cleaned Data) -> For Deep Learning / Advanced Analysis.
    2. .csv files (2D Features)     -> For Classic ML (Random Forest, SVM).
    3. .pdf files (Reports)         -> For Visual Inspection (Quality Control).
    4. .txt files (Logs)            -> Statistics on rejected data/channels.

Execution:
    python ./FM_thesis_ML/src/Preprocessing/preprocess_pipeline.py
=============================================================================
"""

import mne
import numpy as np
import pandas as pd
import os
import glob
from autoreject import AutoReject, Ransac
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
import warnings
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
    CHANNELS, 
    BANDS, 
    SFREQ, 
    EPOCH_LENGTH
)

# Import custom plotting script (assumed to be in same folder)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from preprocessing_plotting import get_plots
except ImportError:
    print("⚠️  Warning: 'preprocessing_plotting.py' not found. Plots will be skipped.")
    def get_plots(*args, **kwargs): return None

# =============================================================================
# 1. CONFIGURATION & PATHS
# =============================================================================

# Directory containing TDBrain Excel metadata lists
META_DIR = TDBRAIN_DIR

# Paths to the 4 specific subject lists (Using Path objects)
PATH_HEALTHY      = META_DIR / "TDBRAIN_participants_HEALTHY.xlsx"
PATH_PAIN         = META_DIR / "TDBRAIN_participants_CHRONIC_PAIN.xlsx"
PATH_UNKNOWN      = META_DIR / "TDBRAIN_participants_UNKNOWN.xlsx"
PATH_UNKNOWN_NANS = META_DIR / "TDBRAIN_participants_UNKNOWN_NaNs.xlsx"

# Output Directory
OUTPUT_DIR = RESULTS_DIR

# --- DATASETS CONFIGURATION ---
# Convert Paths to strings for glob compatibility
DATASETS = [
    (str(CHRONIC_PAIN_DIR / "derivatives"), "*.vhdr", "chronicpain_set"),
    (str(TDBRAIN_DIR / "derivatives"), "*.vhdr", "tdbrain")
]

# EEG Parameters (Loaded from Config)
COMMON_CHANNELS = CHANNELS
FREQ_BANDS = BANDS
# SFREQ and EPOCH_LENGTH are imported directly
CROP_TMAX = 119.4 # Specific crop for this pipeline

warnings.filterwarnings("ignore") 

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def load_ids_from_excel(filepath):
    """Loads a list of subject IDs from an Excel file."""
    if not os.path.exists(filepath):
        print(f"⚠️  NOTE: Metadata file not found: {filepath}")
        return []
    try:
        df = pd.read_excel(filepath)
        # flexible search for 'ID' column
        id_col = next((c for c in df.columns if 'ID' in c or 'sub' in c), None)
        if id_col:
            # Clean: remove spaces, convert to strings
            return [str(x).strip() for x in df[id_col].dropna().unique()]
        else:
            print(f"⚠️  No ID column found in {os.path.basename(filepath)}")
            return []
    except Exception as e:
        print(f"❌ Error reading {filepath}: {e}")
        return []

print("📂 Loading subject lists from Excel...")
TDBRAIN_HEALTHY_IDS = load_ids_from_excel(PATH_HEALTHY)
TDBRAIN_CHRONIC_PAIN_IDS = load_ids_from_excel(PATH_PAIN)
TDBRAIN_UNKNOWN_IDS = load_ids_from_excel(PATH_UNKNOWN)
TDBRAIN_UNKNOWN_NANS_IDS = load_ids_from_excel(PATH_UNKNOWN_NANS)

print(f"   -> {len(TDBRAIN_HEALTHY_IDS)} Healthy subjects.")
print(f"   -> {len(TDBRAIN_CHRONIC_PAIN_IDS)} Chronic Pain subjects.")
print(f"   -> {len(TDBRAIN_UNKNOWN_IDS)} Unknown Status subjects.")
print(f"   -> {len(TDBRAIN_UNKNOWN_NANS_IDS)} Unknown/NaN Indication subjects.")

def get_condition(filename):
    """Determines if the file is Eyes Closed (EC) or Eyes Open (EO)."""
    fname_upper = filename.upper()
    if 'EC' in fname_upper or 'CLOSED' in fname_upper: return 'EC'
    elif 'EO' in fname_upper or 'OPEN' in fname_upper: return 'EO'
    else: return 'unknown'

def smart_rename_channels(raw):
    """Renames channels to standard 10-20 nomenclature."""
    current_names = raw.ch_names
    mapping = {}
    target_map = {ch.lower(): ch for ch in COMMON_CHANNELS}
    synonyms = {
        'fp1': 'Fp1', 'fp2': 'Fp2', 'fpz': 'Fpz', 'f3': 'F3', 'f4': 'F4', 'fz': 'Fz', 'f7': 'F7', 'f8': 'F8',
        't3': 'T7', 't4': 'T8', 't7': 'T7', 't8': 'T8', 'c3': 'C3', 'c4': 'C4', 'cz': 'Cz',
        't5': 'P7', 't6': 'P8', 'p7': 'P7', 'p8': 'P8', 'p3': 'P3', 'p4': 'P4', 'pz': 'Pz',
        'o1': 'O1', 'o2': 'O2', 'oz': 'Oz'
    }
    for ch in current_names:
        clean_ch = ch.replace('EEG', '').replace('Ref', '').replace(' ', '').replace('-', '').replace('.', '').lower()
        if clean_ch in synonyms:
            std = synonyms[clean_ch]
            if std in COMMON_CHANNELS and ch != std: mapping[ch] = std
        elif clean_ch in target_map:
            std = target_map[clean_ch]
            if ch != std: mapping[ch] = std
    if mapping:
        try: raw.rename_channels(mapping)
        except: pass 
    return raw

def fix_scaling_and_units(raw):
    """Detects if data is in Volts (instead of uV) and rescales."""
    data_sample = raw.get_data(start=0, stop=int(10*raw.info['sfreq']))
    mean_amp = np.mean(np.abs(data_sample))
    if mean_amp > 0.1: 
        raw.apply_function(lambda x: x * 1e-6, channel_wise=True)
    raw.apply_function(lambda x: x - np.mean(x), channel_wise=True)
    return raw

def load_raw_data(filepath):
    """Loads .vhdr files, sets channel types, and standardizes names/units."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.vhdr':
        raw = mne.io.read_raw_brainvision(filepath, misc='auto', preload=True, verbose=False)
    else:
        raise ValueError(f"Unknown format: {ext}")
    
    try: raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names if ch not in ['ECG', 'EOG']})
    except: pass
    
    raw = smart_rename_channels(raw)
    raw = fix_scaling_and_units(raw)
    return raw

def extract_features(epochs, subject_id, condition):
    """Computes Relative Power Spectral Density (PSD) using Welch's method."""
    
    
    psd = epochs.compute_psd(method='welch', fmin=0.5, fmax=100, n_fft=2048, verbose=False)
    data, freqs = psd.get_data(return_freqs=True)
    features = []
    for epoch_idx in range(len(data)):
        epoch_feats = {'Subject': subject_id, 'Condition': condition, 'Epoch': epoch_idx}
        for ch_idx, ch_name in enumerate(epochs.ch_names):
            total_power = 0
            for band, (fmin, fmax) in FREQ_BANDS.items():
                mask = (freqs >= fmin) & (freqs <= fmax)
                total_power += np.mean(data[epoch_idx, ch_idx, mask])
            
            for band, (fmin, fmax) in FREQ_BANDS.items():
                mask = (freqs >= fmin) & (freqs <= fmax)
                abs_power = np.mean(data[epoch_idx, ch_idx, mask])
                rel_power = abs_power / total_power if total_power > 0 else 0
                epoch_feats[f"{ch_name}_{band}"] = rel_power
        features.append(epoch_feats)
    return pd.DataFrame(features)

def process_subject(file_path, output_dir, dataset_name):
    """Main processing logic per subject."""
    filename = os.path.basename(file_path)
    subject_id = filename.split('_')[0] 
    condition = get_condition(filename)
    
    # --- Folder Structure & Filtering ---
    if dataset_name == 'chronicpain_set':
        # External dataset is always processed
        sub_folder = 'chronicpain'
    elif dataset_name == 'tdbrain':
        # STRICT Check for TDBrain IDs
        if subject_id in TDBRAIN_HEALTHY_IDS:
            sub_folder = os.path.join('TDBrain', 'healthy')
        elif subject_id in TDBRAIN_CHRONIC_PAIN_IDS:
            sub_folder = os.path.join('TDBrain', 'chronicpain')
        elif subject_id in TDBRAIN_UNKNOWN_IDS:
            sub_folder = os.path.join('TDBrain', 'unknown')
        elif subject_id in TDBRAIN_UNKNOWN_NANS_IDS:
            sub_folder = os.path.join('TDBrain', 'unknown_nans')
        else:
            # Not in any of the 4 lists -> SKIP
            return False, f"Skipped (ID {subject_id} not in selected groups)"
    else:
        sub_folder = 'unknown_dataset'

    save_dir = os.path.join(output_dir, sub_folder, subject_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if already processed
    csv_check = os.path.join(save_dir, f"{subject_id}_{condition}_features.csv")
    if os.path.exists(csv_check):
       return True, f"Skipped (Already exists)"

    try:
        # 1. LOAD DATA
        raw = load_raw_data(file_path)

        # 2. CHANNEL SELECTION
        available = raw.ch_names
        missing = [ch for ch in COMMON_CHANNELS if ch not in available]
        if missing: return False, f"Missing channels: {missing}"
        
        raw.pick_channels(COMMON_CHANNELS)
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        
        # Average Reference (Important for Connectivity/wPLI)
        raw.set_eeg_reference('average', projection=False, verbose=False)

        # 3. CROPPING & FILTERING
        if raw.times[-1] < CROP_TMAX: return False, "Too short"
        raw.crop(tmin=0, tmax=CROP_TMAX)
        raw.notch_filter(np.arange(50, 101, 50), verbose=False) 
        raw.filter(l_freq=0.5, h_freq=100, verbose=False)
        if raw.info['sfreq'] != SFREQ: raw.resample(SFREQ, verbose=False)

        try:
            fig_before = get_plots(raw, step=f"1. Raw (Fixed)", scalings={'eeg': 100e-6}, channel_idx=[9])
        except: fig_before = None

        # 4. EPOCHING
        epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_LENGTH, overlap=0, preload=True, verbose=False)
        if len(epochs) < 10: return False, "Too few epochs"

        # 5A. RANSAC (Bad Channel Detection & Repair)
        rsc = Ransac()
        epochs_ransac = rsc.fit_transform(epochs)
        
        bad_channels = rsc.bad_chs_ if hasattr(rsc, 'bad_chs_') else []
        n_bad_chs = len(bad_channels)

        # 5B. AUTOREJECT (Epoch Cleaning)
        ar = AutoReject()
        cleaned_epochs = ar.fit_transform(epochs_ransac)

        # 6. FEATURE EXTRACTION
        df_features = extract_features(cleaned_epochs, subject_id, condition)
        df_features.to_csv(csv_check, index=False)

        # 7. REPORTING
        try:
            evoked = cleaned_epochs.average()
            fig_erp = evoked.plot(gfp=True, spatial_colors=True, show=False, title=f"ERP {condition}")
        except: fig_erp = None
        
        try:
            data_tmp = cleaned_epochs.get_data(copy=True)
            raw_cln = mne.io.RawArray(np.hstack(data_tmp), cleaned_epochs.info, verbose=False)
            fig_after = get_plots(raw_cln, step="2. Cleaned (RANSAC+AR)", scalings={'eeg': 40e-6}, channel_idx=[9])
        except: fig_after = None

        with PdfPages(os.path.join(save_dir, f"{subject_id}_{condition}_report.pdf")) as pdf:
            if fig_before: pdf.savefig(fig_before)
            if fig_after: pdf.savefig(fig_after)
            if fig_erp: pdf.savefig(fig_erp)
        plt.close('all')

        # 8. CLEAN DATA SAVE
        np.save(os.path.join(save_dir, f"{subject_id}_{condition}_cleaned.npy"), cleaned_epochs.get_data(copy=True))

        # 9. LOGGING
        reject_log = ar.get_reject_log(epochs_ransac)
        n_total = len(epochs)
        n_kept = len(cleaned_epochs)
        log_lines = [
            f"Cleaning Report for {subject_id} ({condition})",
            "="*30,
            f"RANSAC Bad Channels:       {bad_channels}",
            "-"*30,
            f"AutoReject Total Epochs:   {n_total}",
            f"AutoReject Kept Epochs:    {n_kept}",
            f"AutoReject Dropped:        {n_total - n_kept}"
        ]
        with open(os.path.join(save_dir, f"{subject_id}_{condition}_cleaning_log.txt"), 'w') as f:
            f.write("\n".join(log_lines))

        return True, f"OK (RANSAC fix: {n_bad_chs} chs, AR kept: {n_kept} eps)"

    except Exception as e:
        plt.close('all')
        return False, str(e)

# --- MAIN LOOP ---
if __name__ == "__main__":
    print(f"🚀 Starting FINAL Preprocessing Pipeline")
    
    all_files = []
    for folder, pattern, ds_name in DATASETS:
        print(f"📂 Scanning {folder} for {pattern}...")
        found = glob.glob(os.path.join(folder, "**", pattern), recursive=True)
        # Filter files that are already processed (skip 'clean' or 'results' folders)
        found = [f for f in found if 'clean' not in f and 'results' not in f]
        for f in found:
            all_files.append((f, ds_name))

    print(f"\n🔍 Total files to process: {len(all_files)}")
    
    results = []
    # TQDM progress bar
    for file_path, ds_name in tqdm(all_files, desc="Processing"):
        success, msg = process_subject(file_path, OUTPUT_DIR, ds_name)
        results.append((os.path.basename(file_path), success, msg))

    # Summary
    print("\n" + "="*50)
    print("📊 FINAL SUMMARY")
    print("="*50)
    
    successes = [r for r in results if r[1] and "Skipped" not in r[2]]
    skipped = [r for r in results if "Skipped" in r[2]]
    failures = [r for r in results if not r[1]]

    print(f"Total Found: {len(results)}")
    print(f"✅ Processed: {len(successes)}")
    print(f"⏭️  Skipped:   {len(skipped)}")
    print(f"❌ Failed:    {len(failures)}")
    
    if failures:
        print("\n📋 Failures Log:")
        for f in failures[:10]:
            print(f"{f[0][:30]:<30} -> {f[2]}")