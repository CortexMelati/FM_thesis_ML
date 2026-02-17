"""
=============================================================================
🧹 DATA CLEANING: NaN REPLACEMENT UTILITY
=============================================================================
Objective:
    Recursively scan a specific dataset folder for CSV files and replace 
    all missing values (NaNs) with 0.

    This ensures that downstream processing pipelines (e.g., EEG feature 
    extraction) do not crash due to missing data points in the external dataset.

Scope:
    - Target Directory: 'Chronicpainset/derivatives'
    - Operation: Overwrites existing CSV files with the cleaned version.
    - Logging: Creates a log file tracking exactly which files were modified.

Execution:
    python ./FM_thesis_ML/src/Preprocessing/Chronicpain prep/fill_nans_chronicpain.py
=============================================================================
"""

import os
import sys
import pandas as pd
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
# Use config path + 'derivatives'
BASE_PATH = CHRONIC_PAIN_DIR / "derivatives"

LOG_FILE = BASE_PATH / "nans_fixed_log.txt"
LOG_CSV = BASE_PATH / "nans_fixed_log.csv"

def fix_nans():
    """
    Scans for CSV files, replaces NaNs with 0, and logs the changes.
    """
    print(f"🚀 Starting NaN Cleanup in: {BASE_PATH}")
    log_rows = []
    
    if not BASE_PATH.exists():
         print(f"❌ Error: The directory {BASE_PATH} does not exist.")
         return

    os.makedirs(BASE_PATH, exist_ok=True)

    # Initialize Log File
    with open(LOG_FILE, "w", encoding="utf-8") as log:
        log.write("NaN Replacement Log\n")
        log.write("===================\n\n")
        log.write("File\tNaN Count\tTotal Values\tReplaced (%)\n")
        log.write("-" * 60 + "\n")

    # Recursive search for .csv files
    for csv_file in BASE_PATH.rglob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            total_values = df.size
            n_nans = df.isna().sum().sum()

            if n_nans > 0:
                # Replace NaNs with 0
                df = df.fillna(0)
                df.to_csv(csv_file, index=False)
                
                perc = (n_nans / total_values) * 100
                log_rows.append([str(csv_file), n_nans, total_values, perc])
                
                print(f"✅ {csv_file.name}: Replaced {n_nans} NaNs ({perc:.4f}%)")
            else:
                print(f"✔️ {csv_file.name}: No NaNs found")

        except Exception as e:
            print(f"⚠️ Error processing {csv_file.name}: {e}")
            with open(LOG_FILE, "a", encoding="utf-8") as log:
                log.write(f"⚠️ Error processing {csv_file}: {e}\n")

    # Save log as CSV for easier analysis later
    if log_rows:
        log_df = pd.DataFrame(log_rows, columns=["File", "NaN Count", "Total Values", "Replaced (%)"])
        log_df.to_csv(LOG_CSV, index=False)
        print(f"\n📄 detailed CSV log saved to: {LOG_CSV}")

    print("\n🎉 Done! All NaN values have been replaced with 0.")
    print(f"📄 Log file: {LOG_FILE}")

def analyze_log_file():
    """
    Reads the generated CSV log and prints the files with the most replacements.
    """
    print("\n📊 Analyzing cleanup results...")
    try:
        if not LOG_CSV.exists():
            print("   No changes were made, so no CSV log exists.")
            return

        df = pd.read_csv(LOG_CSV)
        df = df.dropna(how="all")
        
        # Sort by percentage replaced (descending)
        df.sort_values("Replaced (%)", ascending=False, inplace=True)
        
        print("\nTop files by modification percentage:")
        print(df.head().to_string(index=False))
    except Exception as e:
        print(f"❌ Could not read log file: {e}")

# def inspect_csv_file():
#     """
#     Helper function to manually inspect a specific file (Debug use only).
#     """
#     csv_file = BASE_PATH / r"sub-060\ses-1\eeg\sub-060_task-EO_eeg_new.csv"
#     if not csv_file.exists():
#         print(f"File not found: {csv_file}")
#         return

#     df = pd.read_csv(csv_file)

#     nan_counts = df.isna().sum()
#     total_counts = len(df)
#     nan_percentage = (nan_counts / total_counts) * 100

#     summary = pd.DataFrame({
#         "Count_NaN": nan_counts,
#         "Percentage": nan_percentage
#     }).sort_values(by="Percentage", ascending=False)

#     print(f"📄 Analysis for: {csv_file.name}")
#     print(summary.head(10))

#     mean_nan = nan_percentage.mean()
#     print(f"\nAverage {mean_nan:.2f}% NaNs per channel across {len(df.columns)} channels.")

if __name__ == "__main__":
    fix_nans()
    analyze_log_file()
    # inspect_csv_file()