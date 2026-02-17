"""
Central Configuration for Thesis EEG Pipeline.
"""

import os
from pathlib import Path

# ==========================================
# 1. PATH CONFIGURATION
# ==========================================
# Dit bestand staat in: .../FM_thesis_ML/src
CURRENT_DIR = Path(__file__).resolve().parent

# PROJECT_ROOT is één map omhoog: .../FM_thesis_ML
PROJECT_ROOT = CURRENT_DIR.parent

# DATA_ROOT is nog een map omhoog (Thesis) en dan naar Data: .../Thesis/Data
DATA_ROOT = PROJECT_ROOT.parent / "Data"

# Input Directories
TDBRAIN_DIR = DATA_ROOT / "TDBRAIN-dataset"
CHRONIC_PAIN_DIR = DATA_ROOT / "Chronicpainset"

# Output Directories (binnen FM_thesis_ML/results)
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
PROCESSED_DATA_DIR = RESULTS_DIR / "processed_data"

# Ensure output directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. EEG PARAMETERS
# ==========================================
SFREQ = 500
EPOCH_LENGTH = 9.95  # seconds
TMIN = 0
TMAX = EPOCH_LENGTH

# Standard 10-20 System (20 Channels)
CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'Oz', 'O2'
]

# Frequency Bands of Interest
BANDS = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 100) 
}

# ==========================================
# 3. ANALYSIS & VISUALIZATION
# ==========================================
RANDOM_STATE = 42
MIN_AGE = 18.0

# Consistent Color Scheme for Plots
COLORS = {
    'Healthy': '#1f77b4',      # Blue
    'ChronicPain': '#d62728',  # Red
    'Unknown': '#7f7f7f'       # Grey
}

if __name__ == "__main__":
    print(f"✅ Configuration Loaded.")
    print(f"📂 Config Location: {CURRENT_DIR}")
    print(f"📂 Project Root:    {PROJECT_ROOT}")
    print(f"📂 Data Root:       {DATA_ROOT}")
    print(f"📂 Results Dir:     {RESULTS_DIR}")
    
    # Check of de mappen ook echt gevonden worden
    if TDBRAIN_DIR.exists():
        print(f"   -> TDBrain Data gevonden! ✅")
    else:
        print(f"   -> ❌ TDBrain Data NIET gevonden op: {TDBRAIN_DIR}")
        
    if (RESULTS_DIR / "final_dataset.csv").exists():
        print(f"   -> final_dataset.csv gevonden! ✅")
    else:
        print(f"   -> ⚠️ final_dataset.csv nog niet in de results map.")