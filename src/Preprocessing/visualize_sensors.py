"""
=============================================================================
SENSOR VISUALIZATION: 10-20 SYSTEM LAYOUT
=============================================================================
Objective:
    Visualize the physical arrangement of the selected 20 EEG channels.
    This ensures that the channel names (e.g., Fp1, Oz) map correctly 
    to their expected locations on the scalp before further analysis.

Outputs:
    1. 2D Topomap (Matplotlib):
       - Shows the "flattened" sensor map (EEGLAB style).
       - Highlights a specific target channel (e.g., Fp1) in Green.
    2. 3D Plot (Interactive):
       - Opens an interactive window showing sensors on a 3D head model.

Configuration:
    - Montage: 'easycap-M1' (Standard 10-20 layout).
    - Channels: The 20 common channels used throughout this thesis (from Config).

Execution:
    python ./FM_thesis_ML/src/Preprocessing/visualize_sensors.py
=============================================================================
"""

import mne
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from config import CHANNELS

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Import channels from centralized config
COMMON_CHANNELS = CHANNELS

TARGET_LABEL = 'Fp1'

def visualize_sensors():
    print("🧠 Loading sensor positions (Standard 10-20)...")
    
    # 1. Load the standard 10-20 montage (easycap-M1)
    montage = mne.channels.make_standard_montage('easycap-M1')
    
    # 2. Create an Info object with ONLY the selected channels
    # This creates a "dummy" dataset container required for plotting
    info = mne.create_info(COMMON_CHANNELS, sfreq=500, ch_types='eeg')
    
    # 3. Apply the montage to the info object
    # MNE automatically maps the 20 channel names to 3D coordinates
    info.set_montage(montage)
    
    # ==========================================================================
    # 2. 2D PLOT (WITH AUTOMATIC RADIUS ADJUSTMENT)
    # ==========================================================================
    print("🎨 Generating 2D Topomap (EEGLAB style)...")
    fig, ax = plt.subplots(figsize=(7, 7))
    
    # Using 'eeglab' sphere parameter for optimal 2D projection
    mne.viz.plot_sensors(info, kind='topomap', sphere="eeglab", show_names=True, 
                         axes=ax, show=False)
    
    
    # --- Highlight Fp1 in Green ---
    if TARGET_LABEL in info.ch_names:
        idx = info.ch_names.index(TARGET_LABEL)
        
        # Retrieve scatter points from plot
        pts = ax.collections[0]
        colors = pts.get_facecolors()
        
        # If all points share one color, expand array to handle individual colors
        if len(colors) < len(info.ch_names): 
            colors = np.array([[0, 0, 0, 1]] * len(info.ch_names))
        
        # Set target channel to Green
        colors[idx] = [0, 0.8, 0, 1] 
        pts.set_facecolors(colors)
        
        # Highlight Label Text
        for text_obj in ax.texts:
            if text_obj.get_text() == TARGET_LABEL:
                text_obj.set_color('green')
                text_obj.set_fontweight('bold')
                text_obj.set_fontsize(14)

    plt.show()

    # ==========================================================================
    # 3. 3D PLOT (INTERACTIVE)
    # ==========================================================================
    print("\n📦 Generating 3D Plot...")
    try:
        # Reconstruct montage specifically for the selected channels
        pos_dict = montage.get_positions()['ch_pos']
        target_pos = {ch: pos_dict[ch] for ch in COMMON_CHANNELS if ch in pos_dict}
        
        clean_montage = mne.channels.make_dig_montage(
            ch_pos=target_pos,
            nasion=montage.get_positions()['nasion'],
            lpa=montage.get_positions()['lpa'],
            rpa=montage.get_positions()['rpa'],
            coord_frame='head'
        )
        
        print("   ⏳ Opening interactive window...")
        clean_montage.plot(kind='3d', show_names=True, show=True)
        
    except Exception as e:
        print(f"❌ Error generating 3D plot: {e}")

if __name__ == "__main__":
    visualize_sensors()