"""
=============================================================================
EEG QUALITY CONTROL VISUALIZATION
=============================================================================
Objective:
    Generate a composite dashboard for visual quality control of EEG data.
    This function is called by the preprocessing pipeline to create PDF reports.

Components:
    1. Raw Signal Trace: Visual overview of time-series data to spot artifacts.
    2. PSD (Power Spectral Density): Frequency spectrum analysis (1-100 Hz).
    3. TFR (Time-Frequency Representation): Multitaper analysis for specific 
       channels to visualize spectral changes over time.

Technical Note:
    Uses the Matplotlib 'Agg' backend to generate figures without requiring 
    an active GUI window, optimized for automated pipelines.

Execution:
    Imported and used by 'preprocess_pipeline.py'.
=============================================================================
"""

import numpy as np
import mne
from mne.time_frequency import tfr_multitaper
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
# Add 'src' to system path to import config
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

# (Optional: Import config if specific constants are needed in future)
# from config import BANDS

# =============================================================================
# 1. SETUP
# =============================================================================

# Use 'Agg' backend for non-interactive plotting (essential for batch processing)
matplotlib.use('Agg') 
mne.set_log_level('WARNING')

def get_plots(raw: mne.io.Raw, step: str, 
              scalings: dict = {'eeg': 40e-6}, 
              xscale: str = 'linear', 
              channel_idx: list = [0],
              plot_ica_overlay: bool = False, 
              ica = None) -> plt.Figure:
    """
    Generates a summary figure containing Raw traces, PSD, and TFR plots.
    """
    
    # 1. Plot Raw Signal
    def plot_raw_img(raw, scalings):
        n_ch = len(raw.ch_names)
        # Force matplotlib backend for MNE to ensure compatibility
        with mne.viz.use_browser_backend('matplotlib'):
            fig = raw.plot(n_channels=n_ch, scalings=scalings, title=step, 
                           show_scrollbars=False, show=False, duration=10.0)
            
            # Render to numpy array
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
        return data

    # 2. Plot Power Spectral Density (PSD)
    def plot_psd_img(raw, xscale):
        # fmax updated to 100Hz to match preprocessing filter settings
        fig = raw.compute_psd(fmin=1, fmax=100).plot(picks='eeg', show=False)
        
        # Render to numpy array
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return data

    # 3. Plot Time-Frequency Representation (TFR)
    def plot_tfr_on_ax(raw, ax, ch_idx):
        freqs = np.arange(4, 45, 2) 
        n_cycles = freqs / 2.0
        
        try:
            # Create temporary epochs for TFR calculation
            epochs = mne.make_fixed_length_epochs(raw, duration=2.0, overlap=0, verbose=False)
            tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, 
                                 average=True, return_itc=False)
            
            # Plot on specific axis
            tfr.plot([ch_idx], baseline=(None, None), mode='logratio', 
                     axes=ax, show=False, colorbar=True)
        except Exception as e:
            print(f"⚠️ TFR Plot failed: {e}")

    # --- Assemble Composite Figure ---
    img_raw = plot_raw_img(raw, scalings)
    img_psd = plot_psd_img(raw, xscale)

    # Define layout: Raw top-left, TFR top-right, PSD bottom
    fig, axes = plt.subplot_mosaic(
        [['ax_raw', 'ax_raw', 'ax_tfr'],
         ['ax_psd', 'ax_psd', 'ax_psd']],
        figsize=(20, 15)
    )
    
    # Place Raw Image
    axes['ax_raw'].imshow(img_raw)
    axes['ax_raw'].axis('off')
    axes['ax_raw'].set_title(f"Raw Signal - {step}", fontsize=15)

    # Place PSD Image
    axes['ax_psd'].imshow(img_psd)
    axes['ax_psd'].axis('off')

    # Place TFR Plot (Safe channel selection)
    target_ch = channel_idx[0] if channel_idx[0] < len(raw.ch_names) else 0
    plot_tfr_on_ax(raw, axes['ax_tfr'], target_ch)
    axes['ax_tfr'].set_title(f'Time-Freq ({raw.ch_names[target_ch]})', fontsize=12)

    fig.suptitle(f"Quality Control Report: {step}", fontsize=20)
    plt.tight_layout()
    
    return fig