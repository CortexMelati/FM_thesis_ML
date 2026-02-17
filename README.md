# Thesis EEG: Chronic Pain Classification using Resting-State EEG

**Author:** Jasmyne
**Date:** February 2026
**Description:** This repository contains the complete data science pipeline for analyzing Resting-State EEG (rsEEG) data to identify spectral biomarkers for Chronic Pain. The project merges the internal **TDBrain** dataset with an external **Chronic Pain** dataset, utilizing both standard Machine Learning (Spectral Power) and Riemannian Geometry approaches.


*Note: The scripts perform analysis on local data stored outside the repository. Update the `BASE_DIR` variable in the scripts to match your local data path.*

## Link to data

This repository contains the processing pipeline code. The raw EEG data is not included due to privacy restrictions. TDBrain data can be requested via [[https://brainclinics.com/resources/](https://brainclinics.com/resources/)] and Chronic pain dataset on OSF, DOI: [10.17605/OSF.IO/M45J2](https://doi.org/10.17605/OSF.IO/M45J2)

## Project Workflow

The research pipeline is divided into three distinct phases:

### Phase 1: Data Preparation

* **Harmonization:** Restructuring external datasets to match TDBrain BIDS-like format.
* **Preprocessing:** MNE-Python pipeline (Filtering, RANSAC, AutoReject).
* **Feature Engineering:** Extracting Relative Power Spectral Density (PSD) and Covariance Matrices.
* **Merging:** Combining datasets based on age, gender, and indication.

### Phase 2: Biological Validation

Before training predictive models, the data quality is verified against known neurophysiological markers:

* **Berger Effect:** Validating Alpha blocking (Eyes Open vs. Eyes Closed).
* **Healthy Aging:** Checking the correlation between Age and Alpha frequency.
* **Global Power:** Statistical comparison of global band powers between groups.

### Phase 3: Machine Learning & Benchmarking

* **Main Benchmark:** Comparing 6 algorithms (Linear & Non-Linear) across 3 scenarios.
* **Riemannian Geometry:** Testing robustness against site effects using Tangent Space Mapping.
* **Model Inspection:** Analyzing Bias/Variance, Feature Importance, and Frequency Band Ablation.

---

## Directory Structure

```text
# 📂 Project Structure

The analysis pipeline is organized into modular components to separate data preparation from statistical analysis and modeling.

thesis-eeg/
├── src/
│   ├── Preprocessing/                 # Data Cleaning & Harmonization
│   │   ├── Chronicpain prep/          # Scripts specific to external dataset
│   │   │   ├── amend_vhdr...          # Fixes BrainVision headers
│   │   │   ├── fill_nans...           # Handles missing values
│   │   │   └── moving_files.py        # Restructures to BIDS format
│   │   ├── preprocess_pipeline.py     # Main MNE-Python pipeline (RANSAC/AutoReject)
│   │   ├── final_prep.py              # Merges features into master CSV
│   │   └── split_participants...      # Splits TDBrain metadata
│   │
│   ├── Visualizations_ML/             # Validation & Inspection Scripts
│   │   ├── validate_physiology.py     # Berger Effect & Age correlations
│   │   ├── visualize_site_effect.py   # Comparison of scanner noise
│   │   ├── visualize_heatmap.py       # Generates Topomaps
│   │   ├── ML_bias_variance.py        # Bias-Variance trade-off analysis
│   │   └── Analysis_Ablation.py       # Feature importance (Leave-One-Band-Out)
│   │
│   ├── ML_Main.py                     # Main Benchmark (LR, XGB, SVM, RF)
│   └── ML_Riemann.py                  # Riemannian Geometry Pipeline
│
├── results/                           # Output directory for CSVs and Figures
├── environment.yml                    # Conda environment specification
└── settings.json                      # VS Code workspace settings
```

## Quick Start

**1. Create the Master Dataset:**

Walk through Chronicpain prep

split .xlsx file of TDbrain into the needed subjects.

```
python ./thesis-eeg/src/split_participants.py
```

```
python ./thesis-eeg/src/preprocess_pipeline.py
```

```
python ./thesis-eeg/src/final_prep.py
```

**2. Run the Main Machine Learning Benchmark:**

```
python ./thesis-eeg/src/ML_Main.py
```

**3. Run Riemannian Geometry Comparison:**

```
python ./thesis-eeg/src/ML_Riemann.py
```
