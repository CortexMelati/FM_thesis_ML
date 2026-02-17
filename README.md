# Thesis EEG: Chronic Pain Classification using Resting-State EEG

**Author:** Jas


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

FM_thesis_ML/
├── src/
│   ├── config.py                     # Centralized configuration (Paths, Bands, Channels)
│   ├── ML_Main.py                    # Main ML Benchmark (Nested CV, Multi-model)
│   ├── ML_Riemann.py                 # Riemannian Geometry & Tangent Space Pipeline
│   ├── RF_validation_EO_EC.py        # EO/EC physiological signal quality validation
│   ├── check_paths.py                # Utility to verify environment directory setup
│   │
│   ├── Preprocessing/                # Data Cleaning & Harmonization
│   │   ├── preprocess_pipeline.py    # Main MNE pipeline (Filtering, RANSAC, AutoReject)
│   │   ├── final_prep.py             # Feature extraction & Master CSV generation
│   │   ├── split_participants_TDBRAIN.py # Metadata categorization
│   │   ├── check_data_completeness.py# Validates .npy file presence vs metadata
│   │   ├── global_powers.py          # Baseline power calculation utilities
│   │   ├── visualize_sensors.py      # Plots 10-20 system electrode layouts
│   │   ├── preprocessing_plotting.py # Visual quality checks for cleaned signals
│   │   ├── general_info.py           # Dataset statistics (Age, Sex distribution)
│   │   │
│   │   └── Chronicpain prep/         # External dataset specific utilities
│   │       ├── amend_vhdr_file_...   # Fixes BrainVision headers
│   │       ├── fill_nans_...         # Metadata cleaning
│   │       └── moving_files.py       # File restructuring for BIDS compliance
│   │
│   ├── Visualizations_ML/            # Validation & Statistical Inspection
│   │   ├── ML_bias_variance.py       # Model generalization analysis
│   │   ├── Analysis_Ablation.py      # Leave-One-Band-Out band importance
│   │   ├── validate_physiology.py    # Berger Effect & Age correlations
│   │   ├── visualize_heatmap.py      # Spatial power distribution (Topomaps)
│   │   ├── visualize_site_effect.py  # Statistical comparison of scanner noise
│   │   ├── stats_global_differences.py # Group-level spectral comparisons
│   │   └── stats_tcd.py              # Thalamocortical Dysrhythmia specific tests
│   │
│   └── __pycache__/                  # Compiled Python files (ignored by Git)
│
├── results/                          # Output directory
│   ├── final_dataset.csv             # The master feature file for ML
│   ├── final_benchmark_mega.csv      # Results from ML_Main.py
│   ├── hyperparameter_report.txt     # Log of optimized model settings
│   ├── TDBrain/                      # Processed TDBrain .npy epochs
│   ├── chronicpain/                  # Processed External .npy epochs
│   ├── processed_data/               # Temporary storage for pipeline stages
│   └── figures/                      # Generated plots (ROC, CM, Topomaps)
│
├── .gitignore                        # Prevents sensitive data from being uploaded
├── environment.yml                   # Conda environment specification
├── LICENSE                           # Legal usage terms
└── README.md                         # Project documentation
```

## Quick Start

**1. Create the Master Dataset:**

Walk through Chronicpain prep folder

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
