# 🧠 Thesis EEG: Chronic Pain Classification using Resting-State EEG

**Author:** Jas  
**Date:** February 2026  

**Description:** This repository contains the complete data science pipeline for analyzing Resting-State EEG (rsEEG) data to identify spectral biomarkers for Chronic Pain. The project merges the internal **TDBrain** dataset with an external **Chronic Pain** dataset, utilizing both standard Machine Learning (Spectral Power) and advanced Riemannian Geometry approaches.

*Note: The scripts perform analysis on local data stored outside the repository. Update the `BASE_DIR` variable in `src/config.py` to match your local data path.*

## Link to Data
This repository contains the processing pipeline code. The raw EEG data is not included due to privacy restrictions. 
* **TDBrain data** can be requested via [Brainclinics Resources](https://brainclinics.com/resources/).
* **External Chronic Pain dataset** is available on OSF, DOI: [10.17605/OSF.IO/M45J2](https://doi.org/10.17605/OSF.IO/M45J2).

---

## Execution Pipeline & Project Workflow

The research pipeline is divided into three distinct phases. To reproduce the findings, execute the scripts in the following order:

### Phase 1: Data Preparation & Harmonization
*Restructuring external datasets, MNE-Python cleaning (Filtering, RANSAC, AutoReject), and extracting Relative Power (RP) features.*

1. **External Data Prep:** Fix BrainVision headers and move files for BIDS compliance.
   * `src/Preprocessing/Chronicpain_prep/amend_vhdr_file_chronicpain.py`
   * `src/Preprocessing/Chronicpain_prep/moving_files.py`
2. **Internal Data Prep:** Split the TDBrain metadata to isolate the required cohorts.
   * `src/Preprocessing/split_participants_TDBRAIN.py`
3. **Core Preprocessing:** Clean the EEG signals and extract epochs.
   * `src/Preprocessing/preprocess_pipeline.py`
4. **Feature Extraction:** Extract spectral power and merge datasets.
   * `src/Preprocessing/final_prep.py`

### Phase 2: Biological Validation & Quality Control
*Before training predictive models, the data quality is verified against known neurophysiological markers (Berger Effect, Hardware bias).*

5. **Physiological Validation:**
   * `src/Visualizations_Prep/validate_physiology.py`
6. **Site-Effect Verification (Hardware Drift):**
   * `src/Visualizations_Prep/visualize_site_effect.py`
7. **Statistical Group Differences (TCD Hypothesis):**
   * `src/Visualizations_Prep/stats_global_differences.py`
8. **Sanity Check (ML):** Validate EO/EC prediction and Site prediction.
   * `src/RF_validation_EO_EC.py`
   * `src/RF_site_prediction.py`

### Phase 3: Machine Learning & Benchmarking
*Evaluating 6 linear/non-linear algorithms and the Riemannian Tangent Space Mapping pipeline.*

9. **Spectral Machine Learning Benchmark:**
   * `src/ML_Main.py`
10. **Riemannian Geometry Pipeline:**
    * `src/ML_Riemann.py`
11. **Feature Ablation (Delta band validation):**
    * `src/ML_checks/Analysis_Ablation.py`

---
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
│   ├── RF_site_prediction.py         # Hardware bias check (predicting recording site)
│   │
│   ├── Preprocessing/                # Data Cleaning & Harmonization
│   │   ├── preprocess_pipeline.py    # Main MNE pipeline (Filtering, RANSAC, AutoReject)
│   │   ├── final_prep.py             # Feature extraction & Master CSV generation
│   │   ├── split_participants_TDBRAIN.py # Metadata categorization
│   │   ├── global_powers.py          # Baseline power calculation utilities
│   │   ├── preprocessing_plotting.py # Visual quality checks for cleaned signals
│   │   ├── general_info.py           # Dataset statistics (Age, Sex distribution)
│   │   │
│   │   └── Chronicpain_prep/         # External dataset specific utilities
│   │       ├── amend_vhdr_file_chronicpain.py # Fixes BrainVision headers
│   │       └── moving_files.py       # File restructuring for BIDS compliance
│   │
│   ├── Visualizations_Prep/          # Validation & Statistical Inspection
│   │   ├── stats_global_differences.py # Group-level spectral comparisons
│   │   ├── stats_tcd.py              # Thalamocortical Dysrhythmia specific tests
│   │   ├── validate_physiology.py    # Berger Effect & Age correlations
│   │   ├── visualize_heatmap.py      # Spatial power distribution (Topomaps)
│   │   ├── visualize_sensors.py      # Plots 10-20 system electrode layouts
│   │   └── visualize_site_effect.py  # Statistical comparison of scanner noise
│   │
│   └── ML_checks/                    # Advanced Model Diagnostics
│       ├── Analysis_Ablation.py      # Leave-One-Band-Out band importance
│       ├── ML_bias_variance.py       # Model generalization analysis
│       └── ML_Main_copy.py           # Expanded parameter grid for overfitting checks
│
├── results/                          # Output directory
│   ├── final_dataset.csv             # The master feature file for ML
│   ├── final_benchmark_mega.csv      # Results from ML_Main.py
│   ├── validation_global_powers.csv  # Processed statistical baseline values
│   ├── validation_stats_report.txt   # Output of statistical significance testing
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
1. Create the Master Dataset:

Note: Prior to running the main pipeline, ensure the external Chronic Pain data headers are fixed using the scripts in src/Preprocessing/Chronicpain_prep/.

First, split the .xlsx file of TDBrain into the needed subjects, followed by preprocessing and feature extraction:

```
python ./src/split_participants_TDBRAIN.py
```

```
python ./src/preprocess_pipeline.py
```

```
python ./src/final_prep.py
```

**2. Run the Main Machine Learning Benchmark:**

```
python ./src/ML_Main.py
```

**3. Run Riemannian Geometry Comparison:**

```
python ./src/ML_Riemann.py
```




