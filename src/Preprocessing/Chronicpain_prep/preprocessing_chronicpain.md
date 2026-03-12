# Data Harmonization: Chronic Pain Dataset

**Objective:** To harmonize the external "Chronic Pain" dataset so that its file structure and file formats exactly match the internal TDBrain dataset. This alignment is crucial to ensure that both datasets can be processed through the unified MNE-Python pipeline without requiring dataset-specific exception handling.

---

# Pipeline

1. python ./FM_thesis_ML/src/Preprocessing/Chronicpain_prep/moving_files.py
2. python ./FM_thesis_ML/src/Preprocessing/Chronicpain_prep/amend_vhdr_file_chronicpain.py

## Directory Restructuring (BIDS-like)

**Rationale:** The original dataset had a flat or inconsistent structure. To maintain compatibility with the TDBrain pipeline, files were reorganized into a BIDS-like hierarchy: `sub-XXX/ses-1/eeg/`.

**Target Structure:**

```text
/data/Chronicpainset/derivatives/
├── sub-001/
│   └── ses-1/
│       └── eeg/
│           ├── sub-001.vhdr
│           ├── sub-001.eeg
│           └── sub-001.vmrk
└── ...
```

### ✅ Summary of Changes

After executing this pipeline, the **Chronic Pain Dataset** is:

1. **Structurally Harmonized:** Aligned with `sub-XXX/ses-1/eeg` BIDS-convention.
2. **Numerically Clean:** Free of `NaN` values in feature CSVs.
3. **Readable:** `.vhdr` headers are now compatible with MNE-Python v1.10+.

The data is now ready for merging with TDBrain.
