"""
Validation report for .npy files
"""

import numpy as np
import os
import pandas as pd

print("=" * 50)
print("VALIDATION REPORT")
print("=" * 50)

# Check main data files
files = [
    "pamap2_10s_windows.npy", "pamap2_5s_windows.npy",
    "wisdm_10s_windows.npy", "wisdm_5s_windows.npy",
    "mhealth_10s_windows.npy", "mhealth_5s_windows.npy",
    "eeg_preprocessed.npy",
    "ecg_preprocessed.npy"
]

for f in files:
    path = f"submission_sample/{f}"
    if os.path.exists(path):
        data = np.load(path)
        print(f"\n{f}:")
        print(f"  - Shape: {data.shape}")
        print(f"  - Data type: {data.dtype}")
        print(f"  - Has NaN: {np.isnan(data).any()}")
        print(f"  - File size: {os.path.getsize(path) / 1024:.1f} KB")
    else:
        print(f"\n{f}: MISSING")

# Check metadata CSV
print("\n" + "=" * 50)
print("METADATA CSV")
print("=" * 50)
meta_path = "submission_sample/metadata.csv"
if os.path.exists(meta_path):
    df = pd.read_csv(meta_path)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Datasets: {df['dataset_name'].unique()}")
else:
    print("metadata.csv MISSING")

# Check EEG event labels
print("\n" + "=" * 50)
print("EEG EVENT LABELS")
print("=" * 50)
event_path = "submission_sample/eeg_event_labels.npy"
if os.path.exists(event_path):
    labels = np.load(event_path)
    print(f"Shape: {labels.shape}")
    print(f"Unique: {np.unique(labels)}")
    print(f"T1 count: {sum(labels=='T1')}, T2 count: {sum(labels=='T2')}")
else:
    print("eeg_event_labels.npy MISSING")

print("\n" + "=" * 50)
print("RESOURCE ESTIMATES")
print("=" * 50)
print("Raw storage: ~500 MB (subsets only)")
print("Processed storage: ~15 MB")
print("Peak RAM: < 500 MB")
print("Runtime: < 10 minutes")