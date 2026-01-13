#!/usr/bin/env python3
import os
import shutil
import pandas as pd

# Input CSVs
csv_files = [
    # "/hpf/projects/jquon/sumin/labels/train_split_MRI_T1.csv",
    # "/hpf/projects/jquon/sumin/labels/val_split_MRI_T1.csv"
    # "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/MRI_T1+C_5folds.csv"
    # "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/MRI_TOFMRA_5folds.csv"
    # "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/T1+C_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv"
    # "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/TOFMRA_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv"
    # "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/combined_T1+C_TOFMRA_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv"
    # "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/T1+C_5folds_scanlvl_RRRRR_NEWSPLITS.csv"
    "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/T1+C_POSTSURG_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv"
]

# Source and destination directories
src_dir = "/hpf/projects/jquon/data/organized_data"
# dst_dir = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset003_AVM_T1+C/imagesTr"
# dst_dir = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset004_AVM_TOFMRA/imagesTr"
# dst_dir = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset044_AVM_TOFMRA/imagesTr"
# dst_dir = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset055_AVM_T1+C_TOFMRA/imagesTr"
dst_dir = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset066_AVM_T1+C_postSurg/imagesTr"


# Make sure destination exists
os.makedirs(dst_dir, exist_ok=True)

# Collect all file names from the CSVs
all_files = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    if "new_name" not in df.columns:
        raise ValueError(f"'new_name' column not found in {csv_file}")
    all_files.extend(df["new_name"].tolist())

# Process each file
for fname in all_files:
    src_path = os.path.join(src_dir, fname)
    if not os.path.exists(src_path):
        print(f"WARNING: {src_path} not found, skipping.")
        continue

    # Destination file name: change .nii.gz → _0000.nii.gz
    if fname.endswith(".nii.gz"):
        new_name = fname.replace(".nii.gz", "_0000.nii.gz")
    else:
        new_name = fname + "_0000.nii.gz"  # fallback

    dst_path = os.path.join(dst_dir, new_name)

    print(f"Copying {src_path} → {dst_path}")
    shutil.copy2(src_path, dst_path)

print("✅ Done copying and renaming files.")
