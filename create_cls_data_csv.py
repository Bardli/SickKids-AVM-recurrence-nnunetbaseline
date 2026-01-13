#!/usr/bin/env python
import pandas as pd
from pathlib import Path

# ================== EDIT THESE ==================
# train_csv = "/hpf/projects/jquon/sumin/labels/train_split_MRI_T1.csv"
# val_csv   = "/hpf/projects/jquon/sumin/labels/val_split_MRI_T1.csv"
# train_csv = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/T1+C_5folds_scanlvl_RRRRR_NEWSPLITS.csv"
# out_csv = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_preprocessed/Dataset003_AVM_T1+C/cls_data.csv"
# train_csv = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/MRI_TOFMRA_5folds.csv"
# out_csv   = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_preprocessed/Dataset004_AVM_TOFMRA/cls_data.csv"
# train_csv = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/T1+C_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv"
# out_csv = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_preprocessed/Dataset033_AVM_T1+C/cls_data.csv"
# train_csv = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/TOFMRA_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv"
# out_csv = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_preprocessed/Dataset044_AVM_TOFMRA/cls_data.csv"
# train_csv = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/combined_T1+C_TOFMRA_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv"
# out_csv = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_preprocessed/Dataset055_AVM_T1+C_TOFMRA/cls_data.csv"
train_csv = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/T1+C_POSTSURG_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv"
out_csv = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_preprocessed/Dataset066_AVM_T1+C_postSurg/cls_data.csv"
# ===============================================

# Columns
id_col    = "new_name"
label_col = "class_label"

def clean_identifier(name: str) -> str:
    """Strip .nii.gz and trailing _0000 if present"""
    if name.endswith(".nii.gz"):
        name = name[:-7]
    if name.endswith("_0000"):
        name = name[:-5]
    return name

# Read and merge
df_train = pd.read_csv(train_csv, usecols=[id_col, label_col])
# df_val   = pd.read_csv(val_csv,   usecols=[id_col, label_col])

# df = pd.concat([df_train, df_val], ignore_index=True)
df = df_train.copy()

# Clean identifiers
df["identifier"] = df[id_col].apply(clean_identifier)
df = df.rename(columns={label_col: "label"})[["identifier", "label"]]

# Save
Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

print(f"✅ Wrote {out_csv} with {len(df)} entries")
