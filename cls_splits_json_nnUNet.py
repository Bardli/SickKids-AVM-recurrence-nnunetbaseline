#!/usr/bin/env python3
import os
import json
import pandas as pd
from collections import Counter

# ================== EDIT THESE ==================
# IN_CSV = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/T1+C_5folds_scanlvl_RRRRR_NEWSPLITS.csv" #dataset003
# OUT_JSON = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset003_AVM_T1+C/splits_final.json"
# IN_CSV = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/T1+C_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv" #dataset033
# OUT_JSON = "/hpf/projects/jquon/sumin/nnUNet_data/nnssl_preprocessed/Dataset033_AVM_T1+C/cls_splits_11folds.json"
# IN_CSV = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/combined_T1+C_TOFMRA_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv" #dataset055
# OUT_JSON = "/hpf/projects/jquon/sumin/nnUNet_data/nnssl_preprocessed/Dataset055_AVM_T1+C_TOFMRA/cls_splits_11folds.json"
# IN_CSV = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/TOFMRA_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv" #dataset044
# OUT_JSON = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset044_AVM_TOFMRA/splits_final.json"
# IN_CSV = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/combined_T1+C_TOFMRA_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv" #dataset055
# OUT_JSON = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset055_AVM_T1+C_TOFMRA/splits_final.json"
IN_CSV = "/hpf/projects/jquon/sumin/labels/PatientLevel_Recurrence_StratifiedKFold/T1+C_POSTSURG_LOO_folds_scanlvl_RRRRR_NEWSPLITS.csv" #dataset066
OUT_JSON = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset066_AVM_T1+C_postSurg/splits_final.json"
# ===============================================


def normalize_id(x: str) -> str:
    """
    Normalize IDs to 'basename.nii.gz'.
    This matches what you were doing before.
    """
    x = str(x)
    base = os.path.basename(x)
    if base.endswith(".nii.gz"):
        return base
    if "." not in base:
        return base + ".nii.gz"
    return base


def strip_nii_ext(name: str) -> str:
    """
    Remove .nii.gz or any other extension.
    nnUNet splits_final.json should contain IDs without extensions.
    """
    base = os.path.basename(name)
    if base.endswith(".nii.gz"):
        return base[:-7]  # remove ".nii.gz"
    # fallback for other extensions
    return os.path.splitext(base)[0]


def class_counts(labels):
    return Counter(labels)


def proportions(cnt: Counter) -> dict:
    total = sum(cnt.values())
    return {k: (v / total if total > 0 else 0.0) for k, v in cnt.items()}


def main():
    df = pd.read_csv(IN_CSV)

    # Normalize file IDs (to something like "foo.nii.gz")
    df["new_name"] = df["new_name"].apply(normalize_id)

    # Detect folds from the CSV instead of hardcoding 5/11
    fold_ids = sorted(df["fold"].unique())

    folds_out = []

    for i in fold_ids:
        val_df = df[df["fold"] == i]
        train_df = df[df["fold"] != i]

        # Full filenames with .nii.gz (as before)
        val_full = sorted(val_df["new_name"].unique().tolist())
        train_full = sorted(
            list(set(train_df["new_name"].tolist()) - set(val_full))
        )

        # Strip extensions for nnUNet
        val_ids = sorted(strip_nii_ext(x) for x in val_full)
        train_ids = sorted(strip_nii_ext(x) for x in train_full)

        # Print class balance report (same as old script)
        train_cnt = class_counts(train_df["class_label"].tolist())
        val_cnt = class_counts(val_df["class_label"].tolist())

        print(f"\n=== Fold {i} ===")
        print(f"n_train: {len(train_ids)} | n_val: {len(val_ids)}")
        print("train_counts:", dict(train_cnt))
        print("val_counts:  ", dict(val_cnt))
        print(
            "train_props: ",
            {k: f"{v:.3f}" for k, v in sorted(proportions(train_cnt).items())},
        )
        print(
            "val_props:   ",
            {k: f"{v:.3f}" for k, v in sorted(proportions(val_cnt).items())},
        )

        # This is already in the final nnUNet "list-of-folds" format
        folds_out.append(
            {
                "train": train_ids,
                "val": val_ids,
                "test": [],  # keep empty list for compatibility if you want
            }
        )

    # Save final splits file
    with open(OUT_JSON, "w") as f:
        json.dump(folds_out, f, indent=2)

    print(f"\nWrote nnUNet splits file to {OUT_JSON}")


if __name__ == "__main__":
    main()
