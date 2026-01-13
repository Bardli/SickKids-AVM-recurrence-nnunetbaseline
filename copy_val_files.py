import json
import os
import shutil

# Paths
json_file = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset003_AVM_T1+C/splits_final.json"
src_folder = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset003_AVM_T1+C/imagesTr"
dst_base = "/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw/Dataset003_AVM_T1+C"

# Load JSON splits
with open(json_file, "r") as f:
    splits = json.load(f)  # list of dicts, one per fold

# Loop over each fold
for fold_idx, fold_dict in enumerate(splits):
    val_list = fold_dict["val"]  # list of file names without .nii.gz
    
    # Destination folder for this fold
    dst_folder = os.path.join(dst_base, f"imagesVal_fold{fold_idx}")
    os.makedirs(dst_folder, exist_ok=True)
    
    # Copy files
    for fname in val_list:
        src_file = os.path.join(src_folder, fname + "_0000.nii.gz")
        dst_file = os.path.join(dst_folder, fname + "_0000.nii.gz")
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
        else:
            print(f"WARNING: File not found: {src_file}")
    
    print(f"Fold {fold_idx}: {len(val_list)} files copied to {dst_folder}")
