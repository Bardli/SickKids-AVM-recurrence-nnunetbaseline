#!/bin/bash
#SBATCH --job-name=f3-f4_TOFMRA_2ndtry_ResEnc_SwinViTTrainer_lr_1e-4
#SBATCH --time=48:00:00
#SBATCH -c 20
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --output=logs/f3-f4_TOFMRA_2ndtry_ResEncL_SwinViTTrainer_lr_1e-4_%j.out
#SBATCH --error=logs/f3-f4_TOFMRA_2ndtry_ResEncL_SwinViTTrainer_lr_1e-4_%j.err

micromamba activate cyy_clone

cd /hpf/projects/jquon/models/nnunetcls_baseline

### python /hpf/projects/jquon/models/baseline_cls/nnUNet/run_cls_infer_ckpt_TOFMRA.py
python /hpf/projects/jquon/models/nnunetcls_baseline/run_cls_infer_ckpt_TOFMRA_f3-f4.py
# python nnunet_cls_infer_nii.py \
#     --input_path /path/to/input/images/ \
#     --output_path /path/to/output/ \
#     --model_path /path/to/trained/model \
#     --fold all \
#     --checkpoint checkpoint_best.pth \
#     --device cuda \
#     --cls_mode mean


# set -euo pipefail

# # Load micromamba shell function
# eval "$(/hpf/projects/jquon/micromamba/bin/micromamba shell hook -s bash)"

# # Activate your environment
# micromamba activate baseline_cls

# export nnUNet_raw=/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw
# export nnUNet_preprocessed=/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_preprocessed
# export nnUNet_results=/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results
# export WANDB_DIR=/hpf/projects/jquon/models/baseline_cls/nnUNet/tmp/wandb
# export WANDB_CACHE_DIR=/hpf/projects/jquon/models/baseline_cls/nnUNet/tmp/wandb_cache
# export TORCH_HOME=/hpf/projects/jquon/models/baseline_cls/nnUNet/tmp/torch
# export TMPDIR=/hpf/projects/jquon/models/baseline_cls/nnUNet/tmp/tmp
# # export TMPDIR=/hpf/projects/jquon/tmp/tmp
# mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TORCH_HOME"


# cd /hpf/projects/jquon/models/baseline_cls/nnUNet

# nnUNetv2_train 004 3d_fullres 4 -tr ViTTrainer_lr_1e_minus4 -p nnUNetResEncUNetLPlans