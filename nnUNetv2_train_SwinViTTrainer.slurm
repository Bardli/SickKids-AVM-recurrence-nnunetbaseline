#!/bin/bash
#SBATCH --job-name=5thattempt_TOFMRA_ResEnc_SwinViTTrainer_lr_1e-4
#SBATCH --time=48:00:00
#SBATCH -c 20
#SBATCH --gres=gpu:NVIDIA_H100_80GB_HBM3:1
#SBATCH --mem=100G
#SBATCH --output=logs/5thattempt_TOFMRA_ResEncL_SwinViTTrainer_lr_1e-4_%j.out
#SBATCH --error=logs/5thattempt_TOFMRA_ResEncL_SwinViTTrainer_lr_1e-4_%j.err

set -euo pipefail

### Load micromamba shell function
eval "$(/hpf/projects/jquon/micromamba/bin/micromamba shell hook -s bash)"

### Activate your environment
micromamba activate cyy_clone
nvidia-smi

export NNUNET_USE_FP16=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export nnUNet_raw=/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_raw
export nnUNet_preprocessed=/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_preprocessed
export nnUNet_results=/hpf/projects/jquon/sumin/nnUNet_data/nnUNet_results
export WANDB_DIR=/hpf/projects/jquon/tmp/wandb
export WANDB_CACHE_DIR=/hpf/projects/jquon/tmp/wandb_cache
export TORCH_HOME=/hpf/projects/jquon/tmp/torch_cache
### export TMPDIR=/hpf/projects/jquon/models/nnunetcls_baseline/tmp/tmp
### export TORCH_HOME=/tmp/$USER/torch_$SLURM_JOB_ID
### export TMPDIR=/tmp/$USER/tmp_$SLURM_JOB_ID
export TMPDIR=/hpf/projects/jquon/tmp/tmp
### mkdir -p "$TORCH_HOME" "$TMPDIR"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$TORCH_HOME"
cd /hpf/projects/jquon/models/nnunetcls_baseline


nnUNetv2_train 004 3d_fullres 0 -tr SwinViTTrainer_ep300_NoMirroring -p nnUNetResEncUNetLPlans
### nnUNetv2_train 004 3d_fullres 0 -tr SwinViTTrainer_ep300 -p nnUNetResEncUNetLPlans
