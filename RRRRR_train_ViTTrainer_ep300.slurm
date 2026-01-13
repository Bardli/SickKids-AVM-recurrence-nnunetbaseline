#!/bin/bash
#SBATCH --job-name=33ViTTrainer_ep300_NoMirroring_x2
#SBATCH --time=48:00:00
#SBATCH -c 20
#SBATCH --gres=gpu:NVIDIA_H100_80GB_HBM3:1
#SBATCH --mem=100G
#SBATCH --array=0-10
#SBATCH --output=logs/33ViTTrainer_ep300_NoMirroring_x2_%A_%a.out
#SBATCH --error=logs/33ViTTrainer_ep300_NoMirroring_x2_%A_%a.err
set -euo pipefail

### Load micromamba shell function
eval "$(/hpf/projects/jquon/micromamba/bin/micromamba shell hook -s bash)"

### Activate your environment
micromamba activate cyy_clone


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


nnUNetv2_train 033 3d_fullres $SLURM_ARRAY_TASK_ID -tr ViTTrainer_ep300_NoMirroring_x2 -p nnUNetResEncUNetLPlans
