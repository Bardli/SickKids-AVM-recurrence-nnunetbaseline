#!/bin/bash
#SBATCH --job-name=TOFMRAf6-11_infer
#SBATCH --time=48:00:00
#SBATCH -c 20
#SBATCH --gres=gpu:1
#SBATCH -p gen_gpu
#SBATCH --mem=100G
#SBATCH --output=logs/TOFMRAf6-11_infer_%j.out
#SBATCH --error=logs/TOFMRAf6-11_infer_%j.err

eval "$(/hpf/projects/jquon/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate cyy_clone
cd /hpf/projects/jquon/models/nnunetcls_baseline
python inference_cls_ckpt_copy.py

# python nnunet_cls_infer_nii.py \
#     --input_path /path/to/input/images/ \
#     --output_path /path/to/output/ \
#     --model_path /path/to/trained/model \
#     --fold all \
#     --checkpoint checkpoint_best.pth \
#     --device cuda \
#     --cls_mode mean
