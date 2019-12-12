#!/bin/bash
#SBATCH --job-name=batch
#SBATCH --output=log/dec11_logs.out
#SBATCH --error=log/dec11_logs.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --partition=short



ARGUMENTS=("$@")
OTHER_PARAMS=${ARGUMENTS[@]:0:${#ARGUMENTS[@]}}

echo "Current host:" hostname

set -x
source activate vilbert
python train.py -batch_size 80  -batch_multiply 1 -lr 2e-5 -image_lr 2e-5 -mask_prob 0.1 -sequences_per_image 2 -start_path checkpoints-release/vqa_pretrained_weights




