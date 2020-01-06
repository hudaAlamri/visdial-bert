#!/bin/bash
#SBATCH --job-name=batch
#SBATCH --output=log/Jan6_logs.out
#SBATCH --error=log/Jan6_logs.err
#SBATCH --gres gpu:8
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --partition=short
#SBATCH -x asimo,calculon,ash,jarvis,bb8

ARGUMENTS=("$@")
OTHER_PARAMS=${ARGUMENTS[@]:0:${#ARGUMENTS[@]}}

echo "Current host:" hostname

set -x
source activate vilbert

python train_language_only_baseline.py -enable_visdom 1 -visdom_server http://asimo.cc.gatech.edu -visdom_server_port 7776 -batch_size 60 -lr 4e-05 -image_lr 2e-05 -batch_multiply 1 -num_options 100 -lm_loss_coeff 1 -num_epochs 1 -mask_prob 0.1 -n_gpus 8 -sequences_per_image 8 -nsp_loss_coeff 1 -num_negative_samples 1 -visdial_tot_rounds 11 -visdom_env debugvisdial_batch_60_lr_4e-05_ilr_2e-05_bm_1_lmcoeff_1_nspcoeff_1_maskprob_0.1_spi_8_tot_rnds_11_job -save_name 06-Jan-20-18:35:47-MonDubgvisdial_batch_60_lr_4e-05_ilr_2e-05_bm_1_lmcoeff_1_nspcoeff_1_maskprob_0.1_spi_8_tot_rnds_11_job 

