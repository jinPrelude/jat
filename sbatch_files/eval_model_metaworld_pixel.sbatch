#!/bin/bash

#SBATCH -J euijin-jat                    # Job 이름
#SBATCH -p base_suma_rtx3090
#SBATCH -q base_qos
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1                      # Job에 사용할 리소스 (GPU)

MODEL_NAME=$1
# Compute log file and redirect output inside the script if necessary.
OUTPUT_PATH="/home/euijinrnd/sbatch_log/$(echo ${MODEL_NAME} | tr '/' '_').out"
# Optionally, redirect output:
exec > "${OUTPUT_PATH}" 2>&1

source activate base
conda activate jat-mw2.0

python my_scripts/meta-world/eval_jat_pixel_metaword.py \
--model_name_or_path "$MODEL_NAME" \
--trust_remote_code \
--num_episodes 50 \
--tasks metaworld \