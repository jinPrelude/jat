#!/bin/bash

#SBATCH -J euijin-jat                    # Job 이름
#SBATCH -p big_suma_rtx3090
#SBATCH -q big_qos
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1                      # Job에 사용할 리소스 (GPU)

MODEL_NAME=$1
# Compute log file and redirect output inside the script if necessary.
OUTPUT_PATH="/home/euijinrnd/sbatch_log/$(echo ${MODEL_NAME} | tr '/' '_').out"
# Optionally, redirect output:
exec > "${OUTPUT_PATH}" 2>&1

source activate base
conda activate jat

python scripts/eval_jat.py \
--model_name_or_path "$MODEL_NAME" \
--trust_remote_code \
--num_episodes 50 \
--tasks metaworld-bin-picking metaworld-box-close metaworld-door-lock metaworld-door-unlock metaworld-hand-insert