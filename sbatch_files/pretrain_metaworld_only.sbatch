#!/bin/bash

#SBATCH -J euijin-jat-pretrain                    # Job 이름
#SBATCH -p base_suma_rtx3090
#SBATCH -q base_qos
#SBATCH -o /home/euijinrnd/sbatch_log/jat_metaworld_only.out
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8                      # Job에 사용할 리소스 (GPU)
#SBATCH --time=12:00:00 

conda activate jat

EXP_NAME="jat_pretrain"

accelerate launch my_scripts/train_jat_tokenized_pretrain_metaworld.py \
--output_dir checkpoints/pre_trained/${EXP_NAME} \
--wandb_project jat_pretraining \
--wandb_watch all \
--run_name ${EXP_NAME} \
--model_name_or_path jat-project/jat \
--tasks metaworld \
--trust_remote_code \
--per_device_train_batch_size 20 \
--gradient_accumulation_steps 2 \
--save_steps 5000 \
--logging_steps 100 \
--logging_first_step \
--dispatch_batches False \
--dataloader_num_workers 8 \
--max_steps 25000 \
--save_safetensors 0 # 모델 저장 issue (shared tensor)