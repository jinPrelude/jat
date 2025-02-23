#!/bin/bash

#SBATCH -J euijin-jat-pretrain                    # Job 이름
#SBATCH -p base_suma_rtx3090
#SBATCH -q base_qos
#SBATCH -o /home/euijinrnd/sbatch_log/jat_metaworld_only_quest_tokenized.out
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8                      # Job에 사용할 리소스 (GPU)

conda activate jat

EXP_NAME="jat_pretrain_quest"

export WANDB_PROJECT=jat_pretraining

accelerate launch my_scripts/meta-world/train_jat_tokenized_pretrain_metaworld_quest_tokenized.py \
--output_dir checkpoints/pre_trained/${EXP_NAME} \
--run_name ${EXP_NAME} \
--model_name_or_path jat-project/jat \
--tasks metaworld \
--trust_remote_code \
--save_strategy epoch \
--
--save_steps 5000 \
--num_train_epochs 100 \
--logging_steps 10 \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 2 \
--logging_first_step \
--dispatch_batches False \
--dataloader_num_workers 8 \
--save_safetensors 0 # 모델 저장 issue (shared tensor)