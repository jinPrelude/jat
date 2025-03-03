#!/bin/bash

#SBATCH -J euijin-jat-pretrain2                    # Job 이름
#SBATCH -p base_suma_rtx3090
#SBATCH -q base_qos
#SBATCH -o /home/euijinrnd/sbatch_log/jat_mini_pretrain_quest_epoch50_lr2e-5_bs64_bf16_worker8.out
#SBATCH --cpus-per-task=9
#SBATCH --gres=gpu:2                     # Job에 사용할 리소스 (GPU)

conda activate jat-mw2.0

EXP_NAME="jat_mini_pretrain_quest_epoch50_lr2e-5_bs52_bf16_worker8"

export WANDB_PROJECT=jat_pretraining

accelerate launch my_scripts/meta-world/train_jat_tokenized_pretrain_metaworld_quest_tokenized.py \
--output_dir checkpoints/pre_trained/${EXP_NAME} \
--run_name ${EXP_NAME} \
--model_name_or_path /scratch/euijinrnd/jat/local/jat-mini \
--config_name /scratch/euijinrnd/jat/local/jat-mini-config \
--tasks metaworld \
--trust_remote_code \
--num_train_epochs 100 \
--save_steps 0.2 \
--logging_steps 10 \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--logging_first_step \
--dispatch_batches False \
--dataloader_num_workers 8 \
--bf16 \
--save_safetensors 0 # 모델 저장 issue (shared tensor) \

