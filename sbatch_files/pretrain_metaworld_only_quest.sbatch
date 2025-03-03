#!/bin/bash

#SBATCH -J euijin-jat-pretrain                    # Job 이름
#SBATCH -p base_suma_rtx3090
#SBATCH -q base_qos
#SBATCH -o /home/euijinrnd/sbatch_log/jat_pretrain_quest_epoch50_lr2e-5_bs52_bf16.out
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:4                      # Job에 사용할 리소스 (GPU)

conda activate jat

EXP_NAME="jat_pretrain_quest_epoch50_lr2e-5_bs52_bf16"

export WANDB_PROJECT=jat_pretraining

accelerate launch my_scripts/meta-world/train_jat_tokenized_pretrain_metaworld_quest_tokenized.py \
--output_dir checkpoints/pre_trained/${EXP_NAME} \
--run_name ${EXP_NAME} \
--model_name_or_path jat-project/jat \
--tasks metaworld \
--trust_remote_code \
--save_strategy epoch \
--num_train_epochs 50 \
--logging_steps 10 \
--per_device_train_batch_size 18 \
--gradient_accumulation_steps 1 \
--learning_rate 2e-5 \
--logging_first_step \
--dispatch_batches False \
--dataloader_num_workers 10 \
--bf16 \
--save_safetensors 0 # 모델 저장 issue (shared tensor)
