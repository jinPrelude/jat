#!/bin/bash

#SBATCH -J euijin-corl                    # Job 이름
#SBATCH -p base_suma_rtx3090
#SBATCH -q base_qos
#SBATCH -o /home/euijinrnd/sbatch_log/jat_metaworld_only_no_obs_pred.out
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:8                      # Job에 사용할 리소스 (GPU)
#SBATCH --time=12:00:00 

accelerate launch my_scripts/train_jat_tokenized_pretrain_metaworld.py \
--output_dir checkpoints/jat \
--model_name_or_path jat-project/jat \
--tasks metaworld \
--trust_remote_code \
--observation_loss_coef 0.0 \
--per_device_train_batch_size 20 \
--gradient_accumulation_steps 2 \
--save_steps 5000 \
--run_name jat_no_obs_pred \
--logging_steps 100 \
--logging_first_step \
--dispatch_batches False \
--dataloader_num_workers 8 \
--max_steps 25000 \
--save_safetensors 0 # 모델 저장 issue (shared tensor)