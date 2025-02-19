#!/bin/bash

#SBATCH -J euijin-jat-finetune                    # Job 이름
#SBATCH -p big_suma_rtx3090
#SBATCH -q big_qos
#SBATCH -o /home/euijinrnd/sbatch_log/jat_metaworld_only_finetune_20000_2.out
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8                      # Job에 사용할 리소스 (GPU)
#SBATCH --time=03:00:00 

conda activate jat

accelerate launch my_scripts/train_jat_tokenized_finetune_metaworld.py \
--model_name_or_path checkpoints/jat/pre-trained/checkpoint-20000 \
--output_dir checkpoints/jat/fine-tuned/pretrained-checkpoint-20000_2 \
--tasks metaworld-bin-picking metaworld-box-close metaworld-door-lock metaworld-door-unlock metaworld-hand-insert \
--trust_remote_code \
--learning_rate 5e-5 \
--per_device_train_batch_size 20 \
--gradient_accumulation_steps 1 \
--save_steps 500 \
--logging_steps 100 \
--logging_first_step \
--dispatch_batches False \
--dataloader_num_workers 6 \
--max_steps 2500 \
--save_safetensors 0 # 모델 저장 issue (shared tensor)
# --top_k_demos 1600 \