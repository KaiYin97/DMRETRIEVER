#!/usr/bin/env bash
set -euo pipefail

python -m DMRetriever.BiAttn_Ada.run_mlm \
  --model_name_or_path path_to_backbone \
  --train_file path_pre_train_data \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 22 \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --mask_token_type blank \
  --data_collator_type default \
  --mlm_probability 0.2 \
  --overwrite_output_dir \
  --output_dir path_to_save_model_ckpt \
  --save_steps 100 \
  --lora_r 16 \
  --evaluation_strategy steps \
  --eval_steps 100 \
  --torch_dtype bfloat16 \
  --attn_implementation eager \
  --report_to none \
  --num_train_epochs 1 \
  --logging_steps 100 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.01 \
  --overwrite_cache False \
  --cache_dir path_to_save_cache_data \
  --bf16
