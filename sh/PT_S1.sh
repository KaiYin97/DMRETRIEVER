#!/usr/bin/env bash
set -euo pipefail

python -m DMRetriever.BiAttn_Ada.run_mlm \
  --model_name_or_path RAG_FT/data/A2_model_cache/qwen3-0.6B \
  --train_file RAG_FT/data/A3_PT_set/raw_txt/merged_disaster_wikitext.txt \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 22 \
  --do_train \
  --do_eval \
  --max_seq_length 512 \
  --mask_token_type blank \
  --data_collator_type default \
  --mlm_probability 0.2 \
  --overwrite_output_dir \
  --output_dir RAG_FT/data/D_train_output/PT_LLM_S1/DM_0.6B_wikiAnddomain \
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
  --cache_dir RAG_FT/data/cache_data \
  --bf16
