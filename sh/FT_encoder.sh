#!/usr/bin/env bash
set -euo pipefail

python -m torch.distributed.run --nproc_per_node 2 --master_port 29601 -m DMRetriever.encoder \
  --do_train \
  --model_name_or_path RAG_FT/data/A2_model_cache/Base_afterPT \
  --cache_dir ./cache/model \
  --train_data RAG_FT/data/A_FT_set/label_pool_inf_within_top2 \
  --cache_path ./cache/data \
  --train_group_size 10 \
  --query_max_len 512 \
  --passage_max_len 512 \
  --pad_to_multiple_of 8 \
  --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \
  --query_instruction_format '{}: {}' \
  --knowledge_distillation False \
  --output_dir RAG_FT/data/D_train_output/FT_encoder/Ablation_CBF_Base \
  --overwrite_output_dir \
  --learning_rate 1e-5 \
  --bf16 \
  --num_train_epochs 1 \
  --dataloader_drop_last True \
  --warmup_ratio 0.01 \
  --logging_steps 100 \
  --save_steps 100 \
  --temperature 0.01 \
  --sentence_pooling_method mean \
  --normalize_embeddings True \
  --kd_loss_type kl_div \
  --report_to none \
  --gradient_accumulation_steps 8 \
  --per_device_train_batch_size 32 \
  --weight_decay 0.005 \
  --no_in_batch_neg_flag True
