#!/bin/bash
set -e 

python -m DMRetriever.decoder \
    --model_name_or_path path_to_backbone_after_PT_BiAttn_Ada \
    --cache_dir ./cache/model \
    --train_data path_to_pre_train_data \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --do_train \
    --cache_path ./cache/data \
    --train_group_size 10 \
    --query_max_len 512 \
    --passage_max_len 512 \
    --pad_to_multiple_of 8 \
    --query_instruction_for_retrieval 'Represent this sentence for searching relevant passages: ' \
    --query_instruction_format '{}: {}' \
    --knowledge_distillation False \
    --output_dir path_to_save_model_ckpt \
    --overwrite_output_dir \
    --learning_rate 1e-4 \
    --bf16 \
    --num_train_epochs 2 \
    --dataloader_drop_last True \
    --warmup_ratio 0.01 \
    --logging_steps 100 \
    --save_steps 100 \
    --temperature 0.01 \
    --sentence_pooling_method mean \
    --normalize_embeddings True \
    --report_to none \
    --gradient_accumulation_steps 1\
    --per_device_train_batch_size 4 \
    --weight_decay 0.01 \
    --same_dataset_within_batch True \
    --deepspeed FT_sh/ds_stage1.json \
    --gradient_checkpointing \
    --backbone_type qwen3bi \
    

