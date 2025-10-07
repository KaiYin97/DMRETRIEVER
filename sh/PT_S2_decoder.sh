#!/bin/bash
set -e  # 出错就退出脚本 

python -m DMRetriever.decoder \
    --model_name_or_path RAG_FT/data/A2_model_cache/merged_Qwen3bi-0.6B \
    --cache_dir ./cache/model \
    --train_data RAG_FT/data/A3_PT_set/new_PT_final/QA_eli5.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/ccnet.jsonl  \
        RAG_FT/data/A3_PT_set/new_PT_final/domain_LLM_filter_remained.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/FC_fever.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/NLI_all_nli.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/NLI_x_nli.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/npr.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/QA_gooaq.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/QA_NQ.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/QA_paq_5M.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/QA_squad.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/QAdoc_hotpotqa.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/QAdoc_msdoc.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/QAdoc_nf.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/s2orc.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/sentence_compression.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/STS_altlex.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/STS_quora_dup.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/STS_quora.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/title_body_agnews.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/Twitter.jsonl \
        RAG_FT/data/A3_PT_set/new_PT_final/xsum.jsonl \
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
    --output_dir RAG_FT/data/D_train_output/PT/Qwen3Bi_all_data_420Gib \
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
    

