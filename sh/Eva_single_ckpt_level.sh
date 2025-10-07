## for encoder-only backbone 
python DMRetriever/eva/orchestrate/pipeline.py \
  --model path_to_model_ckpt \
  --pool mean \
  --eva_test test \
  --rebuild_corpus_emb \
  --rebuild_query_emb \
  --ckpt_type "full" \
  --batch 16 \
  --parent path_to_save_test_result

## for decoder-only backbone 
python DMRetriever/eva/orchestrate/pipeline.py \
    --model path_to_model_ckpt \
    --pool mean \
    --eva_test test \
    --ckpt_type "lora" \
    --batch 16 \
    --backbone_type "qwen3bi" \
    --parent path_to_save_test_result
