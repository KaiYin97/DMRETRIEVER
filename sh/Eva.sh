python DMRetriever/eva/orchestrate/pipeline.py \
  --model DMRetriever/XLM/Final \
  --pool mean \
  --eva_test test \
  --rebuild_corpus_emb \
  --rebuild_query_emb \
  --ckpt_type "full" \
  --batch 16 \
  --parent Test/DM_33M
