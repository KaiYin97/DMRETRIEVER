#!/usr/bin/env bash  


# ===== Configuration ===== #

# Root directory containing model checkpoints (checkpoint-*/ subfolders)
MODEL_DIR="path_to_folder_containing_ckpts_during_training"


LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# GPU list — modify based on your setup (single GPU: (0); multi-GPU: (0 1 2 ...))
GPUS=(0)


SLEEP_SECS=$((10*60))

# Arguments for ckpt_batch.py (no manual --min_ckpt / --max_ckpt needed)
PARAMS=(
  --model_dir "$MODEL_DIR"
  --pool mean
  --eva_test eva
  --rebuild_corpus_emb
  --rebuild_query_emb
  --discard_intermediate
  --ckpt_type lora
  --backbone_type qwen3bi
  --batch 16
  --skip_done
)

while true; do
  TIMESTAMP=$(date +'%Y%m%d_%H%M')

  # Launch evaluation jobs on all specified GPUs in parallel
  for idx in "${!GPUS[@]}"; do
    GPU_ID=${GPUS[$idx]}
    LOG_FILE="$LOG_DIR/eval_${TIMESTAMP}_gpu${GPU_ID}.log"

    echo "[$(date +'%F %T')] ▶ Starting evaluation: GPU $GPU_ID → log $LOG_FILE"
    CUDA_VISIBLE_DEVICES=$GPU_ID \
      python RAG_FT/code/eva/orchestrate/ckpt_batch.py \
        "${PARAMS[@]}" \
      > "$LOG_FILE" 2>&1 &
  done

  wait
  echo "[$(date +'%F %T')] ✔ Evaluation round completed. Sleeping for ${SLEEP_SECS}s..."
  
  sleep $SLEEP_SECS
done
