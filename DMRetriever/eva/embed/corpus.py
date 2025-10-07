# eva/embed/corpus.py
# ---------------------------------------------------------------------------

from __future__ import annotations
import gc
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from eva.utils.config import (
    BASELINE_INDEX_DIR,
    DEFAULT_BATCH,
    DEFAULT_MAXLEN,
    USE_FP16,
    CHECKPOINT_ROOT,
    EVA_TEST_CFG,
)
from eva.utils.embed_utils import embed_texts
from eva.utils.model_utils import load_model_and_tokenizer

# Runtime setup
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE_TORCH = torch.float16 if (USE_FP16 and _DEVICE == "cuda") else torch.float32


def _slug(model_name: str) -> str:
    """Convert model name to a filesystem-safe identifier."""
    return model_name.replace("/", "_")


def _resolve_ckpt_path(model_name: str | Path) -> Path:
    """Resolve checkpoint path (relative to CHECKPOINT_ROOT if needed)."""
    p = Path(model_name)
    return p if p.exists() else Path(CHECKPOINT_ROOT) / model_name


def build_corpus_emb(
    model_name: str | Path,
    pool: str = "cls",
    eva_test: Literal["eva", "test"] = "eva",
    *,
    ckpt_type: Literal["auto", "full", "lora"] = "auto",
    backbone: str | Path | None = None,
    backbone_type: Literal["decoder_only", "qwen3bi"] = "decoder_only",
    rebuild: bool = False,
    use_encode: bool = False,
    padding_side: Literal["left", "right"] = "right",
) -> Path:
    """Encode or load passage embeddings."""
    if eva_test not in EVA_TEST_CFG:
        raise ValueError("eva_test must be 'eva' or 'test'")

    cfg = EVA_TEST_CFG[eva_test]
    corpus_json = Path(cfg["CORPUS_JSON"])
    corpus_base = Path(cfg["CORPUS_BASE"])

    # Load or build ordered corpus
    if not corpus_json.exists():
        print(f"[corpus_emb] building ordered_corpus → {corpus_json}")
        from eva.utils.io_utils import build_ordered_corpus
        build_ordered_corpus(corpus_base, str(corpus_json))
    from eva.utils.io_utils import load_ordered_corpus
    corpus = load_ordered_corpus(str(corpus_json))

    # Cache path for embeddings
    slug = _slug(str(model_name))
    emb_dir = Path(BASELINE_INDEX_DIR)
    emb_dir.mkdir(parents=True, exist_ok=True)
    cache_fp = emb_dir / f"{slug}.{eva_test}.{ 'fp16' if USE_FP16 else 'fp32' }.npy"

    # Use cached embeddings if available
    if cache_fp.exists() and not rebuild:
        print(f"[corpus_emb] cache hit for {model_name} ({eva_test})")
        return cache_fp

    # Load model and tokenizer
    ckpt_path = _resolve_ckpt_path(model_name)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    tok, mdl = load_model_and_tokenizer(
        ckpt_path,
        device=_DEVICE,
        torch_dtype=_DTYPE_TORCH,
        ckpt_type=ckpt_type,
        backbone_path=backbone,
        trust_remote_code=True,
        backbone_type=backbone_type,
        padding_side=padding_side,
    )

    # Encode corpus passages
    embs, _ = embed_texts(
        mdl,
        tok,
        corpus,
        max_len=DEFAULT_MAXLEN,
        batch_size=DEFAULT_BATCH,
        pool_tag=pool,
        device=_DEVICE,
        dtype=_DTYPE_TORCH,
        use_encode=use_encode,
        desc=f"{slug}-corpus-{eva_test}",
    )

    # Save embeddings
    if USE_FP16:
        embs = embs.astype(np.float16)
    np.save(cache_fp, embs)

    del mdl, tok
    gc.collect()
    if _DEVICE == "cuda":
        torch.cuda.empty_cache()

    return cache_fp
