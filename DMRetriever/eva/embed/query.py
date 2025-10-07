# eva/embed/query.py
# ---------------------------------------------------------------------------

from __future__ import annotations
import gc, json
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from eva.utils.config import (
    TASK2PREFIX,
    QUERY_EMB_DIR,
    DEFAULT_BATCH,
    DEFAULT_MAXLEN,
    USE_FP16,
    CHECKPOINT_ROOT,
    EVA_TEST_DIR,
)
from eva.utils.embed_utils import embed_texts
from eva.utils.model_utils import load_model_and_tokenizer

# Runtime setup
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE_TORCH = torch.float16 if (USE_FP16 and _DEVICE == "cuda") else torch.float32


def _slug(model_name: str) -> str:
    return model_name.replace("/", "_")


def _resolve_ckpt_path(model_name: str | Path) -> Path:
    """Resolve checkpoint path relative to training root if necessary."""
    p = Path(model_name)
    return p if p.exists() else Path(CHECKPOINT_ROOT) / model_name


def build_query_emb(
    model_name: str | Path,
    pool: str = "cls",
    eva_test: Literal["eva", "test"] = "eva",
    *,
    ckpt_type: Literal["auto", "full", "lora"] = "auto",
    backbone: str | Path | None = None,
    backbone_type: Literal["decoder_only", "qwen3bi"] = "decoder_only",
    batch: int = DEFAULT_BATCH,
    max_len: int = DEFAULT_MAXLEN,
    rebuild: bool = False,
    use_encode: bool = False,
    padding_side: Literal["left", "right"] = "right",
) -> Path:
    """Encode eva/test query sets for the specified model."""
    if eva_test not in EVA_TEST_DIR:
        raise ValueError("eva_test must be 'eva' or 'test'")

    slug = _slug(str(model_name))
    out_dir = Path(QUERY_EMB_DIR) / f"{slug}__{eva_test}"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_query_dir = Path(EVA_TEST_DIR[eva_test])
    json_files = list(test_query_dir.glob("*.json"))

    if (
        not rebuild
        and json_files
        and all((out_dir / f"{fp.stem}.npy").exists() for fp in json_files)
    ):
        print(f"[query_emb] cache hit for {model_name} ({eva_test}); skip encoding.")
        return out_dir

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

    for fp in json_files:
        npy_fp = out_dir / f"{fp.stem}.npy"
        if npy_fp.exists() and not rebuild:
            continue

        data = json.loads(fp.read_text(encoding="utf-8"))
        task = fp.stem.split("_", 1)[0]
        prefix = TASK2PREFIX.get(task, "")
        queries = [f"{prefix}: {item['user_query'].strip()}" for item in data]

        embs, valid_idx = embed_texts(
            mdl,
            tok,
            queries,
            max_len=max_len,
            batch_size=batch,
            pool_tag=pool,
            device=_DEVICE,
            dtype=_DTYPE_TORCH,
            use_encode=use_encode,
            desc=f"{slug}-{eva_test}-{fp.name}",
            show_progress=False,
        )

        if len(valid_idx) != len(queries):
            raise RuntimeError(
                f"{fp.name}: expect {len(queries)}, got {len(valid_idx)} embeddings"
            )

        if USE_FP16:
            embs = embs.astype(np.float16)
        np.save(npy_fp, embs)

    del mdl, tok
    gc.collect()
    if _DEVICE == "cuda":
        torch.cuda.empty_cache()

    return out_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Encode query embeddings for eva/test")
    parser.add_argument("--model_name", "--model", dest="model_name", required=True,
                        help="checkpoint or adapter directory")
    parser.add_argument("--pool", default="cls")
    parser.add_argument("--eva_test", choices=["eva", "test"], default="eva")
    parser.add_argument("--ckpt_type", choices=["auto", "full", "lora"], default="auto",
                        help="checkpoint type: auto/full/lora")
    parser.add_argument("--backbone", help="base model dir when ckpt_type=lora")
    parser.add_argument("--backbone_type", choices=["decoder_only","qwen3bi"], default="decoder_only",
                        help="choose backbone class to instantiate")
    parser.add_argument("--batch",   type=int, default=DEFAULT_BATCH)
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAXLEN)
    parser.add_argument("--rebuild", action="store_true",
                        help="force recomputation even if cache exists")
    parser.add_argument("--use_encode", action="store_true",
                        help="use model.encode() when available")
    parser.add_argument("--padding_side", choices=["left", "right"], default="right",
                        help="set tokenizer padding side (default: right)")

    args = parser.parse_args()
    build_query_emb(
        model_name=args.model_name,
        pool=args.pool,
        eva_test=args.eva_test,
        ckpt_type=args.ckpt_type,
        backbone=args.backbone,
        backbone_type=args.backbone_type,
        batch=args.batch,
        max_len=args.max_len,
        rebuild=args.rebuild,
        use_encode=args.use_encode,
        padding_side=args.padding_side,
    )
