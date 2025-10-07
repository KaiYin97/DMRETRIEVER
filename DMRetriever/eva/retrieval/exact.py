# eva/retrieval/exact.py
# ---------------------------------------------------------------------------
"""Exact dot product retrieval that produces **label pools**."""

from __future__ import annotations

import os
import sys

if __name__ == "__main__" and __package__ is None:
    here = os.path.abspath(os.path.dirname(__file__))
    code_root = os.path.abspath(os.path.join(here, "..", ".."))
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    __package__ = "eva.retrieval"

import gc
import json
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import torch

from eva.embed.corpus import build_corpus_emb
from eva.utils.config import (
    QUERY_EMB_DIR,
    LABEL_POOL_DIR,
    DEFAULT_TOPK,
    USE_FP16,
    EVA_TEST_CFG,
)
from eva.utils.io_utils import (
    build_ordered_corpus,
    load_ordered_corpus,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE_TORCH = torch.float16 if (USE_FP16 and _DEVICE == "cuda") else torch.float32
_DTYPE_NP = np.float16 if USE_FP16 else np.float32


def _slug(model_name: str) -> str:
    """Replace '/' with '_' for filename safety."""
    return model_name.replace("/", "_")


def _iter_tasks(test_query_dir: Path) -> set[str]:
    """Return the set of task names appearing in the query directory."""
    return {fp.stem.split("_", 1)[0] for fp in test_query_dir.glob("*.json")}


def exact_search(
    model_name: str | Path,
    pool: str = "cls",
    eva_test: Literal["eva", "test"] = "eva",
    *,
    tasks: Iterable[str] | None = None,
    ckpt_type: Literal["auto", "full", "lora"] = "auto",
    backbone: str | Path | None = None,
    backbone_type: Literal["decoder_only", "qwen3bi"] = "decoder_only",
    rebuild_corpus_emb: bool = False,
    topk: int = DEFAULT_TOPK,
    use_encode: bool = False,
    padding_side: Literal["left", "right"] = "right",
) -> Path:
    """Run dense retrieval and save label pools."""
    if eva_test not in EVA_TEST_CFG:
        raise ValueError("eva_test must be 'eva' or 'test'")

    cfg = EVA_TEST_CFG[eva_test]
    test_query_dir: Path = Path(
        str(Path("DMRetriever/data/C_test_set") / ("test_query_eva" if eva_test == "eva" else "test_query_test"))
    )
    corpus_json: Path = Path(cfg["CORPUS_JSON"])
    corpus_base: Path = Path(cfg["CORPUS_BASE"])

    # Ordered corpus
    if not corpus_json.exists():
        print(f"[exact] building ordered_corpus → {corpus_json}")
        build_ordered_corpus(corpus_base, str(corpus_json))
    corpus: list[str] = load_ordered_corpus(str(corpus_json))
    print(f"[exact] loaded corpus: {len(corpus):,} passages")

    # Corpus embeddings
    corpus_emb_fp = build_corpus_emb(
        model_name=model_name,
        pool=pool,
        eva_test=eva_test,
        ckpt_type=ckpt_type,
        backbone=backbone,
        backbone_type=backbone_type,
        rebuild=rebuild_corpus_emb,
        use_encode=use_encode,
        padding_side=padding_side,
    )
    corpus_emb = np.load(corpus_emb_fp, mmap_mode="r")
    corpus_emb_T = torch.tensor(corpus_emb, dtype=_DTYPE_TORCH, device=_DEVICE)

    # Output directory
    slug = _slug(str(model_name))
    label_out_dir = Path(LABEL_POOL_DIR) / slug
    label_out_dir.mkdir(parents=True, exist_ok=True)

    # Task filtering
    all_tasks = _iter_tasks(test_query_dir)
    chosen_tasks = set(tasks) if tasks else all_tasks
    bad_tasks = chosen_tasks - all_tasks
    if bad_tasks:
        raise ValueError(f"tasks {bad_tasks} not found in {test_query_dir}")

    # Main loop
    for fp in test_query_dir.glob("*.json"):
        task = fp.stem.split("_", 1)[0]
        if task not in chosen_tasks:
            continue

        arr = json.loads(fp.read_text(encoding="utf-8"))
        queries = [it["user_query"].strip() for it in arr]

        q_emb_fp = Path(QUERY_EMB_DIR) / f"{slug}__{eva_test}" / f"{fp.stem}.npy"
        if not q_emb_fp.exists():
            print(f"[exact] WARN skip {fp.name} missing query_embeddings")
            continue

        q_emb = np.load(q_emb_fp).astype(_DTYPE_NP)
        q_emb_t = torch.tensor(q_emb, dtype=_DTYPE_TORCH, device=_DEVICE)

        # Batched dot product + topk
        label_list: list[list[str]] = []
        B = min(128, q_emb_t.size(0))
        with torch.no_grad():
            for i in range(0, q_emb_t.size(0), B):
                sims = q_emb_t[i : i + B] @ corpus_emb_T.T
                _, idx = sims.topk(topk, dim=1)
                for inds in idx.cpu().numpy():
                    label_list.append([corpus[k] for k in inds])

        # Write label pool
        out_items = [
            {"user_query": q, "passages": ps}
            for q, ps in zip(queries, label_list)
        ]
        out_fp = label_out_dir / f"{fp.stem}_label_pool.json"
        out_fp.write_text(
            json.dumps(out_items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Cleanup
    del corpus_emb, corpus_emb_T, q_emb, q_emb_t
    gc.collect()
    if _DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"[exact] finished label pools saved to {label_out_dir}")
    return label_out_dir


# Lightweight CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Exact dense retrieval (label pool)")
    parser.add_argument("--model", required=True, help="checkpoint or adapter directory")
    parser.add_argument("--pool", default="cls", help="pooling method cls/mean/last")
    parser.add_argument("--eva_test", choices=["eva", "test"], default="eva")
    parser.add_argument("--task", nargs="*", help="only run specified search tasks")
    parser.add_argument("--ckpt_type", choices=["auto", "full", "lora"], default="auto",
                        help="checkpoint type: auto/full/lora")
    parser.add_argument("--backbone", help="base model dir when ckpt_type=lora")
    parser.add_argument("--backbone_type", choices=["decoder_only", "qwen3bi"],
                        default="decoder_only", help="choose backbone class")
    parser.add_argument("--rebuild_corpus_emb", action="store_true",
                        help="recompute corpus embeddings")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                        help="number of passages per query")
    parser.add_argument("--use_encode", action="store_true",
                        help="use model.encode() when available")
    parser.add_argument("--padding_side", choices=["left", "right"], default="right",
                        help="set tokenizer padding side (default: right)")

    args = parser.parse_args()
    exact_search(
        model_name=args.model,
        pool=args.pool,
        eva_test=args.eva_test,
        tasks=args.task,
        ckpt_type=args.ckpt_type,
        backbone=args.backbone,
        backbone_type=args.backbone_type,
        rebuild_corpus_emb=args.rebuild_corpus_emb,
        topk=args.topk,
        use_encode=args.use_encode,
        padding_side=args.padding_side,
    )
