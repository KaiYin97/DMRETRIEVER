# eva/orchestrate/pipeline.py
# ---------------------------------------------------------------------------
"""Evaluation pipeline for a single checkpoint."""

from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Iterable, Literal

if __name__ == "__main__" and __package__ is None:
    here = os.path.abspath(os.path.dirname(__file__))
    code_root = os.path.abspath(os.path.join(here, "..", ".."))
    if code_root not in sys.path:
        sys.path.insert(0, code_root)
    __package__ = "eva.orchestrate"

from eva.embed.query import build_query_emb
from eva.retrieval.exact import exact_search
from eva.evaluation.metrics import calc_metrics
from eva.utils.config import (
    DEFAULT_BATCH,
    DEFAULT_MAXLEN,
    DEFAULT_TOPK,
    QUERY_EMB_DIR,
    LABEL_POOL_DIR,
    BASELINE_INDEX_DIR,
)

import logging
LOGGER = logging.getLogger(__name__)


def _slug(model: str | Path) -> str:
    return str(model).replace("/", "_")


def _cleanup_intermediate(slug: str, eva_test: str) -> None:
    """Delete intermediate artifacts (query_emb / corpus_emb / label_pool)."""
    from shutil import rmtree
    qdir = Path(QUERY_EMB_DIR) / f"{slug}__{eva_test}"
    if qdir.exists():
        rmtree(qdir)
    lp_dir = Path(LABEL_POOL_DIR) / slug
    if lp_dir.exists():
        rmtree(lp_dir)
    for ext in ("fp16.npy", "fp32.npy"):
        for fp in (
            Path(BASELINE_INDEX_DIR) / f"{slug}.{ext}",
            Path(BASELINE_INDEX_DIR) / f"{slug}.eva.{ext}",
            Path(BASELINE_INDEX_DIR) / f"{slug}.test.{ext}",
        ):
            if fp.exists():
                fp.unlink()


def run_pipeline(
    model: str | Path,
    parent: str,
    *,
    eva_test: Literal["eva", "test"] = "eva",
    pool: str = "cls",
    batch: int = DEFAULT_BATCH,
    max_len: int = DEFAULT_MAXLEN,
    tasks: Iterable[str] | None = None,
    ckpt_type: Literal["auto", "full", "lora"] = "auto",
    backbone: str | Path | None = None,
    backbone_type: Literal["decoder_only", "qwen3bi"] = "decoder_only",
    rebuild_query_emb: bool = False,
    rebuild_corpus_emb: bool = False,
    topk: int = DEFAULT_TOPK,
    keep_intermediate: bool = True,
    use_encode: bool = False,
    padding_side: Literal["left", "right"] = "right",
) -> None:
    """Execute the full evaluation pipeline on a single checkpoint."""
    model_str = str(model)

    print("¶ STEP 1  build/load query embeddings")
    build_query_emb(
        model_name=model_str,
        pool=pool,
        eva_test=eva_test,
        ckpt_type=ckpt_type,
        backbone=backbone,
        backbone_type=backbone_type,
        batch=batch,
        max_len=max_len,
        rebuild=rebuild_query_emb,
        use_encode=use_encode,
        padding_side=padding_side,
    )

    print("¶ STEP 2  exact dense retrieval")
    exact_search(
        model_name=model_str,
        pool=pool,
        eva_test=eva_test,
        tasks=tasks,
        ckpt_type=ckpt_type,
        backbone=backbone,
        backbone_type=backbone_type,
        rebuild_corpus_emb=rebuild_corpus_emb,
        topk=topk,
        use_encode=use_encode,
        padding_side=padding_side,
    )

    print("¶ STEP 3  calc metrics")
    out_dir = calc_metrics(model=model_str, parent=parent)

    overall_fp = out_dir / "ndcg_overall.txt"
    if overall_fp.is_file():
        print(f"[RESULT] {model_str} · {overall_fp.read_text().strip()}")

    if not keep_intermediate:
        _cleanup_intermediate(_slug(model_str), eva_test)

    print(f"Pipeline finished for {model_str}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="One-shot evaluation pipeline for a single checkpoint"
    )
    parser.add_argument("--model", required=True, help="checkpoint or adapter directory")
    parser.add_argument("--parent", required=True, help="group name under performance/")
    parser.add_argument("--eva_test", choices=["eva", "test"], default="eva")
    parser.add_argument("--pool", default="cls")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAXLEN)
    parser.add_argument("--tasks", nargs="*", metavar="TASK")
    parser.add_argument("--ckpt_type", choices=["auto", "full", "lora"], default="auto",
                        help="checkpoint type: auto/full/lora")
    parser.add_argument("--backbone", help="base model dir when ckpt_type=lora")
    parser.add_argument("--backbone_type", choices=["decoder_only", "qwen3bi"],
                        default="decoder_only", help="choose backbone class")
    parser.add_argument("--rebuild_query_emb", action="store_true")
    parser.add_argument("--rebuild_corpus_emb", action="store_true")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--discard_intermediate", action="store_true",
                        help="delete intermediate artifacts after run")
    parser.add_argument("--use_encode", action="store_true",
                        help="use model.encode() when available")
    parser.add_argument("--padding_side", choices=["left", "right"], default="right",
                        help="set tokenizer padding side (default: right)")

    args = parser.parse_args()
    run_pipeline(
        model=args.model,
        parent=args.parent,
        eva_test=args.eva_test,
        pool=args.pool,
        batch=args.batch,
        max_len=args.max_len,
        tasks=args.tasks,
        ckpt_type=args.ckpt_type,
        backbone=args.backbone,
        backbone_type=args.backbone_type,
        rebuild_query_emb=args.rebuild_query_emb,
        rebuild_corpus_emb=args.rebuild_corpus_emb,
        topk=args.topk,
        keep_intermediate=not args.discard_intermediate,
        use_encode=args.use_encode,
        padding_side=args.padding_side,
    )
