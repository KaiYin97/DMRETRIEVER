# -*- coding: utf-8 -*-
"""Unified loader for HuggingFace Tokenizer / Model with optional LoRA and custom backbones."""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Tuple

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

try:
    from peft import PeftModel, PeftConfig
except ImportError as e:
    raise RuntimeError("PEFT support requires: pip install peft") from e

try:
    from bidirectional_qwen3.bidirectional_qwen3 import Qwen3BiModel  
except Exception:
    try:
        from bidirectional_qwen3 import Qwen3BiModel  
    except Exception:
        Qwen3BiModel = None  


def _str2path(p) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _load_tokenizer(
    path: str | Path,
    *,
    trust_remote_code: bool = True,
    padding_side: Literal["left", "right"] = "right",
) -> AutoTokenizer:
    """Load tokenizer with fallback padding configuration."""
    tok = AutoTokenizer.from_pretrained(
        path, trust_remote_code=trust_remote_code, local_files_only=False, use_fast=False
    )
    if getattr(tok, "pad_token_id", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token
    if hasattr(tok, "padding_side"):
        tok.padding_side = padding_side
    return tok


def load_model_and_tokenizer(
    ckpt_path: str | Path,
    *,
    device: str = "cpu",
    torch_dtype: torch.dtype | None = None,
    ckpt_type: Literal["auto", "full", "lora"] = "auto",
    backbone_path: str | Path | None = None,
    trust_remote_code: bool = True,
    backbone_type: Literal["decoder_only", "qwen3bi"] = "decoder_only",
    padding_side: Literal["left", "right"] = "right",
) -> Tuple[AutoTokenizer, torch.nn.Module]:
    """Main entry: load model and tokenizer depending on checkpoint type and backbone."""
    ckpt_path = _str2path(ckpt_path)
    backbone_path = _str2path(backbone_path) if backbone_path else None

    # detect adapter
    is_adapter_dir = (ckpt_path / "adapter_config.json").exists() \
        or (ckpt_path / "adapter_model.safetensors").exists()

    if ckpt_type == "full":
        use_adapter = False
    elif ckpt_type == "lora":
        use_adapter = True
    else:
        use_adapter = is_adapter_dir

    # load model
    if use_adapter:
        # LoRA adapter case
        if backbone_path is None:
            peft_cfg = PeftConfig.from_pretrained(ckpt_path)
            backbone_path = _str2path(peft_cfg.base_model_name_or_path)

        tokenizer = _load_tokenizer(
            backbone_path,
            trust_remote_code=trust_remote_code,
            padding_side=padding_side,
        )

        # select backbone
        if backbone_type == "qwen3bi":
            if Qwen3BiModel is None:
                raise RuntimeError("Qwen3BiModel not found. Please ensure it's installed or importable.")
            base_model = Qwen3BiModel.from_pretrained(  # type: ignore
                backbone_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                local_files_only=False,
            )
        else:
            base_model = AutoModel.from_pretrained(
                backbone_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                local_files_only=False,
            )

        model = (
            PeftModel.from_pretrained(base_model, ckpt_path)
            .to(device)
            .eval()
        )

    else:
        # full checkpoint case
        tokenizer = _load_tokenizer(
            ckpt_path,
            trust_remote_code=trust_remote_code,
            padding_side=padding_side,
        )
        if backbone_type == "qwen3bi":
            if Qwen3BiModel is None:
                raise RuntimeError("Qwen3BiModel not found. Please ensure it's installed or importable.")
            model = (
                Qwen3BiModel.from_pretrained(  # type: ignore
                    ckpt_path,
                    torch_dtype=torch_dtype,
                    trust_remote_code=trust_remote_code,
                    local_files_only=False,
                )
                .to(device)
                .eval()
            )
        else:
            config = AutoConfig.from_pretrained(
                ckpt_path,
                trust_remote_code=True,
                local_files_only=False,
            )
            config.local_path = str(ckpt_path)
            model = (
                AutoModel.from_pretrained(
                    ckpt_path,
                    config=config,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    local_files_only=False,
                )
                .to(device)
                .eval()
            )

    return tokenizer, model
