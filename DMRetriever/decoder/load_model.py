# DMRetriever/decoder/load_model.py
import os
import re
import torch
from typing import Optional
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

# Use in-repo Qwen3Bi implementation
from DMRetriever.bidirectional_qwen3 import Qwen3BiModel  # type: ignore


def find_largest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    pat = re.compile(r"checkpoint-(\d+)")
    maxn, maxc = -1, None
    for f in os.listdir(checkpoint_dir):
        m = pat.search(f)
        if m:
            n = int(m.group(1))
            if n > maxn:
                maxn, maxc = n, f
    return os.path.join(checkpoint_dir, maxc) if maxc else None


def get_model(
    model_args,
    output_dir: str,
    resize: bool,
    resize_tokens: int,
    *,
    backbone_type: str = "decoder_only",
):
    if model_args.config_name:
        config = AutoConfig.from_pretrained(
            model_args.config_name,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )
    config.use_cache = False

    if backbone_type == "qwen3bi":
        if Qwen3BiModel is None:  # type: ignore
            raise ImportError("bidirectional_qwen3 is missing.")
        model = Qwen3BiModel.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True,
            config=config,
        )
    else:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            trust_remote_code=model_args.trust_remote_code,
        )

    if model_args.raw_peft:
        model.set_input_embeddings(torch.load(os.path.join(model_args.raw_peft, "embedding", "emb.pth")))
        model = PeftModel.from_pretrained(model, model_args.raw_peft).merge_and_unload()

    if resize:
        model.resize_token_embeddings(resize_tokens)
        os.makedirs(os.path.join(output_dir, "embedding"), exist_ok=True)
        torch.save(model.embed_tokens, os.path.join(output_dir, "embedding", "emb.pth"))
        target_modules = model_args.target_modules
    else:
        target_modules = [t for t in model_args.target_modules if t != "embed_tokens"]

    if model_args.from_peft:
        if os.path.exists(os.path.join(model_args.from_peft, "embedding")):
            model.set_input_embeddings(torch.load(os.path.join(model_args.from_peft, "embedding", "emb.pth")))
            torch.save(model.embed_tokens, os.path.join(output_dir, "embedding", "emb.pth"))
        model = PeftModel.from_pretrained(model, model_args.from_peft, is_trainable=True)
    elif model_args.use_lora:
        peft_cfg = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=model_args.lora_rank,
            target_modules=target_modules,
            modules_to_save=model_args.modules_to_save,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_cfg)

    return model


def save_merged_model(model_args, output_dir: str):
    backbone_type = getattr(model_args, "backbone_type", "decoder_only")

    if backbone_type == "qwen3bi":
        if Qwen3BiModel is None:  # type: ignore
            raise ImportError("bidirectional_qwen3 is missing.")
        base_model = Qwen3BiModel.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            trust_remote_code=True,
        )
    else:
        base_model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            token=model_args.token,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
        )

    emb_path = os.path.join(output_dir, "embedding", "emb.pth")
    if os.path.exists(emb_path):
        base_model.set_input_embeddings(torch.load(emb_path))

    try:
        base_model = PeftModel.from_pretrained(base_model, output_dir).merge_and_unload()
    except Exception:
        base_model = PeftModel.from_pretrained(base_model, find_largest_checkpoint(output_dir)).merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    tokenizer.save_pretrained(os.path.join(output_dir, "merged_model"))
    base_model.config.vocab_size = len(tokenizer)
    base_model.save_pretrained(os.path.join(output_dir, "merged_model"))
