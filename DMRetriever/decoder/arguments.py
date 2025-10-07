# DMRetriever/decoder/arguments.py
from typing import Optional, List
from dataclasses import dataclass, field
from DMRetriever.core import ModelArguments


def default_target_modules() -> List[str]:
    return ["v_proj", "q_proj", "k_proj", "gate_proj", "down_proj", "o_proj", "up_proj"]


@dataclass
class DecoderModelArguments(ModelArguments):
    backbone_type: str = field(default="decoder_only", metadata={"choices": ["decoder_only", "qwen3bi"]})

    peft_model_path: str = field(default="")

    use_lora: bool = field(default=True)
    lora_rank: int = field(default=64)
    lora_alpha: float = field(default=16.0)
    lora_dropout: float = field(default=0.1)
    target_modules: List[str] = field(default_factory=default_target_modules)

    use_flash_attn: bool = field(default=False)
    use_slow_tokenizer: bool = field(default=False)

    from_peft: Optional[str] = field(default=None)
    modules_to_save: Optional[str] = field(default=None)
    raw_peft: Optional[str] = field(default=None)
    additional_special_tokens: Optional[List[str]] = field(default=None, metadata={"nargs": "+"})
    save_merged_lora_model: bool = field(default=False)
    only_merge_lora_model: bool = field(default=False)

    mlp_out_dim: Optional[int] = field(default=None)
    mlp_hidden_dim: Optional[int] = field(default=None)
