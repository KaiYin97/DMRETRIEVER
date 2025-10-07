# DMRetriever/bidirectional_qwen3/bidirectional_qwen3.py
from __future__ import annotations
from typing import Optional, Any
import torch
from torch import nn
from transformers.cache_utils import Cache  # noqa: F401
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3Model,
    Qwen3ForCausalLM,
    Qwen3PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

try:
    from peft import PeftModel  # type: ignore
except ImportError:
    PeftModel = Any  # type: ignore

logger = logging.get_logger(__name__)


class ModifiedQwen3Attention(Qwen3Attention):
    """Full-context attention (no causal mask, no SWA)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_causal = False
        self.sliding_window = None


class ModifiedQwen3DecoderLayer(Qwen3DecoderLayer):
    """Decoder layer using full-context attention."""
    def __init__(self, config: PretrainedConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = ModifiedQwen3Attention(config=config, layer_idx=layer_idx)
        self.attention_type = "full_attention"
        self.sliding_window = None


class Qwen3BiModel(Qwen3Model):
    """Qwen-3 backbone with bidirectional self-attention."""
    _no_split_modules = ["ModifiedQwen3DecoderLayer"]

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([ModifiedQwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.has_sliding_layers = False

    @staticmethod
    def _build_pad_bias(pad_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        # Convert padding mask [B,L] to additive bias [B,1,1,L]
        neg_inf = torch.finfo(dtype).min
        bias = (~pad_mask.bool()).to(dtype) * neg_inf
        return bias[:, None, None, :]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if attention_mask is None:
            if input_ids is None:
                raise ValueError("Either attention_mask or input_ids must be provided.")
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        pad_bias = self._build_pad_bias(attention_mask, self.embed_tokens.weight.dtype)
        attn_mask_dict = {"full_attention": pad_bias}
        return super().forward(input_ids=input_ids, attention_mask=attn_mask_dict, **kwargs)


class Qwen3BiForMLM(Qwen3ForCausalLM):
    """Bidirectional Qwen-3 with LM head for masked token tasks."""
    def __init__(self, config: PretrainedConfig):
        Qwen3PreTrainedModel.__init__(self, config)
        self.model = Qwen3BiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def generate(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError("generate() is disabled for bidirectional backbone.")

    def get_model_for_peft(self):
        return self.model

    def set_model_for_peft(self, model: PeftModel):  # type: ignore[override]
        self.model = model

    def save_peft_model(self, path: str):
        if isinstance(self.model, PeftModel):  # type: ignore[arg-type]
            self.model.save_pretrained(path)
        else:
            raise ValueError("Backbone is not a PEFT model.")
