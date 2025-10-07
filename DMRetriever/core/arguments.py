# DMRetriever/core/arguments.py
import os
from typing import Optional
from dataclasses import dataclass, field
from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "HF model path."})
    config_name: Optional[str] = field(default=None, metadata={"help": "HF config path."})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "HF tokenizer path."})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache dir."})
    trust_remote_code: bool = field(default=False, metadata={"help": "HF trust_remote_code."})
    token: Optional[str] = field(default_factory=lambda: os.getenv("HF_TOKEN", None), metadata={"help": "HF token."})


@dataclass
class DataArguments:
    train_data: list = field(default=None, metadata={"help": "One or more json/jsonl paths.", "nargs": "+"})
    cache_path: Optional[str] = field(default=None, metadata={"help": "Datasets cache dir."})
    train_group_size: int = field(default=8)
    query_max_len: int = field(default=32)
    passage_max_len: int = field(default=128)
    pad_to_multiple_of: Optional[int] = field(default=None)

    max_example_num_per_dataset: int = field(default=100000000)

    query_instruction_for_retrieval: Optional[str] = field(default=None)
    query_instruction_format: str = field(default="{}{}")

    knowledge_distillation: bool = field(default=True)

    passage_instruction_for_retrieval: Optional[str] = field(default=None)
    passage_instruction_format: str = field(default="{}{}")

    shuffle_ratio: float = field(default=0.0)

    same_dataset_within_batch: bool = field(default=False)
    small_threshold: int = field(default=0)
    drop_threshold: int = field(default=0)

    def __post_init__(self):
        if self.query_instruction_format and "\\n" in self.query_instruction_format:
            self.query_instruction_format = self.query_instruction_format.replace("\\n", "\n")
        if self.passage_instruction_format and "\\n" in self.passage_instruction_format:
            self.passage_instruction_format = self.passage_instruction_format.replace("\\n", "\n")
        if not self.train_data or len(self.train_data) == 0:
            raise ValueError("`train_data` must be provided.")
        for p in self.train_data:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Not found: {p}")


@dataclass
class TrainingArguments(HFTrainingArguments):
    negatives_cross_device: bool = field(default=False)
    temperature: Optional[float] = field(default=0.02)
    fix_position_embedding: bool = field(default=False)
    sentence_pooling_method: str = field(default="cls", metadata={"choices": ["cls", "mean", "last_token"]})
    normalize_embeddings: bool = field(default=True)
    sub_batch_size: Optional[int] = field(default=None)
    kd_loss_type: str = field(default="kl_div", metadata={"choices": ["kl_div", "m3_kd_loss"]})
    distill_loss_weight: float = field(default=1.0)
    no_in_batch_neg_flag: bool = field(default=False)
