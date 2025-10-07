# DMRetriever/decoder/runner.py
import logging
from typing import Tuple
from pathlib import Path
from transformers import AutoTokenizer, PreTrainedTokenizer, set_seed
from DMRetriever.core import (
    DataArguments,
    TrainingArguments,
    TrainDataset,
    SameDatasetTrainDataset,
    Collator,
    SameDatasetCollator,
    TrainerCallbackForDataRefresh,
)
from .arguments import DecoderModelArguments
from .trainer import DecoderTrainer
from .modeling import BiDecoderOnlyEmbedderModel
from .load_model import get_model, save_merged_model

logger = logging.getLogger(__name__)


class DecoderRunner:
    """Decoder-only / qwen3bi training entry."""

    def __init__(self, model_args: DecoderModelArguments, data_args: DataArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        set_seed(training_args.seed)

        self.tokenizer, self.model = self._load_tokenizer_and_model()
        self.train_dataset = self._load_train_dataset()
        self.data_collator = self._load_collator()
        self.trainer = self._load_trainer()

    def _load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, BiDecoderOnlyEmbedderModel]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name or self.model_args.model_name_or_path,
            token=self.model_args.token,
            cache_dir=self.model_args.cache_dir,
            use_fast=not self.model_args.use_slow_tokenizer,
            add_eos_token=True,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.unk_token_id if tokenizer.unk_token else tokenizer.eos_token_id
        tokenizer.padding_side = "right"

        resize = False
        if self.model_args.additional_special_tokens:
            added = tokenizer.add_special_tokens({"additional_special_tokens": self.model_args.additional_special_tokens})
            if added > 0:
                resize = True

        backbone_type = self.model_args.backbone_type if hasattr(self.model_args, "backbone_type") else (
            "qwen3bi" if "qwen3bi" in self.model_args.model_name_or_path.lower() else "decoder_only"
        )

        base_model = get_model(
            self.model_args,
            self.training_args.output_dir,
            resize,
            len(tokenizer),
            backbone_type=backbone_type,
        )

        pooling_default = "mean" if backbone_type == "qwen3bi" else "last_token"
        chosen_pooling = self.training_args.sentence_pooling_method or pooling_default

        model = BiDecoderOnlyEmbedderModel(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            sentence_pooling_method=chosen_pooling,
            normalize_embeddings=self.training_args.normalize_embeddings,
        )

        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        if self.training_args.fix_position_embedding:
            for name, param in model.named_parameters():
                if "position_embeddings" in name:
                    param.requires_grad = False
        return tokenizer, model

    def _load_train_dataset(self):
        if self.data_args.same_dataset_within_batch:
            ds = SameDatasetTrainDataset(
                args=self.data_args,
                default_batch_size=self.training_args.per_device_train_batch_size,
                seed=self.training_args.seed,
                tokenizer=self.tokenizer,
                process_index=self.training_args.process_index,
                num_processes=self.training_args.world_size,
            )
            self.training_args.per_device_train_batch_size = 1
            self.training_args.dataloader_num_workers = 0
        else:
            ds = TrainDataset(args=self.data_args, tokenizer=self.tokenizer)
        return ds

    def _load_collator(self):
        Coll = SameDatasetCollator if self.data_args.same_dataset_within_batch else Collator
        return Coll(
            tokenizer=self.tokenizer,
            query_max_len=self.data_args.query_max_len,
            passage_max_len=self.data_args.passage_max_len,
            sub_batch_size=self.training_args.sub_batch_size,
            pad_to_multiple_of=self.data_args.pad_to_multiple_of,
            padding=True,
            return_tensors="pt",
            default_no_in_batch_neg_flag=self.training_args.no_in_batch_neg_flag,
        )

    def _load_trainer(self):
        trainer = DecoderTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        if self.data_args.same_dataset_within_batch:
            trainer.add_callback(TrainerCallbackForDataRefresh(self.train_dataset))
        return trainer

    def run(self):
        if not self.model_args.only_merge_lora_model:
            Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)
            self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
            self.trainer.save_model()

        if self.model_args.save_merged_lora_model and self.training_args.process_index == 0:
            save_merged_model(self.model_args, self.training_args.output_dir)
