# DMRetriever/encoder/runner.py
import os
import logging
from pathlib import Path
from typing import Tuple
from transformers import set_seed, AutoTokenizer, AutoModel, AutoConfig, PreTrainedTokenizer
from DMRetriever.core import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    BaseEmbedderTrainer,
    TrainDataset,
    SameDatasetTrainDataset,
    Collator,
    SameDatasetCollator,
    TrainerCallbackForDataRefresh,
)
from .modeling import BiEncoderOnlyEmbedderModel
from .trainer import EncoderTrainer

logger = logging.getLogger(__name__)


class EncoderRunner:
    """Encoder-only training entry."""

    def __init__(self, model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
        ):
            raise ValueError(f"Output dir ({training_args.output_dir}) exists and is not empty. Use --overwrite_output_dir.")

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )
        set_seed(training_args.seed)

        self.tokenizer, self.model = self._load_tokenizer_and_model()
        self.train_dataset = self._load_train_dataset()
        self.data_collator = self._load_collator()
        self.trainer = self._load_trainer()

    def _load_tokenizer_and_model(self) -> Tuple[PreTrainedTokenizer, BiEncoderOnlyEmbedderModel]:
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        base_model = AutoModel.from_pretrained(
            self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )
        _ = AutoConfig.from_pretrained(
            self.model_args.config_name or self.model_args.model_name_or_path,
            num_labels=1,
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.token,
            trust_remote_code=self.model_args.trust_remote_code,
        )

        model = BiEncoderOnlyEmbedderModel(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=self.training_args.negatives_cross_device,
            temperature=self.training_args.temperature,
            sub_batch_size=self.training_args.sub_batch_size,
            kd_loss_type=self.training_args.kd_loss_type,
            kd_loss_weight=self.training_args.distill_loss_weight,
            sentence_pooling_method=self.training_args.sentence_pooling_method,
            normalize_embeddings=self.training_args.normalize_embeddings,
        )
        if self.training_args.gradient_checkpointing:
            model.enable_input_require_grads()
        if self.training_args.fix_position_embedding:
            for n, p in model.named_parameters():
                if "position_embeddings" in n:
                    p.requires_grad = False
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

    def _load_trainer(self) -> BaseEmbedderTrainer:
        trainer = EncoderTrainer(
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
        Path(self.training_args.output_dir).mkdir(parents=True, exist_ok=True)
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)
        self.trainer.save_model()
