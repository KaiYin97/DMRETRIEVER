# DMRetriever/core/dataset.py
import os
import math
import random
import logging
import datasets
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer,
    DataCollatorWithPadding,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)

from .arguments import DataArguments

logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    """Concat multiple json/jsonl datasets. Optional KD."""

    def __init__(self, args: DataArguments, tokenizer: PreTrainedTokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.shuffle_ratio = args.shuffle_ratio

        dsets = []
        for data_dir in args.train_data:
            if not os.path.isdir(data_dir):
                if not (data_dir.endswith(".json") or data_dir.endswith(".jsonl")):
                    continue
                t = self._load_dataset(data_dir)
                if len(t) > 0:
                    dsets.append(t)
            else:
                for f in os.listdir(data_dir):
                    if not (f.endswith(".json") or f.endswith(".jsonl")):
                        continue
                    t = self._load_dataset(os.path.join(data_dir, f))
                    if len(t) > 0:
                        dsets.append(t)
        if not dsets:
            raise ValueError("No valid training dataset found.")
        self.dataset = datasets.concatenate_datasets(dsets)

    def _load_dataset(self, file_path: str):
        t = datasets.load_dataset("json", data_files=file_path, split="train", cache_dir=self.args.cache_path)
        if len(t) > self.args.max_example_num_per_dataset:
            t = t.shuffle().select(range(self.args.max_example_num_per_dataset))
        if not self.args.knowledge_distillation:
            for col in ["pos_scores", "neg_scores"]:
                if col in t.column_names:
                    t = t.remove_columns([col])
        else:
            if "pos_scores" not in t.column_names or "neg_scores" not in t.column_names:
                raise ValueError(f"KD=True but scores missing in {file_path}")
        return t

    def _shuffle_text(self, text: str):
        if self.shuffle_ratio > 0 and len(text) > 100 and random.random() < self.shuffle_ratio:
            chunks, step = [], len(text) // 3 + 1
            for i in range(0, len(text), step):
                chunks.append(text[i : i + step])
            random.shuffle(chunks)
            return " ".join(chunks)
        return text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        d = self.dataset[idx]
        g = self.args.train_group_size

        q = d["query"]
        if self.args.query_instruction_for_retrieval is not None:
            q = self.args.query_instruction_format.format(
                d.get("prompt", self.args.query_instruction_for_retrieval), q
            )

        passages, teacher_scores = [], []

        pos_idx = random.choice(list(range(len(d["pos"]))))
        passages.append(self._shuffle_text(d["pos"][pos_idx]))

        neg_all = list(range(len(d["neg"])))
        if len(d["neg"]) < g - 1:
            num = math.ceil((g - 1) / len(d["neg"]))
            neg_idxs = random.sample(neg_all * num, g - 1)
        else:
            neg_idxs = random.sample(neg_all, g - 1)
        for n in neg_idxs:
            passages.append(d["neg"][n])

        if self.args.knowledge_distillation:
            teacher_scores.append(d["pos_scores"][pos_idx])
            for n in neg_idxs:
                teacher_scores.append(d["neg_scores"][n])
            if not all(isinstance(s, (int, float)) for s in teacher_scores):
                raise ValueError("Non-numeric KD scores.")
        else:
            teacher_scores = None

        if self.args.passage_instruction_for_retrieval is not None:
            passages = [
                self.args.passage_instruction_format.format(self.args.passage_instruction_for_retrieval, p)
                for p in passages
            ]

        return q, passages, teacher_scores


@dataclass
class Collator(DataCollatorWithPadding):
    query_max_len: int = 32
    passage_max_len: int = 128
    sub_batch_size: int = -1
    default_no_in_batch_neg_flag: bool = False

    def __call__(self, features):
        queries = [f[0] for f in features]
        passages = [f[1] for f in features]
        teacher_scores = [f[2] for f in features]

        teacher_scores = None if teacher_scores[0] is None else sum(teacher_scores, [])
        queries = sum(queries, []) if isinstance(queries[0], list) else queries
        passages = sum(passages, []) if isinstance(passages[0], list) else passages

        q_inputs = self.tokenizer(queries, truncation=True, max_length=self.query_max_len, return_tensors=None)
        d_inputs = self.tokenizer(passages, truncation=True, max_length=self.passage_max_len, return_tensors=None)

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.tokenizer.pad(
                q_inputs, padding=self.padding, max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors
            )
            d_collated = self.tokenizer.pad(
                d_inputs, padding=self.padding, max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors
            )
        else:
            bs = self.sub_batch_size
            q_collated, d_collated = [], []
            for i in range(0, len(q_inputs["attention_mask"]), bs):
                sub = {k: v[i : min(len(q_inputs["attention_mask"]), i + bs)] for k, v in q_inputs.items()}
                q_collated.append(self.tokenizer.pad(
                    sub, padding=self.padding, max_length=self.query_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors
                ))
            for i in range(0, len(d_inputs["attention_mask"]), bs):
                sub = {k: v[i : min(len(d_inputs["attention_mask"]), i + bs)] for k, v in d_inputs.items()}
                d_collated.append(self.tokenizer.pad(
                    sub, padding=self.padding, max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors
                ))

        return {
            "queries": q_collated,
            "passages": d_collated,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": self.default_no_in_batch_neg_flag,
        }


class SameDatasetTrainDataset(TrainDataset):
    """Batch comes from the same dataset group."""

    def __init__(
        self,
        args: DataArguments,
        default_batch_size: int,
        seed: int,
        tokenizer: PreTrainedTokenizer,
        process_index: int = 0,
        num_processes: int = 1,
    ):
        import math as _math

        self.args = args
        self.shuffle_ratio = args.shuffle_ratio
        self.defaut_batch_size = default_batch_size
        self.tokenizer = tokenizer
        self.process_index = process_index
        self.num_processes = num_processes
        self.step = 0

        rng = np.random.default_rng(seed)

        dsets = []
        each_idxs, bs_idxs, flags = [], [], []
        cur = 0
        small_threshold, drop_threshold = args.small_threshold, args.drop_threshold

        for data_dir in args.train_data:
            if not os.path.isdir(data_dir):
                no_in_batch_neg_flag = data_dir.split(".")[-2].endswith("no_in_batch_neg")
                if not (data_dir.endswith(".json") or data_dir.endswith(".jsonl")):
                    continue
                t = self._load_dataset(data_dir)
                if len(t) == 0 or len(t) < small_threshold:
                    continue
                dsets.append(t)
                each_idxs.append(np.arange(len(t)) + cur)
                cur += len(t)
                bs_idxs.append(self._get_file_batch_size(t, default_batch_size))
                flags.append(no_in_batch_neg_flag)
            else:
                smalls, small_bs = [], _math.inf
                no_in_batch_neg_flag = data_dir.endswith("no_in_batch_neg")
                for f in os.listdir(data_dir):
                    if not (f.endswith(".json") or f.endswith(".jsonl")):
                        continue
                    t = self._load_dataset(os.path.join(data_dir, f))
                    if len(t) == 0:
                        continue
                    if len(t) < small_threshold:
                        smalls.append(t)
                        small_bs = min(small_bs, self._get_file_batch_size(t, default_batch_size))
                    else:
                        dsets.append(t)
                        each_idxs.append(np.arange(len(t)) + cur)
                        cur += len(t)
                        bs_idxs.append(self._get_file_batch_size(t, default_batch_size))
                        flags.append(no_in_batch_neg_flag)
                if smalls:
                    merged = datasets.concatenate_datasets(smalls)
                    if len(merged) >= drop_threshold:
                        dsets.append(merged)
                        each_idxs.append(np.arange(len(merged)) + cur)
                        cur += len(merged)
                        bs_idxs.append(small_bs)
                        flags.append(no_in_batch_neg_flag)

        if not dsets:
            raise ValueError("No valid training dataset found.")
        self.dataset = datasets.concatenate_datasets(dsets)
        self.each_data_idxs = each_idxs
        self.datasets_inxs = np.arange(len(each_idxs))
        self.batch_size_idxs = bs_idxs
        self.no_in_batch_neg_flags = flags

        self.rng = rng
        self.refresh_epoch()

    def _get_file_batch_size(self, temp_dataset: datasets.Dataset, default_batch_size: int):
        if "batch_size" in temp_dataset.column_names:
            return temp_dataset["batch_size"][0]
        if "type" in temp_dataset.column_names and "symmetric" in temp_dataset["type"][0]:
            return default_batch_size // 2
        return default_batch_size

    def refresh_epoch(self):
        self.rng.shuffle(self.datasets_inxs)
        batch_datas = []
        for ds_idx in self.datasets_inxs:
            self.rng.shuffle(self.each_data_idxs[ds_idx])
            cur_bs = self.batch_size_idxs[ds_idx] * self.num_processes
            flag = self.no_in_batch_neg_flags[ds_idx]
            for start in range(0, len(self.each_data_idxs[ds_idx]), cur_bs):
                if len(self.each_data_idxs[ds_idx]) - start < cur_bs:
                    break
                batch_datas.append((self.each_data_idxs[ds_idx][start : start + cur_bs], flag))
        self.rng.shuffle(batch_datas)
        self.batch_datas = batch_datas
        self.step = 0

    def __len__(self):
        return len(self.batch_datas) * self.num_processes

    def _get_train_group_size(self, batch_raw):
        if "type" in batch_raw:
            t = batch_raw["type"][0]
            if t in ["only_1neg"]:
                return 2, t
            if t in ["symmetric_class"]:
                return min(len(batch_raw["neg"][0]) + 1, self.args.train_group_size), t
            return self.args.train_group_size, t
        if "train_group_size" in batch_raw:
            v = batch_raw["train_group_size"][0]
            if isinstance(v, int) and v > 0:
                return v, None
        return self.args.train_group_size, None

    def __getitem__(self, _):
        batch_indices, no_in_batch_neg_flag = self.batch_datas[self.step]
        cur_bs = int(len(batch_indices) / self.num_processes)
        batch_indices = batch_indices[self.process_index * cur_bs : (self.process_index + 1) * cur_bs]
        batch = self.dataset[batch_indices]
        self.step += 1

        queries, passages, teacher_scores = [], [], []
        gsize, dtype = self._get_train_group_size(batch)

        for i in range(len(batch["query"])):
            if dtype is not None:
                assert batch["type"][i] == dtype
            q = self.args.query_instruction_format.format(
                batch["prompt"][i] if "prompt" in batch else self.args.query_instruction_for_retrieval,
                batch["query"][i],
            )
            queries.append(q)

            tmp = []
            pos_idx = random.choice(list(range(len(batch["pos"][i]))))
            tmp.append(self._shuffle_text(batch["pos"][i][pos_idx]))

            neg_all = list(range(len(batch["neg"][i])))
            if len(batch["neg"][i]) < gsize - 1:
                num = math.ceil((gsize - 1) / len(batch["neg"][i]))
                negs = random.sample(neg_all * num, gsize - 1)
            else:
                negs = random.sample(neg_all, gsize - 1)
            for n in negs:
                tmp.append(batch["neg"][i][n])

            if self.args.knowledge_distillation:
                if "pos_scores" in batch and batch["pos_scores"][i] is not None:
                    teacher_scores.append(batch["pos_scores"][i][pos_idx])
                for n in negs:
                    if "neg_scores" in batch and batch["neg_scores"][i] is not None:
                        teacher_scores.append(batch["neg_scores"][i][n])
            else:
                teacher_scores = None

            if dtype in ["symmetric_sts", "symmetric_clustering"]:
                tmp = [
                    self.args.query_instruction_format.format(
                        batch["prompt"][i] if "prompt" in batch else self.args.query_instruction_for_retrieval, p
                    )
                    for p in tmp
                ]
            elif self.args.passage_instruction_for_retrieval is not None:
                tmp = [
                    self.args.passage_instruction_format.format(self.args.passage_instruction_for_retrieval, p)
                    for p in tmp
                ]

            passages.extend(tmp)

            if teacher_scores is not None and len(teacher_scores) > 0 and len(passages) > 0:
                assert len(teacher_scores) == len(passages)

        return queries, passages, teacher_scores, no_in_batch_neg_flag


@dataclass
class SameDatasetCollator(DataCollatorWithPadding):
    """Used with SameDatasetTrainDataset (per_device_train_batch_size must be 1)."""

    query_max_len: int = 32
    passage_max_len: int = 128
    sub_batch_size: int = -1
    default_no_in_batch_neg_flag: bool = False

    def __call__(self, features):
        queries, passages, teacher_scores, no_in_batch_neg_flag = features[0]
        if self.default_no_in_batch_neg_flag:
            no_in_batch_neg_flag = True

        q_inputs = self.tokenizer(queries, truncation=True, max_length=self.query_max_len, return_tensors=None)
        d_inputs = self.tokenizer(passages, truncation=True, max_length=self.passage_max_len, return_tensors=None)

        if self.sub_batch_size is None or self.sub_batch_size <= 0:
            q_collated = self.tokenizer.pad(
                q_inputs, padding=self.padding, max_length=self.query_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors
            )
            d_collated = self.tokenizer.pad(
                d_inputs, padding=self.padding, max_length=self.passage_max_len,
                pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors
            )
        else:
            bs = self.sub_batch_size
            q_collated, d_collated = [], []
            for i in range(0, len(q_inputs["attention_mask"]), bs):
                sub = {k: v[i : min(len(q_inputs["attention_mask"]), i + bs)] for k, v in q_inputs.items()}
                q_collated.append(self.tokenizer.pad(
                    sub, padding=self.padding, max_length=self.query_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors
                ))
            for i in range(0, len(d_inputs["attention_mask"]), bs):
                sub = {k: v[i : min(len(d_inputs["attention_mask"]), i + bs)] for k, v in d_inputs.items()}
                d_collated.append(self.tokenizer.pad(
                    sub, padding=self.padding, max_length=self.passage_max_len,
                    pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=self.return_tensors
                ))

        if isinstance(teacher_scores, list) and len(teacher_scores) == 0:
            teacher_scores = None

        return {
            "queries": q_collated,
            "passages": d_collated,
            "teacher_scores": teacher_scores,
            "no_in_batch_neg_flag": no_in_batch_neg_flag,
        }


class TrainerCallbackForDataRefresh(TrainerCallback):
    def __init__(self, train_dataset: SameDatasetTrainDataset):
        self.train_dataset = train_dataset

    def on_epoch_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        self.train_dataset.refresh_epoch()
