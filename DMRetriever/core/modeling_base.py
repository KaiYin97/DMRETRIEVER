# DMRetriever/core/modeling_base.py
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
from transformers import PreTrainedTokenizer
from transformers.file_utils import ModelOutput
from dataclasses import dataclass
from typing import Dict, Optional, List, Union, Callable


@dataclass
class EmbedderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BaseEmbedderModel(nn.Module):
    """Concrete base with KD/in-batch/cross-device logic. Subclasses implement encode/compute_score/save."""

    def __init__(
        self,
        base_model,
        tokenizer: PreTrainedTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        sub_batch_size: int = -1,
        kd_loss_type: str = "kl_div",
        kd_loss_weight: float = 1.0,
    ):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError("Distributed not initialized.")
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        self.sub_batch_size = sub_batch_size
        self.kd_loss_type = kd_loss_type
        self.kd_loss_weight = kd_loss_weight

    @staticmethod
    def _get_local_score(q, p, all_scores):
        g = p.size(0) / q.size(0)
        g = int(g)
        idx = torch.arange(0, q.size(0), device=q.device) * g
        cols = [all_scores[torch.arange(q.size(0), device=q.device), idx + i] for i in range(g)]
        return torch.stack(cols, dim=1).view(q.size(0), -1)

    def _compute_local_score(self, q, p, compute_score_func: Optional[Callable] = None, **kwargs):
        s = self.compute_score(q, p) if compute_score_func is None else compute_score_func(q, p, **kwargs)
        return self._get_local_score(q, p, s)

    def _compute_no_in_batch_neg_loss(self, q, p, teacher_targets=None, compute_score_func=None, **kwargs):
        local_scores = self._compute_local_score(q, p, compute_score_func, **kwargs)
        if teacher_targets is not None:
            if self.kd_loss_type == "kl_div":
                kd = self.distill_loss(self.kd_loss_type, teacher_targets, local_scores, group_size=local_scores.size(1))
                local_targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long)
                ce = self.compute_loss(local_scores, local_targets)
                loss = ce + self.kd_loss_weight * kd
            else:
                kd = self.distill_loss(self.kd_loss_type, teacher_targets, local_scores, group_size=local_scores.size(1))
                loss = self.kd_loss_weight * kd
        else:
            local_targets = torch.zeros(local_scores.size(0), device=local_scores.device, dtype=torch.long)
            loss = self.compute_loss(local_scores, local_targets)
        return local_scores, loss

    def _compute_in_batch_neg_loss(self, q, p, teacher_targets=None, compute_score_func=None, **kwargs):
        g = int(p.size(0) / q.size(0))
        scores = self.compute_score(q, p) if compute_score_func is None else compute_score_func(q, p, **kwargs)
        if teacher_targets is not None:
            if self.kd_loss_type == "kl_div":
                student = self._get_local_score(q, p, scores)
                kd = self.distill_loss(self.kd_loss_type, teacher_targets, student, g)
                idx = torch.arange(q.size(0), device=q.device, dtype=torch.long)
                targets = idx * g
                ce = self.compute_loss(scores, targets)
                loss = ce + self.kd_loss_weight * kd
            elif self.kd_loss_type == "m3_kd_loss":
                kd = self.distill_loss(self.kd_loss_type, teacher_targets, scores, g)
                loss = self.kd_loss_weight * kd
            else:
                raise ValueError(f"Invalid kd_loss_type: {self.kd_loss_type}")
        else:
            idx = torch.arange(q.size(0), device=q.device, dtype=torch.long)
            targets = idx * g
            loss = self.compute_loss(scores, targets)
        return scores, loss

    def _compute_cross_device_neg_loss(self, q, p, teacher_targets=None, compute_score_func=None, **kwargs):
        g = int(p.size(0) / q.size(0))
        cross_q = self._dist_gather_tensor(q)
        cross_p = self._dist_gather_tensor(p)
        cross_scores = self.compute_score(cross_q, cross_p) if compute_score_func is None else compute_score_func(cross_q, cross_p, **kwargs)

        if teacher_targets is not None:
            if self.kd_loss_type == "kl_div":
                student = self._get_local_score(cross_q, cross_p, cross_scores)
                student = student[q.size(0) * self.process_rank : q.size(0) * (self.process_rank + 1)]
                kd = self.distill_loss(self.kd_loss_type, teacher_targets, student, g)
                idx = torch.arange(cross_q.size(0), device=cross_q.device, dtype=torch.long)
                targets = idx * g
                ce = self.compute_loss(cross_scores, targets)
                loss = ce + self.kd_loss_weight * kd
            elif self.kd_loss_type == "m3_kd_loss":
                cross_teacher = self._dist_gather_tensor(teacher_targets)
                kd = self.distill_loss(self.kd_loss_type, cross_teacher, cross_scores, g)
                loss = self.kd_loss_weight * kd
            else:
                raise ValueError(f"Invalid kd_loss_type: {self.kd_loss_type}")
        else:
            idx = torch.arange(cross_q.size(0), device=cross_q.device, dtype=torch.long)
            targets = idx * g
            loss = self.compute_loss(cross_scores, targets)
        return cross_scores, loss

    def forward(
        self,
        queries: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        passages: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None,
        teacher_scores: Union[None, List[float]] = None,
        no_in_batch_neg_flag: bool = False,
        compute_score_func: Optional[Callable] = None,
        **kwargs,
    ):
        q_reps = self.encode(queries)
        p_reps = self.encode(passages)

        if self.training:
            if teacher_scores is not None:
                teacher_scores = torch.tensor(teacher_scores, device=q_reps.device).view(q_reps.size(0), -1).detach()
                teacher_targets = F.softmax(teacher_scores, dim=-1)
            else:
                teacher_targets = None
            compute_loss_func = self._compute_no_in_batch_neg_loss if no_in_batch_neg_flag else (
                self._compute_cross_device_neg_loss if self.negatives_cross_device else self._compute_in_batch_neg_loss
            )
            scores, loss = compute_loss_func(q_reps, p_reps, teacher_targets=teacher_targets, compute_score_func=compute_score_func)
        else:
            loss, scores = None, None

        return EmbedderOutput(loss=loss, q_reps=q_reps, p_reps=p_reps, scores=scores)

    @staticmethod
    def distill_loss(kd_loss_type, teacher_targets, student_scores, group_size=None):
        if kd_loss_type == "kl_div":
            return -torch.mean(torch.sum(torch.log_softmax(student_scores, dim=-1) * teacher_targets, dim=-1))
        elif kd_loss_type == "m3_kd_loss":
            labels = torch.arange(student_scores.size(0), device=student_scores.device, dtype=torch.long) * group_size
            loss = 0
            mask = torch.zeros_like(student_scores)
            for i in range(group_size):
                tgt = labels + i
                sc = student_scores + mask
                l = torch.nn.functional.cross_entropy(sc, tgt, reduction="none")
                loss += torch.mean(teacher_targets[:, i] * l)
                mask = torch.scatter(mask, dim=-1, index=tgt.unsqueeze(-1), value=torch.finfo(student_scores.dtype).min)
            return loss
        else:
            raise ValueError(f"Invalid kd_loss_type: {kd_loss_type}")

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()
        all_t = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_t, t)
        all_t[self.process_rank] = t
        return torch.cat(all_t, dim=0)

    # to implement in subclasses
    def encode(self, features):  # pragma: no cover
        raise NotImplementedError

    def compute_loss(self, scores, target):  # pragma: no cover
        raise NotImplementedError

    def compute_score(self, q_reps, p_reps):  # pragma: no cover
        raise NotImplementedError

    def save(self, output_dir: str):  # pragma: no cover
        raise NotImplementedError
