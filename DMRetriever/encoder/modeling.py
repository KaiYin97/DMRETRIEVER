# DMRetriever/encoder/modeling.py
import torch
from transformers import AutoModel, PreTrainedModel, PreTrainedTokenizer
from DMRetriever.core import BaseEmbedderModel


class BiEncoderOnlyEmbedderModel(BaseEmbedderModel):
    """Bi-encoder for encoder-only backbones."""

    TRANSFORMER_CLS = AutoModel

    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer = None,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        sub_batch_size: int = -1,
        kd_loss_type: str = "kl_div",
        kd_loss_weight: float = 1.0,
        sentence_pooling_method: str = "cls",
        normalize_embeddings: bool = False,
    ):
        super().__init__(
            base_model,
            tokenizer=tokenizer,
            negatives_cross_device=negatives_cross_device,
            temperature=temperature,
            sub_batch_size=sub_batch_size,
            kd_loss_type=kd_loss_type,
            kd_loss_weight=kd_loss_weight,
        )
        self.sentence_pooling_method = sentence_pooling_method
        self.normalize_embeddings = normalize_embeddings
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")

    def encode(self, features):
        if features is None:
            return None
        if not isinstance(features, list):
            if self.sub_batch_size is not None and self.sub_batch_size > 0:
                reps = []
                for i in range(0, len(features["attention_mask"]), self.sub_batch_size):
                    sub = {k: v[i : min(len(features["attention_mask"]), i + self.sub_batch_size)] for k, v in features.items()}
                    last = self.model(**sub, return_dict=True).last_hidden_state
                    reps.append(self._sentence_embedding(last, sub["attention_mask"]))
                out = torch.cat(reps, 0).contiguous()
            else:
                last = self.model(**features, return_dict=True).last_hidden_state
                out = self._sentence_embedding(last, features["attention_mask"])
        else:
            reps = []
            for sub in features:
                last = self.model(**sub, return_dict=True).last_hidden_state
                reps.append(self._sentence_embedding(last, sub["attention_mask"]))
            out = torch.cat(reps, 0).contiguous()

        if self.normalize_embeddings:
            out = torch.nn.functional.normalize(out, dim=-1)
        return out.contiguous()

    def _sentence_embedding(self, last_hidden_state, attention_mask):
        if self.sentence_pooling_method == "cls":
            return last_hidden_state[:, 0]
        if self.sentence_pooling_method == "mean":
            s = torch.sum(last_hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return s / d
        if self.sentence_pooling_method == "last_token":
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                return last_hidden_state[:, -1]
            seq = attention_mask.sum(dim=1) - 1
            bsz = last_hidden_state.shape[0]
            return last_hidden_state[torch.arange(bsz, device=last_hidden_state.device), seq]
        raise NotImplementedError(f"Unknown pooling: {self.sentence_pooling_method}")

    def compute_score(self, q_reps, p_reps):
        return (self._sim(q_reps, p_reps) / self.temperature).view(q_reps.size(0), -1)

    @staticmethod
    def _sim(q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1)) if len(p_reps.size()) == 2 else torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        self.model.save_pretrained(output_dir, state_dict=type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()}))
