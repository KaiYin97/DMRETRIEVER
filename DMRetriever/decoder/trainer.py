# DMRetriever/decoder/trainer.py
import os
import torch
from typing import Optional
from DMRetriever.core import BaseEmbedderTrainer


class DecoderTrainer(BaseEmbedderTrainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if not hasattr(self.model, "save"):
            raise NotImplementedError(f"Model {self.model.__class__.__name__} lacks `save`.")
        self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
