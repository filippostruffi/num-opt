from typing import Dict, Any, Tuple
import torch
from torch import nn
from benchmark.tasks.base_task import BaseTask
from benchmark.metrics.translation_metrics import TranslationMetrics
from benchmark.config.registry import register_task


def pad_seq(seqs, pad_id=0):
    max_len = max(s.size(0) for s in seqs)
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.size(0)] = s
    return out


@register_task("machine_translation")
class MachineTranslationTask(BaseTask):
    @staticmethod
    def task_name() -> str:
        return "machine_translation"

    @staticmethod
    def primary_metric_key() -> str:
        return "bleu"

    def __init__(self):
        super().__init__()
        try:
            self.criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        except TypeError: self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.metric = TranslationMetrics()
        self.runtime_objects: Dict[str, Any] = {}

    def create_metric_accumulator(self) -> Dict[str, Any]:
        return self.metric.create_accumulator()

    def update_metrics(self, acc: Dict[str, Any], outputs: torch.Tensor, targets: torch.Tensor) -> None:
        # outputs: BxTxV, targets: BxT
        preds = outputs.argmax(dim=-1)
        self.metric.update_ids(acc, preds.cpu(), targets.cpu())

    def compute_metrics(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        # Use simplistic detokenization
        src_vocab = self.runtime_objects.get("src_vocab", None)
        tgt_vocab = self.runtime_objects.get("tgt_vocab", None)
        return self.metric.compute(acc, tgt_vocab=tgt_vocab)

    def collate_fn(self, batch):
        src = [b["src"] for b in batch]
        tgt = [b["tgt"] for b in batch]
        src_pad = pad_seq(src, pad_id=0)
        # Teacher forcing: input is tgt[:-1], target is tgt[1:]
        tgt_pad = pad_seq(tgt, pad_id=0)
        tgt_in = tgt_pad[:, :-1].contiguous()
        tgt_out = tgt_pad[:, 1:].contiguous()
        return {"src": src_pad, "tgt_in": tgt_in, "tgt_out": tgt_out}

    def train_step(self, model: nn.Module, batch: Dict[str, Any]):
        logits = model(batch["src"], batch["tgt_in"])
        B, T, V = logits.shape
        loss = self.criterion(logits.view(B * T, V), batch["tgt_out"].view(B * T))
        return loss, logits, batch["tgt_out"]

    def eval_step(self, model: nn.Module, batch: Dict[str, Any]):
        logits = model(batch["src"], batch["tgt_in"])
        B, T, V = logits.shape
        loss = self.criterion(logits.view(B * T, V), batch["tgt_out"].view(B * T))
        return loss, logits, batch["tgt_out"]


