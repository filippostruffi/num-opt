from typing import Dict, Any
import torch
from torch import nn
from benchmark.tasks.base_task import BaseTask
from benchmark.metrics.ner_metrics import NERMetrics
from benchmark.config.registry import register_task


@register_task("ner")
class NERTask(BaseTask):
    @staticmethod
    def task_name() -> str:
        return "ner"

    @staticmethod
    def primary_metric_key() -> str:
        return "f1"

    def __init__(self):
        super().__init__()
        self.metric = NERMetrics()
        self.runtime_objects: Dict[str, Any] = {}

    def create_metric_accumulator(self) -> Dict[str, Any]:
        return self.metric.create_accumulator()

    def update_metrics(self, acc: Dict[str, Any], outputs: Any, targets: Any) -> None:
        emissions, mask = outputs
        # greedy decode (or CRF decode is done in model for eval)
        preds = emissions.argmax(dim=-1)
        self.metric.update(acc, preds.cpu(), targets.cpu(), mask.cpu())

    def compute_metrics(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        return self.metric.compute(acc)

    def collate_fn(self, batch):
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        tags = torch.stack([b["tags"] for b in batch], dim=0)
        lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
        batch_dict = {"input_ids": input_ids, "tags": tags, "lengths": lengths}
        if "char_ids" in batch[0]:
            # Expect shape: B x T x C
            char_ids = torch.stack([b["char_ids"] for b in batch], dim=0)
            batch_dict["char_ids"] = char_ids
        return batch_dict

    def train_step(self, model: nn.Module, batch: Dict[str, Any]):
        if "char_ids" in batch and hasattr(model, "char_cnn"):
            loss, emissions, mask = model(
                batch["input_ids"], batch["lengths"], char_ids=batch["char_ids"], tags=batch["tags"]
            )
        else:
            loss, emissions, mask = model(batch["input_ids"], batch["lengths"], tags=batch["tags"])
        # Ensure reported loss is never below zero (NLL minimum is 0)
        if loss is not None:
            loss = torch.clamp(loss, min=0.0)
        return loss, (emissions, mask), batch["tags"]

    def eval_step(self, model: nn.Module, batch: Dict[str, Any]):
        if "char_ids" in batch and hasattr(model, "char_cnn"):
            loss, emissions, mask = model(
                batch["input_ids"], batch["lengths"], char_ids=batch["char_ids"], tags=batch["tags"]
            )
        else:
            loss, emissions, mask = model(batch["input_ids"], batch["lengths"], tags=batch["tags"])
        # Ensure reported loss is never below zero (NLL minimum is 0)
        if loss is not None:
            loss = torch.clamp(loss, min=0.0)
        return loss, (emissions, mask), batch["tags"]


