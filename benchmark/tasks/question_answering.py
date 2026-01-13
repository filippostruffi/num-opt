from typing import Dict, Any
import torch
from torch import nn
from benchmark.tasks.base_task import BaseTask
from benchmark.metrics.qa_metrics import QAMetrics
from benchmark.config.registry import register_task


@register_task("question_answering")
class QATask(BaseTask):
    @staticmethod
    def task_name() -> str:
        return "question_answering"

    @staticmethod
    def primary_metric_key() -> str:
        return "f1"

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.metric = QAMetrics()
        self.runtime_objects: Dict[str, Any] = {}

    def create_metric_accumulator(self) -> Dict[str, Any]:
        return self.metric.create_accumulator()

    def update_metrics(self, acc: Dict[str, Any], outputs: Any, targets: Any) -> None:
        start_logits, end_logits = outputs
        starts = targets[0]
        ends = targets[1]
        start_pred = start_logits.argmax(dim=-1)
        end_pred = end_logits.argmax(dim=-1)
        self.metric.update(acc, start_pred.cpu(), end_pred.cpu(), starts.cpu(), ends.cpu())

    def compute_metrics(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        return self.metric.compute(acc)

    def collate_fn(self, batch):
        ctx = torch.stack([b["context_ids"] for b in batch], dim=0)
        q = torch.stack([b["question_ids"] for b in batch], dim=0)
        starts = torch.stack([b["start"] for b in batch], dim=0)
        ends = torch.stack([b["end"] for b in batch], dim=0)
        return {"context_ids": ctx, "question_ids": q, "starts": starts, "ends": ends}

    def train_step(self, model: nn.Module, batch: Dict[str, Any]):
        start_logits, end_logits = model(batch["context_ids"], batch["question_ids"])
        loss = self.criterion(start_logits, batch["starts"]) + self.criterion(end_logits, batch["ends"])
        return loss, (start_logits, end_logits), (batch["starts"], batch["ends"])

    def eval_step(self, model: nn.Module, batch: Dict[str, Any]):
        start_logits, end_logits = model(batch["context_ids"], batch["question_ids"])
        loss = self.criterion(start_logits, batch["starts"]) + self.criterion(end_logits, batch["ends"])
        return loss, (start_logits, end_logits), (batch["starts"], batch["ends"])


