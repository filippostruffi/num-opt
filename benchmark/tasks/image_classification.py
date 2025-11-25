from typing import Dict, Any, Tuple
import torch
from torch import nn
from benchmark.tasks.base_task import BaseTask
from benchmark.metrics.classification import ClassificationMetrics
from benchmark.config.registry import register_task


@register_task("image_classification")
class ImageClassificationTask(BaseTask):
    @staticmethod
    def task_name() -> str:
        return "image_classification"

    @staticmethod
    def primary_metric_key() -> str:
        return "accuracy"

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.metric = ClassificationMetrics()

    def create_metric_accumulator(self) -> Dict[str, Any]:
        return self.metric.create_accumulator()

    def update_metrics(self, acc: Dict[str, Any], outputs: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(outputs, dim=1)
        self.metric.update(acc, preds.cpu(), targets.cpu())

    def compute_metrics(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        return self.metric.compute(acc)

    def train_step(self, model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        logits = model(x)
        loss = self.criterion(logits, y)
        return loss, logits, y

    def eval_step(self, model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        logits = model(x)
        loss = self.criterion(logits, y)
        return loss, logits, y


