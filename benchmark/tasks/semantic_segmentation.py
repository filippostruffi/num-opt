from typing import Dict, Any, Tuple
import torch
from torch import nn
from benchmark.tasks.base_task import BaseTask
from benchmark.metrics.segmentation_metrics import SegmentationMetrics
from benchmark.config.registry import register_task


@register_task("semantic_segmentation")
class SemanticSegmentationTask(BaseTask):
    @staticmethod
    def task_name() -> str:
        return "semantic_segmentation"

    @staticmethod
    def primary_metric_key() -> str:
        return "miou"

    def __init__(self):
        super().__init__()
        # VOC uses 0..20 classes and 255 as ignore
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.metric = SegmentationMetrics(num_classes=21, ignore_index=255)

    def create_metric_accumulator(self) -> Dict[str, Any]:
        return self.metric.create_accumulator()

    def update_metrics(self, acc: Dict[str, Any], outputs: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(outputs, dim=1)
        self.metric.update(acc, preds.cpu(), targets.cpu())

    def compute_metrics(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        return self.metric.compute(acc)

    def train_step(self, model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        out = model(x)
        logits = out["out"] if isinstance(out, dict) else out
        loss = self.criterion(logits, y)
        return loss, logits, y

    def eval_step(self, model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        out = model(x)
        logits = out["out"] if isinstance(out, dict) else out
        loss = self.criterion(logits, y)
        return loss, logits, y


