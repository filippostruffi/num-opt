from typing import Dict, Any, Tuple
import torch
from torch import nn
from benchmark.tasks.base_task import BaseTask
from benchmark.metrics.generation_metrics import GenerationMetrics
from benchmark.config.registry import register_task


@register_task("text_generation")
class TextGenerationTask(BaseTask):
    @staticmethod
    def task_name() -> str:
        return "text_generation"

    @staticmethod
    def primary_metric_key() -> str:
        return "perplexity"

    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.metric = GenerationMetrics()
        self.runtime_objects: Dict[str, Any] = {}

    def is_better(self, current: float, best: float) -> bool:
        # Lower perplexity is better
        return current < best

    def create_metric_accumulator(self) -> Dict[str, Any]:
        return self.metric.create_accumulator()

    def update_metrics(self, acc: Dict[str, Any], outputs: torch.Tensor, targets: torch.Tensor) -> None:
        preds = outputs.argmax(dim=-1)
        self.metric.update_ids(acc, preds.cpu(), targets.cpu())

    def compute_metrics(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        vocab = self.runtime_objects.get("vocab", None)
        return self.metric.compute(acc, vocab=vocab)

    def train_step(self, model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        logits = model(x)
        B, T, V = logits.shape
        loss = self.criterion(logits.view(B * T, V), y.view(B * T))
        return loss, logits, y

    def eval_step(self, model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        logits = model(x)
        B, T, V = logits.shape
        loss = self.criterion(logits.view(B * T, V), y.view(B * T))
        return loss, logits, y


