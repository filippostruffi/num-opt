from typing import Dict, Any, Tuple
import torch
from torch import nn
from benchmark.tasks.base_task import BaseTask
from benchmark.metrics.segmentation_metrics import SegmentationMetrics
from benchmark.config.registry import register_task
import torch.nn.functional as F


@register_task("semantic_segmentation")
class SemanticSegmentationTask(BaseTask):
    @staticmethod
    def task_name() -> str:
        return "semantic_segmentation"

    @staticmethod
    def primary_metric_key() -> str:
        return "dice"

    def __init__(self):
        super().__init__()
        # VOC uses 0..20 classes and 255 as ignore
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        self.metric = SegmentationMetrics(num_classes=21, ignore_index=255)
        # Lazy init for class weights (computed on first batch)
        self._ce_weight_set = False
        self._dice_weight = 0.5  # CE + 0.5 * Dice tends to stabilize foreground learning

    def create_metric_accumulator(self) -> Dict[str, Any]:
        return self.metric.create_accumulator()

    def update_metrics(self, acc: Dict[str, Any], outputs: torch.Tensor, targets: torch.Tensor) -> None:
        preds = torch.argmax(outputs, dim=1)
        self.metric.update(acc, preds.cpu(), targets.cpu())

    def compute_metrics(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        return self.metric.compute(acc)

    @staticmethod
    def _compute_batch_ce_weights(targets: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
        with torch.no_grad():
            valid = targets != ignore_index
            if valid.any():
                counts = torch.bincount(targets[valid].view(-1), minlength=num_classes).float()
                total = counts.sum().clamp(min=1.0)
                weights = total / (counts.clamp(min=1.0) * float(num_classes))
                # Normalize to mean weight ~1.0 for stability
                weights = weights * (float(num_classes) / weights.sum().clamp(min=1e-6))
            else:
                weights = torch.ones(num_classes, dtype=torch.float, device=targets.device)
        return weights

    def _soft_dice_loss(self, logits: torch.Tensor, targets: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
        # logits: BxCxHxW, targets: BxHxW
        probs = F.softmax(logits, dim=1)
        with torch.no_grad():
            t = targets.clone()
            mask = (t == ignore_index)
            t[mask] = 0
            one_hot = F.one_hot(t, num_classes=num_classes).permute(0, 3, 1, 2).float()
            one_hot[mask.unsqueeze(1).expand_as(one_hot)] = 0.0
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        union = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        mdice = dice.mean()
        return 1.0 - mdice

    def _ensure_ce_criterion_aligned(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        # Build or fix CE criterion so its weight length equals logits channels
        desired_c = int(logits.shape[1])
        need_rebuild = False
        cur_weight = None
        if hasattr(self, "criterion") and isinstance(self.criterion, nn.CrossEntropyLoss):
            cur_weight = getattr(self.criterion, "weight", None)
            if cur_weight is not None:
                try:
                    cur_numel = int(cur_weight.numel())
                except Exception:
                    cur_numel = 0
                if cur_numel not in (0, desired_c):
                    need_rebuild = True
            else:
                # No weight set; we will set one
                need_rebuild = True
        else:
            need_rebuild = True
        if (not self._ce_weight_set) or need_rebuild:
            weights = self._compute_batch_ce_weights(targets, num_classes=desired_c, ignore_index=255).to(logits.device)
            self.criterion = nn.CrossEntropyLoss(ignore_index=255, weight=weights)
            self._ce_weight_set = True

    def train_step(self, model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        out = model(x)
        logits = out["out"] if isinstance(out, dict) else out
        # Ensure CE criterion weight vector matches logits channels
        self._ensure_ce_criterion_aligned(logits, y)
        try:
            ce_loss = self.criterion(logits, y)
        except RuntimeError as e:
            msg = str(e)
            if "weight tensor should be defined either for all or no classes" in msg:
                # Fallback to unweighted CE to guarantee progress
                self.criterion = nn.CrossEntropyLoss(ignore_index=255)
                ce_loss = self.criterion(logits, y)
            else:
                raise
        dice_loss = self._soft_dice_loss(logits, y, num_classes=int(logits.shape[1]), ignore_index=255)
        loss = ce_loss + self._dice_weight * dice_loss
        return loss, logits, y

    def eval_step(self, model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]):
        x, y = batch
        out = model(x)
        logits = out["out"] if isinstance(out, dict) else out
        # Ensure CE criterion weight vector matches logits channels (eval-only runs etc.)
        self._ensure_ce_criterion_aligned(logits, y)
        try:
            ce_loss = self.criterion(logits, y)
        except RuntimeError as e:
            msg = str(e)
            if "weight tensor should be defined either for all or no classes" in msg:
                self.criterion = nn.CrossEntropyLoss(ignore_index=255)
                ce_loss = self.criterion(logits, y)
            else:
                raise
        dice_loss = self._soft_dice_loss(logits, y, num_classes=int(logits.shape[1]), ignore_index=255)
        loss = ce_loss + self._dice_weight * dice_loss
        return loss, logits, y


