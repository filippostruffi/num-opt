from typing import Dict, Any
import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def create_accumulator(self) -> Dict[str, Any]:
        return {"confusion": torch.zeros(self.num_classes, self.num_classes, dtype=torch.long), "pix_correct": 0, "pix_total": 0}

    def update(self, acc: Dict[str, Any], preds: torch.Tensor, targets: torch.Tensor) -> None:
        # preds, targets: BxHxW
        preds = preds.view(-1)
        targets = targets.view(-1)
        mask = targets != self.ignore_index
        preds = preds[mask]
        targets = targets[mask]
        acc["pix_correct"] += int((preds == targets).sum().item())
        acc["pix_total"] += int(targets.numel())
        # Guard against any out-of-range labels/preds; skip them from confusion update
        k = (targets >= 0) & (targets < self.num_classes) & (preds >= 0) & (preds < self.num_classes)
        inds = self.num_classes * targets[k] + preds[k]
        c = torch.bincount(inds, minlength=self.num_classes ** 2)
        acc["confusion"] += c.view(self.num_classes, self.num_classes)

    def compute(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        conf = acc["confusion"].float()
        diag = torch.diag(conf)
        pix_acc = acc["pix_correct"] / max(1, acc["pix_total"])
        class_acc = diag / (conf.sum(1) + 1e-9)
        mpa = float(class_acc.mean().item())
        dice = 2 * diag / (conf.sum(1) + conf.sum(0) + 1e-9)
        mdice = float(dice.mean().item())
        return {
            "pixel_accuracy": float(pix_acc),
            "mean_pixel_accuracy": mpa,
            "dice": mdice,
        }


