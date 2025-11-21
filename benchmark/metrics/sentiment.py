from typing import Dict, Any
import torch


class SentimentMetrics:
    def create_accumulator(self) -> Dict[str, Any]:
        return {"correct": 0, "total": 0, "tp": 0, "fp": 0, "fn": 0}

    def update(self, acc: Dict[str, Any], preds: torch.Tensor, targets: torch.Tensor) -> None:
        acc["correct"] += int((preds == targets).sum().item())
        acc["total"] += int(targets.numel())
        # binary metrics
        tp = int(((preds == 1) & (targets == 1)).sum().item())
        fp = int(((preds == 1) & (targets == 0)).sum().item())
        fn = int(((preds == 0) & (targets == 1)).sum().item())
        acc["tp"] += tp
        acc["fp"] += fp
        acc["fn"] += fn

    def compute(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        accuracy = acc["correct"] / acc["total"] if acc["total"] > 0 else 0.0
        precision = acc["tp"] / (acc["tp"] + acc["fp"]) if (acc["tp"] + acc["fp"]) > 0 else 0.0
        recall = acc["tp"] / (acc["tp"] + acc["fn"]) if (acc["tp"] + acc["fn"]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


