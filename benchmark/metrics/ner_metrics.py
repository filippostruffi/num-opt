from typing import Dict, Any
import torch


class NERMetrics:
    def create_accumulator(self) -> Dict[str, Any]:
        return {"tp": 0, "fp": 0, "fn": 0}

    def update(self, acc: Dict[str, Any], preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> None:
        # Exclude padding (mask==0)
        p = preds[mask.bool()]
        t = targets[mask.bool()]
        # Treat exact matches excluding 'O' as entity matches (simple proxy)
        entity_mask = t != 0
        tp = int(((p == t) & entity_mask).sum().item())
        fp = int(((p != t) & (p != 0)).sum().item())
        fn = int(((p != t) & (t != 0)).sum().item())
        acc["tp"] += tp
        acc["fp"] += fp
        acc["fn"] += fn

    def compute(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        precision = acc["tp"] / (acc["tp"] + acc["fp"]) if (acc["tp"] + acc["fp"]) > 0 else 0.0
        recall = acc["tp"] / (acc["tp"] + acc["fn"]) if (acc["tp"] + acc["fn"]) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return {
            "entity_accuracy": precision,  # proxy
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }


