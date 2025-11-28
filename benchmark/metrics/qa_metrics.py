from typing import Dict, Any
import torch


class QAMetrics:
    def create_accumulator(self) -> Dict[str, Any]:
        return {"em": 0, "f1_sum": 0.0, "p_sum": 0.0, "r_sum": 0.0, "n": 0}

    def update(self, acc: Dict[str, Any], start_pred: torch.Tensor, end_pred: torch.Tensor, start_true: torch.Tensor, end_true: torch.Tensor) -> None:
        for sp, ep, st, et in zip(start_pred, end_pred, start_true, end_true):
            sp = int(sp.item())
            ep = int(ep.item())
            st = int(st.item())
            et = int(et.item())
            # exact match (span equality)
            em = 1 if (sp == st and ep == et) else 0
            acc["em"] += em
            # overlap-based P/R/F1
            pred_set = set(range(min(sp, ep), max(sp, ep) + 1))
            true_set = set(range(min(st, et), max(st, et) + 1))
            inter = len(pred_set.intersection(true_set))
            p = inter / max(1, len(pred_set))
            r = inter / max(1, len(true_set))
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            acc["f1_sum"] += f1
            acc["p_sum"] += p
            acc["r_sum"] += r
            acc["n"] += 1

    def compute(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        n = max(1, acc["n"])
        return {
            "em": acc["em"] / n,
            "f1": acc["f1_sum"] / n,
            "precision": acc["p_sum"] / n,
            "recall": acc["r_sum"] / n,
        }


