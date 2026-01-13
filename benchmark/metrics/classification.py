from typing import Dict, Any
import torch


class ClassificationMetrics:
    def create_accumulator(self) -> Dict[str, Any]:
        return {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "correct": 0,
            "total": 0,
            # Per-class tallies
            "tp_by_class": {},          # class_idx -> int
            "fp_by_class": {},          # class_idx -> int
            "fn_by_class": {},          # class_idx -> int
            "support_by_class": {},     # class_idx -> int (count in targets)
        }

    def update(self, acc: Dict[str, Any], preds: torch.Tensor, targets: torch.Tensor) -> None:
        acc["correct"] += int((preds == targets).sum().item())
        acc["total"] += int(targets.numel())
        # Compute per-class tp/fp/fn and supports.
        # Use union of classes present in preds or targets for robustness.
        present_classes = torch.unique(torch.cat([preds.view(-1), targets.view(-1)], dim=0)).tolist()
        for c in present_classes:
            p_c = preds == c
            t_c = targets == c
            tp = int((p_c & t_c).sum().item())
            fp = int((p_c & (~t_c)).sum().item())
            fn = int(((~p_c) & t_c).sum().item())
            support = int(t_c.sum().item())
            # Global (micro) tallies
            acc["tp"] += tp
            acc["fp"] += fp
            acc["fn"] += fn
            # Per-class tallies
            acc["tp_by_class"][c] = acc["tp_by_class"].get(c, 0) + tp
            acc["fp_by_class"][c] = acc["fp_by_class"].get(c, 0) + fp
            acc["fn_by_class"][c] = acc["fn_by_class"].get(c, 0) + fn
            acc["support_by_class"][c] = acc["support_by_class"].get(c, 0) + support

    def compute(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        accuracy = acc["correct"] / acc["total"] if acc["total"] > 0 else 0.0
        # Micro metrics (preserve existing keys)
        micro_precision = acc["tp"] / (acc["tp"] + acc["fp"]) if (acc["tp"] + acc["fp"]) > 0 else 0.0
        micro_recall = acc["tp"] / (acc["tp"] + acc["fn"]) if (acc["tp"] + acc["fn"]) > 0 else 0.0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # Per-class metrics
        class_indices = sorted(set(list(acc["support_by_class"].keys()) + list(acc["tp_by_class"].keys())))
        per_class_precisions = []
        per_class_recalls = []
        per_class_f1s = []
        per_class_weights = []

        result: Dict[str, Any] = {
            "accuracy": accuracy,
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        }

        for c in class_indices:
            tp_c = acc["tp_by_class"].get(c, 0)
            fp_c = acc["fp_by_class"].get(c, 0)
            fn_c = acc["fn_by_class"].get(c, 0)
            support_c = acc["support_by_class"].get(c, 0)
            prec_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
            rec_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
            f1_c = 2 * prec_c * rec_c / (prec_c + rec_c) if (prec_c + rec_c) > 0 else 0.0
            # Collect for aggregates
            per_class_precisions.append(prec_c)
            per_class_recalls.append(rec_c)
            per_class_f1s.append(f1_c)
            per_class_weights.append(support_c)
            # Flat per-class keys (CSV-friendly)
            result[f"precision_c{c}"] = prec_c
            result[f"recall_c{c}"] = rec_c
            result[f"f1_c{c}"] = f1_c
            result[f"support_c{c}"] = support_c

        # Macro averages (unweighted mean over classes that appeared)
        num_classes = len(class_indices)
        if num_classes > 0:
            result["macro_precision"] = sum(per_class_precisions) / num_classes
            result["macro_recall"] = sum(per_class_recalls) / num_classes
            result["macro_f1"] = sum(per_class_f1s) / num_classes
        else:
            result["macro_precision"] = 0.0
            result["macro_recall"] = 0.0
            result["macro_f1"] = 0.0

        # Weighted averages (weighted by class support)
        total_support = sum(per_class_weights)
        if total_support > 0:
            w_prec = sum(w * p for w, p in zip(per_class_weights, per_class_precisions)) / total_support
            w_rec = sum(w * r for w, r in zip(per_class_weights, per_class_recalls)) / total_support
            w_f1 = sum(w * f for w, f in zip(per_class_weights, per_class_f1s)) / total_support
            result["weighted_precision"] = w_prec
            result["weighted_recall"] = w_rec
            result["weighted_f1"] = w_f1
        else:
            result["weighted_precision"] = 0.0
            result["weighted_recall"] = 0.0
            result["weighted_f1"] = 0.0

        return result


