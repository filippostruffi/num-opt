import csv
import json
import os
from typing import Dict, Any, List


class ResultLogger:
    def __init__(self, output_dir: str, run_metadata: Dict[str, Any]):
        self.output_dir = output_dir
        self.meta = run_metadata

    def log_step(
        self,
        optimizer_name: str,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
        runtime_so_far_s: float,
        convergence_time_s: Any,
    ) -> None:
        # Console logging (simple)
        pmk = None
        if "task" in self.meta:
            from benchmark.config.registry import TASK_REGISTRY
            task_cls = TASK_REGISTRY[self.meta["task"]]
            pmk = task_cls.primary_metric_key()
        task = self.meta.get("task", "?")
        model = self.meta.get("model", "?")
        dataset = self.meta.get("dataset", "?")
        primary_val = val_metrics.get(pmk) if pmk else None
        conv_str = f"{convergence_time_s:.2f}s" if isinstance(convergence_time_s, (float, int)) else "NA"
        print(
            f"[{task}] [{model}][{dataset}] [{optimizer_name}] Epoch {epoch} | "
            f"train loss={train_metrics.get('loss'):.4f} | "
            f"val loss={val_metrics.get('loss'):.4f} | "
            f"{pmk}={primary_val} | "
            f"runtime={runtime_so_far_s:.2f}s | convergence_time={conv_str}"
        )

    def save_json(self, history: List[Dict[str, Any]], filename: str) -> None:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    def save_csv(self, history: List[Dict[str, Any]], filename: str) -> None:
        if not history:
            return
        path = os.path.join(self.output_dir, filename)
        keys = sorted(history[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for row in history:
                writer.writerow(row)


