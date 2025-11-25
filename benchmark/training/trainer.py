import time
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self, task, device, logger):
        self.task = task
        self.device = device
        self.logger = logger

    def fit(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        optimizer_name: str,
        plateau_patience: int = 3,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        train_hist: List[Dict[str, Any]] = []
        val_hist: List[Dict[str, Any]] = []
        best_primary = None
        epochs_to_converge = None
        epochs_since_improve = 0
        epochs_to_plateau = None

        start_time = time.time()
        cumu_train_time_s = 0.0
        cumu_total_time_s = 0.0
        plateau_time_s = None
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self._train_one_epoch(model, optimizer, train_loader, epoch)
            t1 = time.time()
            val_metrics = self._validate(model, val_loader)
            t2 = time.time()

            train_metrics["epoch"] = epoch
            train_metrics["epoch_runtime_s"] = t1 - t0
            val_metrics["epoch"] = epoch
            val_metrics["eval_runtime_s"] = t2 - t1
            epoch_total_time_s = (t2 - t0)
            cumu_train_time_s += train_metrics["epoch_runtime_s"]
            cumu_total_time_s += epoch_total_time_s

            train_hist.append(train_metrics)
            val_hist.append(val_metrics)

            # Convergence / plateau tracking
            primary_key = self.task.primary_metric_key()
            current_primary = val_metrics.get(primary_key, None)
            if current_primary is not None:
                if (best_primary is None) or self.task.is_better(current_primary, best_primary):
                    best_primary = current_primary
                    epochs_to_converge = epoch
                    epochs_since_improve = 0
                    # If we had marked a plateau earlier but improvement happened later,
                    # reset plateau markers to reflect the latest training dynamics.
                    epochs_to_plateau = None
                    plateau_time_s = None
                else:
                    epochs_since_improve += 1
                    if (epochs_to_plateau is None) and (epochs_since_improve >= plateau_patience):
                        epochs_to_plateau = epoch
                        plateau_time_s = cumu_train_time_s

            self.logger.log_step(
                optimizer_name=optimizer_name,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                runtime_so_far_s=cumu_total_time_s,
                convergence_time_s=plateau_time_s,
            )

        total_time = time.time() - start_time
        times = {
            "total_runtime_s": total_time,
            "epochs_to_converge": epochs_to_converge,
            "time_to_converge_s": None if epochs_to_converge is None else sum([h["epoch_runtime_s"] for h in train_hist[:epochs_to_converge]]),
            "epochs_to_plateau": epochs_to_plateau,
            "time_to_plateau_s": None if epochs_to_plateau is None else sum([h["epoch_runtime_s"] for h in train_hist[:epochs_to_plateau]]),
        }
        return train_hist, val_hist, times

    def _train_one_epoch(self, model, optimizer, loader, epoch: int) -> Dict[str, Any]:
        model.train()
        device = self.device
        total_loss = 0.0
        num_batches = 0
        metric_accumulator = self.task.create_metric_accumulator()

        for batch in loader:
            batch = self.task.move_batch_to_device(batch, device)
            optimizer.zero_grad()
            loss, outputs, targets = self.task.train_step(model, batch)
            loss.backward()
            # Support SAM/GSAM two-step update if available
            if hasattr(optimizer, "first_step") and hasattr(optimizer, "second_step"):
                optimizer.first_step()
                optimizer.zero_grad()
                loss2, _, _ = self.task.train_step(model, batch)
                loss2.backward()
                optimizer.second_step()
            else:
                optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1
            self.task.update_metrics(metric_accumulator, outputs, targets)

        metrics = self.task.compute_metrics(metric_accumulator)
        metrics["loss"] = total_loss / max(1, num_batches)
        return metrics

    def _validate(self, model, loader) -> Dict[str, Any]:
        model.eval()
        device = self.device
        total_loss = 0.0
        num_batches = 0
        metric_accumulator = self.task.create_metric_accumulator()

        with torch.no_grad():
            for batch in loader:
                batch = self.task.move_batch_to_device(batch, device)
                loss, outputs, targets = self.task.eval_step(model, batch)
                total_loss += float(loss.item())
                num_batches += 1
                self.task.update_metrics(metric_accumulator, outputs, targets)

        metrics = self.task.compute_metrics(metric_accumulator)
        metrics["loss"] = total_loss / max(1, num_batches)
        return metrics


