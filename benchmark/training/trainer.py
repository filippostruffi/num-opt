import time
from typing import Dict, Any, List, Tuple

import torch
from torch.utils.data import DataLoader
import copy


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
        lr_scheduler: Any = None,
        scheduler_mode: str = "epoch",
        early_stopping_patience: int = None,
        checkpoint_dir: str = None,
        max_grad_norm: float = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        train_hist: List[Dict[str, Any]] = []
        val_hist: List[Dict[str, Any]] = []
        best_primary = None
        best_state_dict = None
        epochs_to_converge = None
        epochs_since_improve = 0
        epochs_to_plateau = None
        no_improve_epochs = 0

        start_time = time.time()
        cumu_train_time_s = 0.0
        cumu_total_time_s = 0.0
        plateau_time_s = None
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_metrics = self._train_one_epoch(
                model, optimizer, train_loader, epoch, max_grad_norm=max_grad_norm
            )
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
                    # Save best model state
                    try:
                        best_state_dict = copy.deepcopy(model.state_dict())
                    except Exception:
                        best_state_dict = None
                    epochs_to_converge = epoch
                    epochs_since_improve = 0
                    no_improve_epochs = 0
                    # If we had marked a plateau earlier but improvement happened later,
                    # reset plateau markers to reflect the latest training dynamics.
                    epochs_to_plateau = None
                    plateau_time_s = None
                else:
                    epochs_since_improve += 1
                    no_improve_epochs += 1
                    if (epochs_to_plateau is None) and (epochs_since_improve >= plateau_patience):
                        epochs_to_plateau = epoch
                        plateau_time_s = cumu_train_time_s

            # Scheduler step
            if lr_scheduler is not None:
                try:
                    if scheduler_mode == "plateau":
                        # Default to monitoring validation loss if present; otherwise primary metric
                        monitored = val_metrics.get("loss", None)
                        if monitored is None and current_primary is not None:
                            # Invert if task uses "higher is better" so plateau scheduler still interprets lower as better
                            monitored = -float(current_primary) if self.task.is_better(1.0, 0.0) else float(current_primary)
                        if monitored is not None:
                            lr_scheduler.step(monitored)
                    else:
                        lr_scheduler.step()
                except Exception:
                    pass

            # Early stopping
            if isinstance(early_stopping_patience, int) and early_stopping_patience > 0:
                if no_improve_epochs >= early_stopping_patience:
                    break

            self.logger.log_step(
                optimizer_name=optimizer_name,
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                runtime_so_far_s=cumu_total_time_s,
                convergence_time_s=plateau_time_s,
            )

        # Save best checkpoint if requested
        if checkpoint_dir is not None and best_state_dict is not None:
            try:
                import os
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(best_state_dict, os.path.join(checkpoint_dir, "best_model.pt"))
            except Exception:
                pass

        total_time = time.time() - start_time
        times = {
            "total_runtime_s": total_time,
            "epochs_to_converge": epochs_to_converge,
            "time_to_converge_s": None if epochs_to_converge is None else sum([h["epoch_runtime_s"] for h in train_hist[:epochs_to_converge]]),
            "epochs_to_plateau": epochs_to_plateau,
            "time_to_plateau_s": None if epochs_to_plateau is None else sum([h["epoch_runtime_s"] for h in train_hist[:epochs_to_plateau]]),
        }
        return train_hist, val_hist, times

    def _train_one_epoch(self, model, optimizer, loader, epoch: int, max_grad_norm: float = None) -> Dict[str, Any]:
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
            if isinstance(max_grad_norm, (int, float)) and max_grad_norm and max_grad_norm > 0:
                try:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                except Exception:
                    pass
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


