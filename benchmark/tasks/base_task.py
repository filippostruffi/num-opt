from typing import Any, Dict
import torch
from benchmark.training.utils import default_reset_parameters


class BaseTask:
    """
    Base interface for tasks.
    """
    @staticmethod
    def task_name() -> str:
        raise NotImplementedError

    @staticmethod
    def primary_metric_key() -> str:
        raise NotImplementedError

    @staticmethod
    def is_better(current: float, best: float) -> bool:
        # Default: higher is better
        return current > best

    @staticmethod
    def reset_parameters(m: torch.nn.Module) -> None:
        default_reset_parameters(m)

    def collate_fn(self, batch):
        # Override if the dataset needs special collation
        return torch.utils.data.default_collate(batch)

    def move_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        if isinstance(batch, dict):
            return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        if isinstance(batch, (list, tuple)):
            return type(batch)((x.to(device) if hasattr(x, "to") else x) for x in batch)
        if hasattr(batch, "to"):
            return batch.to(device)
        return batch

    # Hooks to override
    def create_metric_accumulator(self) -> Dict[str, Any]:
        return {}

    def update_metrics(self, acc: Dict[str, Any], outputs: Any, targets: Any) -> None:
        pass

    def compute_metrics(self, acc: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def train_step(self, model: torch.nn.Module, batch: Any):
        """
        Returns: loss (tensor), outputs (any), targets (any)
        """
        raise NotImplementedError

    def eval_step(self, model: torch.nn.Module, batch: Any):
        """
        Returns: loss (tensor), outputs (any), targets (any)
        """
        raise NotImplementedError


