import torch
from torch import nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def unwrap_base_dataset(dataset):
    """
    Follow .dataset attributes to unwrap torch.utils.data.Subset or other wrappers.
    """
    base = dataset
    seen = set()
    while hasattr(base, "dataset") and id(base) not in seen:
        seen.add(id(base))
        base = getattr(base, "dataset")
    return base


def infer_num_classes_from_dataset(dataset, default: int = 10) -> int:
    """
    Try to infer number of classes from common dataset attributes.
    Works with Subset and TorchVision datasets (CIFAR10/MNIST).
    """
    base = unwrap_base_dataset(dataset)
    # TorchVision datasets usually expose 'classes'
    if hasattr(base, "classes") and isinstance(base.classes, (list, tuple)):
        try:
            return int(len(base.classes))
        except Exception:
            pass
    # Some expose 'targets' as list or Tensor
    for attr in ("targets", "labels"):
        if hasattr(base, attr):
            targets = getattr(base, attr)
            try:
                if torch.is_tensor(targets):
                    return int(torch.max(targets).item() + 1)
                if isinstance(targets, (list, tuple)) and len(targets) > 0:
                    return int(max(targets) + 1)
            except Exception:
                continue
    return default


