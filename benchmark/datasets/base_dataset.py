from typing import Optional, Tuple, Any


def maybe_limit_split(train_set: Any, val_set: Any, max_samples: Optional[int]) -> Tuple[Any, Any]:
    """
    Return limited subsets for faster demos.
    """
    if max_samples is None:
        return train_set, val_set
    from torch.utils.data import Subset
    train_indices = list(range(min(max_samples, len(train_set))))
    val_indices = list(range(min(max_samples, len(val_set))))
    return Subset(train_set, train_indices), Subset(val_set, val_indices)


