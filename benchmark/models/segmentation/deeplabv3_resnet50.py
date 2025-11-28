from typing import Any
from torch import nn
from torchvision import models
from benchmark.config.registry import register_model


@register_model("deeplabv3_resnet50")
def build_model(task: Any, dataset: Any) -> nn.Module:
    # Prefer dataset-declared num_classes when available (e.g., ADE20K=150)
    num_classes = None
    base_ds = dataset
    try:
        depth_guard = 0
        while hasattr(base_ds, "dataset") and depth_guard < 5:
            base_ds = getattr(base_ds, "dataset")
            depth_guard += 1
    except Exception:
        base_ds = dataset
    ds_nc = getattr(base_ds, "num_classes", None)
    if isinstance(ds_nc, int) and ds_nc > 1:
        num_classes = ds_nc
    else:
        # Fall back to inferring from a sample mask, ignoring 255 if present
        _, mask = dataset[0]
        if mask.dtype.is_floating_point:
            mask = mask.long()
        valid = mask[mask != 255]
        if valid.numel() > 0:
            num_classes = int(valid.max().item()) + 1
        else:
            num_classes = int(mask.max().item()) + 1
    model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=num_classes)
    return model


