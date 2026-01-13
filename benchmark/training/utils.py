from typing import Any, Dict
import torch


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def default_reset_parameters(m: torch.nn.Module) -> None:
    if hasattr(m, "reset_parameters") and callable(getattr(m, "reset_parameters")):
        m.reset_parameters()


