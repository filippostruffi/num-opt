from typing import Dict, Callable, Iterable
import torch

# Registry of optimizer builders
OPTIMIZER_REGISTRY: Dict[str, Callable[..., torch.optim.Optimizer]] = {}


def register_optimizer(name: str):
    def _decorator(fn: Callable[..., torch.optim.Optimizer]):
        OPTIMIZER_REGISTRY[name.lower()] = fn
        return fn
    return _decorator


# Built-ins and custom imports to populate registry
from benchmark.optimizers.sgd import register_sgd_variants  # noqa: E402
from benchmark.optimizers.adaptive import register_adaptive_variants  # noqa: E402
from benchmark.optimizers.advanced import register_advanced_variants  # noqa: E402

# Initialize registrations
register_sgd_variants()
register_adaptive_variants()
register_advanced_variants()


