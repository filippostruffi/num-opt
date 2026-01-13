from typing import Iterable
import torch
from benchmark.optimizers.factory import register_optimizer


def register_sgd_variants():
    @register_optimizer("sgd")
    def _sgd(params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        return torch.optim.SGD(params, lr=lr)

    @register_optimizer("sgd_momentum")
    def _sgd_momentum(params: Iterable[torch.nn.Parameter], lr: float = 1e-3, momentum: float = 0.9):
        return torch.optim.SGD(params, lr=lr, momentum=momentum)


