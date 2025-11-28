from typing import Iterable
import torch
from benchmark.optimizers.factory import register_optimizer


def register_adaptive_variants():
    @register_optimizer("rmsprop")
    def _rmsprop(params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        return torch.optim.RMSprop(params, lr=lr)

    @register_optimizer("adagrad")
    def _adagrad(params: Iterable[torch.nn.Parameter], lr: float = 1e-2):
        return torch.optim.Adagrad(params, lr=lr)

    @register_optimizer("adadelta")
    def _adadelta(params: Iterable[torch.nn.Parameter], lr: float = 1.0):
        return torch.optim.Adadelta(params, lr=lr)

    @register_optimizer("adam")
    def _adam(params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        return torch.optim.Adam(params, lr=lr)

    @register_optimizer("adamw")
    def _adamw(params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        return torch.optim.AdamW(params, lr=lr)

    @register_optimizer("radam")
    def _radam(params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        return torch.optim.RAdam(params, lr=lr)


