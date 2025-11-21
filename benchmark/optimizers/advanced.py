from typing import Iterable
import math
import torch
from torch.optim.optimizer import Optimizer
from benchmark.optimizers.factory import register_optimizer


class Lion(Optimizer):
    """
    Alignment with Lion (Chen et al., 2023):
    - Intent: sign-descent on an EMA of gradients with decoupled weight decay.
    - This implementation follows the common Lion update pattern:
      m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
      w_{t+1} = w_t - lr * sign(m_t)
      with decoupled weight decay applied to weights.
    """
    # https://github.com/google/automl/tree/master/lion (simplified)
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, _beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                m = state["m"]
                # Decoupled weight decay
                if wd != 0:
                    p.data.mul_(1 - lr * wd)
                # Lion momentum and sign step
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                p.add_(m.sign(), alpha=-lr)
        return loss


class LARS(Optimizer):
    """
    Alignment with LARS (You et al., 2017):
    - Implements layer-wise trust ratio: eta * ||w|| / ||g|| with momentum and
      optional weight decay. This is a standard minimal LARS formulation and is
      aligned in spirit with the paper (details like per-layer eps variants may vary).
    """
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0, eta=0.001, eps=1e-9):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            eta = group["eta"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)
                w_norm = p.norm().clamp(min=eps)
                g_norm = g.norm().clamp(min=eps)
                # Exclude 1D params (e.g., biases, LayerNorm scales) from LARS scaling
                if p.ndim == 1:
                    trust_ratio = 1.0
                else:
                    trust_ratio = eta * w_norm / g_norm
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g, alpha=trust_ratio)
                p.add_(buf, alpha=-lr)
        return loss


class LAMB(Optimizer):
    """
    Alignment with LAMB (You et al., 2019):
    - Adam moments with bias correction, decoupled weight decay, and layer-wise
      trust ratio ||w||/||u||. This follows the core algorithm.
    - Simplifications: no trust-ratio clipping/windowing and no extra implementation
      niceties sometimes seen in reference code.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] = 0
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                # Adam moments
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                # Bias correction
                step = state["step"]
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2
                adam_step = m_hat / (v_hat.sqrt().add(eps))
                # Decoupled weight decay
                if wd != 0:
                    adam_step = adam_step.add(p, alpha=wd)
                # Layer-wise adaptive scaling, exclude 1D params (bias, LN)
                if p.ndim == 1:
                    trust_ratio = 1.0
                else:
                    w_norm = p.norm(p=2)
                    u_norm = adam_step.norm(p=2)
                    trust_ratio = (w_norm / u_norm) if (w_norm > 0 and u_norm > 0) else 1.0
                p.add_(adam_step, alpha=-lr * trust_ratio)
        return loss


class AdaBelief(Optimizer):
    """
    Alignment with AdaBelief (Zhuang et al., 2020):
    - Tracks the belief (g - m)^2 for second moment; uses bias-corrected moments
      and decoupled weight decay. This matches the paper's core update.
    - Minor simplifications: no special epsilon scheduling or optional stabilizers.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                    state["s"] = torch.zeros_like(p)
                    state["step"] = 0
                m = state["m"]
                s = state["s"]
                state["step"] += 1
                # First moment
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                # Belief (variance of gradient prediction)
                grad_residual = g - m
                s.mul_(beta2).addcmul_(grad_residual, grad_residual, value=1 - beta2)
                # Bias corrections
                step = state["step"]
                m_hat = m / (1 - beta1 ** step)
                s_hat = s / (1 - beta2 ** step)
                update = m_hat / (s_hat.sqrt().add(eps))
                # Decoupled weight decay
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class Yogi(Optimizer):
    """
    Alignment with Yogi (Zaheer et al., 2018):
    - Uses the Yogi variance rule: v_t = v_{t-1} - (1 - beta2) * sign(v_{t-1} - g^2) * g^2,
      with bias corrections on m and v. This mirrors the paper's update.
    - Simplifications: standard decoupled weight decay; no auxiliary stabilizers.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-3, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["step"] = 0
                m = state["m"]
                v = state["v"]
                state["step"] += 1
                # First moment
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                # Yogi variance update (no extra EMA pass)
                g2 = g * g
                v.add_(- (1.0 - beta2) * torch.sign(v - g2) * g2)
                v.clamp_(min=eps)
                # Bias corrections
                step = state["step"]
                m_hat = m / (1 - beta1 ** step)
                v_hat = v / (1 - beta2 ** step)
                update = m_hat / (v_hat.sqrt().add(eps))
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class Adafactor(Optimizer):
    """
    Alignment with Adafactor (Shazeer & Stern, 2018):
    - Supports factored second-moment estimates for matrices and RMS-based clipping.
    - Simplifications: does not implement parameter-scale relative updates or the
      full learning-rate schedule/relative step heuristics from the paper; non-factored
      path used for vectors/tensors < 2D.
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        eps2=(1e-30, 1e-3),
        weight_decay=0.0,
        clip_threshold=1.0,
        decay_rate=0.8,
        factored=True,
        relative_step=False,
        warmup_init=False,
    ):
        defaults = dict(
            lr=lr,
            eps2=eps2,
            weight_decay=weight_decay,
            clip_threshold=clip_threshold,
            decay_rate=decay_rate,
            factored=factored,
            relative_step=relative_step,
            warmup_init=warmup_init,
        )
        super().__init__(params, defaults)

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        return x.pow(2).mean().sqrt()

    def _get_lr(self, group, p, state_step: int) -> float:
        lr = group["lr"]
        if group.get("relative_step", False):
            if group.get("warmup_init", False):
                # Suggested warmup schedule in paper: min(1e-2, 1/sqrt(t))
                return min(1e-2, 1.0 / math.sqrt(max(state_step, 1)))
            return 1.0 / math.sqrt(max(state_step, 1))
        return lr

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            eps2_small, eps2_large = group["eps2"]
            wd = group["weight_decay"]
            clip_threshold = group["clip_threshold"]
            decay_rate = group["decay_rate"]
            factored = group.get("factored", True)
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                lr = self._get_lr(group, p, state["step"])
                if factored and p.ndim >= 2:
                    # Factored second moment: flatten to 2D (rows x cols) using last dim as columns
                    rows = int(p.numel() // p.shape[-1])
                    cols = int(p.shape[-1])
                    if ("exp_avg_sq_row" not in state) or (state["exp_avg_sq_row"].numel() != rows) or (state["exp_avg_sq_col"].numel() != cols):
                        state["exp_avg_sq_row"] = torch.zeros(rows, dtype=p.dtype, device=p.device)
                        state["exp_avg_sq_col"] = torch.zeros(cols, dtype=p.dtype, device=p.device)
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    # Update factored statistics from flattened gradient squares
                    grad_sq_flat = (g.detach() * g.detach()).view(rows, cols)
                    exp_avg_sq_row.mul_(decay_rate).add_(grad_sq_flat.mean(dim=1), alpha=1 - decay_rate)
                    exp_avg_sq_col.mul_(decay_rate).add_(grad_sq_flat.mean(dim=0), alpha=1 - decay_rate)
                    # Compose approximate second moment and reshape back
                    r = exp_avg_sq_row
                    c = exp_avg_sq_col
                    rc_mean = r.mean().clamp(min=eps2_small)
                    v_flat = torch.outer(r, c) / rc_mean
                    v = v_flat.view_as(p)
                else:
                    # Non-factored path
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    v = state["exp_avg_sq"]
                    v.mul_(decay_rate).addcmul_(g, g, value=1 - decay_rate)
                # Normalize gradient
                r_factor = g * (v + eps2_small).rsqrt()
                # RMS-based clipping
                rms_update = r_factor.pow(2).mean().sqrt() + eps2_large
                clip = min(clip_threshold, rms_update.item())
                update = r_factor / max(rms_update.item(), 1e-8) * clip
                # Optional parameter-scale relative update (approximate)
                if group.get("relative_step", False):
                    param_scale = max(self._rms(p.data).item(), eps2_large)
                    lr = lr * param_scale
                if wd != 0:
                    p.add_(p, alpha=-lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class SAM(Optimizer):
    """
    Alignment with SAM (Foret et al., 2021):
    - Two-step procedure via closure: (1) perturb weights along normalized gradient
      by radius rho; (2) restore and apply base optimizer step. This captures the
      core algorithm with AdamW as base by default.
    - Simplifications: single global rho; no layer-wise or per-parameter rho variants.
    """
    # Sharpness-Aware Minimization (minimal)
    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        if isinstance(params, list):
            self.params = params
        defaults = dict(rho=rho, **kwargs)
        self.base_optimizer = base_optimizer_cls(params, **kwargs)
        self.rho = rho
        self.param_groups = self.base_optimizer.param_groups
        self._sam_state = {}

    @torch.no_grad()
    def first_step(self):
        grad_norm = torch.norm(
            torch.stack([p.grad.norm(p=2) for group in self.param_groups for p in group["params"] if p.grad is not None]),
            p=2,
        )
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Save original weights in SAM's private state to avoid polluting base optimizer state
                self._sam_state[p] = p.data.clone()
                e_w = p.grad * scale
                p.add_(e_w)

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p in self._sam_state:
                    p.data = self._sam_state[p]
        # Clear saved weights
        self._sam_state.clear()
        self.base_optimizer.step()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "SAM requires closure for two forward-backward passes."
        # first step
        loss = closure()
        self.first_step()
        # second step
        loss = closure()
        self.second_step()
        return loss

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @property
    def state(self):
        return self.base_optimizer.state


def register_advanced_variants():
    @register_optimizer("lion")
    def _lion(params: Iterable[torch.nn.Parameter], lr: float = 1e-4):
        return Lion(params, lr=lr)

    @register_optimizer("lars")
    def _lars(params: Iterable[torch.nn.Parameter], lr: float = 0.1):
        return LARS(params, lr=lr)

    @register_optimizer("lamb")
    def _lamb(params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        return LAMB(params, lr=lr)

    @register_optimizer("adabelief")
    def _adabelief(params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        return AdaBelief(params, lr=lr)

    @register_optimizer("yogi")
    def _yogi(params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        return Yogi(params, lr=lr)

    @register_optimizer("adafactor")
    def _adafactor(params: Iterable[torch.nn.Parameter], lr: float = 1e-3):
        return Adafactor(params, lr=lr)

    @register_optimizer("sam")
    def _sam(params: Iterable[torch.nn.Parameter], lr: float = 1e-3, rho: float = 0.05):
        # SAM over AdamW as common choice
        base = torch.optim.AdamW
        return SAM(params, base_optimizer_cls=base, lr=lr, rho=rho)

    @register_optimizer("gsam")
    def _gsam(params: Iterable[torch.nn.Parameter], lr: float = 1e-3, rho: float = 0.05, gamma: float = 0.1):
        # Minimal GSAM approximation:
        # The GSAM paper proposes gradient-guided sharpness alignment beyond simply
        # increasing rho. Here we approximate GSAM by scaling SAM's rho by (1 + gamma),
        # which captures only the increased perturbation magnitude, not the full method.
        base = torch.optim.AdamW
        return SAM(params, base_optimizer_cls=base, lr=lr, rho=rho * (1.0 + gamma))


