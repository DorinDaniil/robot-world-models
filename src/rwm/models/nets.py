"""Общие нейросетевые компоненты для SAC, TD-MPC2 и Dreamer.

Содержимое:
    mlp            — многослойный перцептрон с LayerNorm и Mish (как в TD-MPC2);
    symlog/symexp  — сжатие масштаба наград и ценностей (DreamerV3, L4 §4.3);
    TwoHot         — дискретная регрессия на сетке бинов (L5 §5.3);
    SimNorm        — симплексная нормализация латента (L5 §5.3);
    CNNEncoder/CNNDecoder — свёрточные сети для изображений 64×64;
    TanhGaussian   — голова политики с tanh-сжатием и корректным log-prob (L1 §1.7);
    RunningScale   — скользящая оценка размаха доходностей для нормировки (L4 §4.4);
    soft_update    — экспоненциальное сглаживание целевых сетей.

Всё написано на «голом» PyTorch без внешних зависимостей, чтобы каждый компонент можно было
прочитать целиком.
"""

from __future__ import annotations

import itertools
import math

import torch
import torch.nn.functional as F
from torch import nn

# ----------------------------------------------------------------------------- MLP


def mlp(
    in_dim: int,
    hidden: int | tuple[int, ...],
    out_dim: int,
    *,
    act: type[nn.Module] = nn.Mish,
    norm: bool = True,
) -> nn.Sequential:
    """Linear → [LayerNorm] → act, повторить; последний слой линейный без нормализации."""
    dims = (in_dim, *((hidden,) if isinstance(hidden, int) else hidden))
    layers: list[nn.Module] = []
    for a, b in itertools.pairwise(dims):
        layers.append(nn.Linear(a, b))
        if norm:
            layers.append(nn.LayerNorm(b))
        layers.append(act())
    layers.append(nn.Linear(dims[-1], out_dim))
    return nn.Sequential(*layers)


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """θ̄ ← (1 − τ) θ̄ + τ θ."""
    for p_t, p_s in zip(target.parameters(), source.parameters()):
        p_t.lerp_(p_s, tau)


# ----------------------------------------------------------------------------- symlog / two-hot


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.expm1(x.abs())


class TwoHot(nn.Module):
    """Дискретная регрессия: скаляр y ↦ распределение на сетке бинов в symlog-пространстве.

    encode(y) — two-hot вектор: масса распределена между двумя соседними бинами так, что
    математическое ожидание равно symlog(y) (L5, упр. 3). decode(logits) — symexp от
    ожидания по softmax(logits). loss(logits, y) — кросс-энтропия с encode(y).
    """

    def __init__(self, num_bins: int = 101, low: float = -10.0, high: float = 10.0):
        super().__init__()
        self.register_buffer("bins", torch.linspace(low, high, num_bins))

    @property
    def num_bins(self) -> int:
        return self.bins.numel()

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        y = symlog(y).clamp(self.bins[0], self.bins[-1])
        idx = torch.bucketize(y, self.bins, right=True).clamp(1, self.num_bins - 1)
        lo, hi = self.bins[idx - 1], self.bins[idx]
        w_hi = (y - lo) / (hi - lo)
        target = torch.zeros(*y.shape, self.num_bins, device=y.device)
        target.scatter_(-1, (idx - 1).unsqueeze(-1), (1 - w_hi).unsqueeze(-1))
        target.scatter_add_(-1, idx.unsqueeze(-1), w_hi.unsqueeze(-1))
        return target

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        return symexp(torch.softmax(logits, -1) @ self.bins)

    def loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Поэлементная кросс-энтропия, форма как у y."""
        return -(self.encode(y) * torch.log_softmax(logits, -1)).sum(-1)


# ----------------------------------------------------------------------------- SimNorm


class SimNorm(nn.Module):
    """Латент разбивается на группы по `dim` компонент, к каждой применяется softmax.

    Образ — произведение симплексов, поэтому ‖z‖ ограничена и латент не может коллапсировать
    в нуль или разрастись (Lavoie et al., 2023; Hansen et al., 2024).
    """

    def __init__(self, dim: int = 8):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.view(*shape[:-1], -1, self.dim)
        return torch.softmax(x, -1).view(shape)


# ----------------------------------------------------------------------------- CNN


class CNNEncoder(nn.Module):
    """RGB uint8 (B, H, W, 3) при H = W = 64 → вектор признаков. Четыре свёртки stride 2."""

    def __init__(self, depth: int = 32, out_dim: int | None = None):
        super().__init__()
        chans = (3, depth, 2 * depth, 4 * depth, 8 * depth)
        layers: list[nn.Module] = []
        for a, b in itertools.pairwise(chans):
            layers += [nn.Conv2d(a, b, 4, stride=2, padding=1), nn.SiLU()]
        self.conv = nn.Sequential(*layers)
        self.flat_dim = 8 * depth * 4 * 4
        self.proj = nn.Identity() if out_dim is None else nn.Linear(self.flat_dim, out_dim)
        self.out_dim = out_dim or self.flat_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float().permute(0, 3, 1, 2) / 255.0 - 0.5
        return self.proj(self.conv(x).flatten(1))


class CNNDecoder(nn.Module):
    """Вектор → RGB (B, 64, 64, 3) в диапазоне [0, 1] (среднее гауссовой плотности)."""

    def __init__(self, in_dim: int, depth: int = 32):
        super().__init__()
        self.depth = depth
        self.fc = nn.Linear(in_dim, 8 * depth * 4 * 4)
        chans = (8 * depth, 4 * depth, 2 * depth, depth)
        layers: list[nn.Module] = []
        for a, b in itertools.pairwise(chans):
            layers += [nn.ConvTranspose2d(a, b, 4, stride=2, padding=1), nn.SiLU()]
        layers.append(nn.ConvTranspose2d(depth, 3, 4, stride=2, padding=1))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 8 * self.depth, 4, 4)
        return self.deconv(x).permute(0, 2, 3, 1) + 0.5


# ----------------------------------------------------------------------------- политика


class TanhGaussian(nn.Module):
    """Голова стохастической политики: u ~ N(μ(x), σ(x)²), a = tanh(u).

    log π(a|x) = log N(u; μ, σ²) − Σ_i log(1 − tanh² u_i); второе слагаемое вычисляется
    устойчиво как 2(log 2 − u − softplus(−2u)) (L1, упр. 3).
    """

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, in_dim: int, act_dim: int):
        super().__init__()
        self.out = nn.Linear(in_dim, 2 * act_dim)

    def forward(self, x: torch.Tensor, deterministic: bool = False):
        mu, log_std = self.out(x).chunk(2, -1)
        log_std = log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        log_prob = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        log_prob = log_prob - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return a, log_prob


# ----------------------------------------------------------------------------- нормировка


class RunningScale(nn.Module):
    """Скользящая оценка размаха S = P95 − P05 величины; нормировка x / max(1, S).

    Деление на max(1, S), а не на S, не усиливает шум при малых доходностях (DreamerV3).
    """

    def __init__(self, tau: float = 0.01):
        super().__init__()
        self.tau = tau
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        lo, hi = torch.quantile(x.detach().flatten().float(), torch.tensor([0.05, 0.95]))
        s = (hi - lo).clamp_min(1e-8)
        if self.initialized:
            self.scale.lerp_(s, self.tau)
        else:
            self.scale.copy_(s)
            self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale.clamp_min(1.0)
