"""TD-MPC2 (Hansen, Su, Wang, 2024) — модель в латентном пространстве и планирование MPPI.

Компоненты (лекция 5, §5.3): энкодер h(o) → z (SimNorm), латентная динамика d(z, a) → z'
(SimNorm), голова награды R(z, a) и ансамбль из K критиков Q_i(z, a) — оба two-hot над
бинами в symlog-пространстве, policy prior p(z) → TanhGaussian. EMA-копия ансамбля Q для целей.

Лосс модели на подпоследовательности длины T (латент прокатывается по d, не перекодируется):
    L(θ) = Σ_t ρ^t [ c_1 ‖d(z_t, a_t) − sg(h(o_{t+1}))‖² + c_2 CE(R(z_t, a_t), r_t)
                     + c_3 Σ_i CE(Q_i(z_t, a_t), y_t) ],
    y_t = r_t + γ (1 − d_t) min_{i∈I} Q̄_i(sg(h(o_{t+1})), ã),  ã ~ p(·|h(o_{t+1})),  |I| = 2.
Лосс policy prior (латенты — константы):
    L_p(θ) = Σ_t ρ^t [ α log p(ã_t | z_t) − Q̂(z_t, ã_t) / max(1, S) ],  Q̂ — среднее по ансамблю,
    S — скользящий размах Q (RunningScale).

Планирование (алгоритм 5.1): MPPI на горизонте H с N гауссовых кандидатов и N_π траекторий
policy prior, оценка плана G = Σ_k γ^k R(z_k, a_k) + γ^H Q̂(z_H, p(z_H)), веса
ω_i = exp((G_i − max_j G_j) / τ) по top-k кандидатам, тёплый старт среднего с прошлого шага.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from rwm.data.replay import ReplayBuffer
from rwm.models.nets import (
    CNNEncoder,
    RunningScale,
    SimNorm,
    TanhGaussian,
    TwoHot,
    mlp,
    soft_update,
)


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        act_dim: int,
        latent_dim: int,
        hidden: int,
        num_q: int,
        num_bins: int,
        simnorm_dim: int,
        cnn_depth: int,
    ):
        super().__init__()
        self.pixels = len(obs_shape) == 3
        if self.pixels:
            self.encoder = nn.Sequential(
                CNNEncoder(cnn_depth, out_dim=latent_dim), SimNorm(simnorm_dim)
            )
        else:
            self.encoder = nn.Sequential(
                mlp(obs_shape[0], (hidden,), latent_dim), SimNorm(simnorm_dim)
            )
        self.dynamics = nn.Sequential(
            mlp(latent_dim + act_dim, (hidden, hidden), latent_dim), SimNorm(simnorm_dim)
        )
        self.reward = mlp(latent_dim + act_dim, (hidden, hidden), num_bins)
        self.qs = nn.ModuleList(
            [mlp(latent_dim + act_dim, (hidden, hidden), num_bins) for _ in range(num_q)]
        )
        self.policy_trunk = mlp(latent_dim, (hidden,), hidden)
        self.policy_head = TanhGaussian(hidden, act_dim)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Наблюдение произвольной формы (..., *obs_shape) → латент (..., L)."""
        if self.pixels:
            lead = obs.shape[:-3]
            z = self.encoder(obs.reshape(-1, *obs.shape[-3:]))
            return z.reshape(*lead, -1)
        return self.encoder(obs.float())

    def next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([z, a], -1))

    def reward_logits(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.reward(torch.cat([z, a], -1))

    def q_logits(self, z: torch.Tensor, a: torch.Tensor, qs: nn.ModuleList | None = None):
        """Логиты всех K критиков: (K, ..., num_bins)."""
        x = torch.cat([z, a], -1)
        return torch.stack([q(x) for q in (qs if qs is not None else self.qs)])

    def pi(self, z: torch.Tensor, deterministic: bool = False):
        return self.policy_head(torch.relu(self.policy_trunk(z)), deterministic)


class TDMPC2:
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        act_dim: int,
        *,
        latent_dim: int = 128,
        hidden: int = 256,
        num_q: int = 5,
        num_bins: int = 101,
        simnorm_dim: int = 8,
        cnn_depth: int = 32,
        horizon: int = 3,
        gamma: float = 0.99,
        rho: float = 0.5,
        lr: float = 3e-4,
        tau: float = 0.01,
        batch_size: int = 256,
        grad_clip: float = 20.0,
        consistency_coef: float = 20.0,
        reward_coef: float = 0.1,
        value_coef: float = 0.1,
        entropy_coef: float = 1e-4,
        mpc: bool = True,
        num_samples: int = 512,
        num_pi_trajs: int = 24,
        iterations: int = 6,
        num_elites: int = 64,
        temperature: float = 0.5,
        min_std: float = 0.05,
        max_std: float = 2.0,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.act_dim = act_dim
        self.horizon, self.gamma, self.rho = horizon, gamma, rho
        self.tau, self.batch_size, self.grad_clip = tau, batch_size, grad_clip
        self.coefs = (consistency_coef, reward_coef, value_coef)
        self.entropy_coef = entropy_coef
        self.mpc = mpc
        self.num_samples, self.num_pi_trajs = num_samples, num_pi_trajs
        self.iterations, self.num_elites, self.temperature = iterations, num_elites, temperature
        self.min_std, self.max_std = min_std, max_std

        self.model = WorldModel(
            obs_shape, act_dim, latent_dim, hidden, num_q, num_bins, simnorm_dim, cnn_depth
        ).to(self.device)
        self.qs_target = copy.deepcopy(self.model.qs).requires_grad_(False)
        self.twohot = TwoHot(num_bins).to(self.device)
        self.scale = RunningScale().to(self.device)

        policy_params = list(self.model.policy_trunk.parameters()) + list(
            self.model.policy_head.parameters()
        )
        policy_ids = {id(p) for p in policy_params}
        model_params = [p for p in self.model.parameters() if id(p) not in policy_ids]
        self.opt_model = torch.optim.Adam(model_params, lr=lr)
        self.opt_pi = torch.optim.Adam(policy_params, lr=lr)
        self._model_params = model_params
        self._policy_params = policy_params
        self._prev_mean: torch.Tensor | None = None

    # ------------------------------------------------------------------ вспомогательное
    def q_value(self, z: torch.Tensor, a: torch.Tensor, target: bool = False) -> torch.Tensor:
        """Скалярные значения всех критиков: (K, ...)."""
        logits = self.model.q_logits(z, a, self.qs_target if target else None)
        return self.twohot.decode(logits)

    def reset(self) -> None:
        self._prev_mean = None

    # ------------------------------------------------------------------ действие
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs), device=self.device).unsqueeze(0)
        z = self.model.encode(obs_t)
        if self.mpc:
            a = self.plan(z, deterministic)
        else:
            a = self.model.pi(z, deterministic)[0].squeeze(0)
        return a.cpu().numpy()

    @torch.no_grad()
    def _estimate_value(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """G = Σ_k γ^k R(z_k, a_k) + γ^H Q̂(z_H, p(z_H)) для батча планов actions (H, N, A)."""
        g, discount = 0.0, 1.0
        for a in actions:
            g = g + discount * self.twohot.decode(self.model.reward_logits(z, a))
            z = self.model.next(z, a)
            discount *= self.gamma
        a_h = self.model.pi(z)[0]
        return g + discount * self.q_value(z, a_h).mean(0)

    @torch.no_grad()
    def plan(self, z: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """MPPI (алгоритм 5.1) из латента z формы (1, L); возвращает a_t ∈ [−1, 1]^A."""
        h, n, n_pi, a_dim = self.horizon, self.num_samples, self.num_pi_trajs, self.act_dim
        dev = self.device
        # траектории policy prior
        pi_actions = torch.empty(h, n_pi, a_dim, device=dev)
        z_pi = z.repeat(n_pi, 1)
        for t in range(h):
            pi_actions[t] = self.model.pi(z_pi)[0]
            z_pi = self.model.next(z_pi, pi_actions[t])

        z_all = z.repeat(n + n_pi, 1)
        mean = torch.zeros(h, a_dim, device=dev)
        std = torch.full((h, a_dim), self.max_std, device=dev)
        if self._prev_mean is not None:
            mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(h, n + n_pi, a_dim, device=dev)
        actions[:, :n_pi] = pi_actions
        for _ in range(self.iterations):
            noise = torch.randn(h, n, a_dim, device=dev)
            actions[:, n_pi:] = (mean[:, None] + std[:, None] * noise).clamp(-1, 1)
            value = self._estimate_value(z_all, actions).nan_to_num(0.0)
            elite_idx = torch.topk(value, self.num_elites).indices
            elite_value, elite_actions = value[elite_idx], actions[:, elite_idx]
            score = torch.exp((elite_value - elite_value.max()) / self.temperature)
            score = score / score.sum()
            mean = (score[None, :, None] * elite_actions).sum(1)
            std = (score[None, :, None] * (elite_actions - mean[:, None]) ** 2).sum(1).sqrt()
            std = std.clamp(self.min_std, self.max_std)
        self._prev_mean = mean
        a = mean[0]
        if not deterministic:
            a = a + std[0] * torch.randn(a_dim, device=dev)
        return a.clamp(-1, 1)

    # ------------------------------------------------------------------ обучение
    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        b = buffer.sample_sequences(self.batch_size, self.horizon)
        obs, act, rew, term = b["obs"], b["action"], b["reward"], b["terminated"]
        t_len = act.shape[1]
        c_cons, c_rew, c_val = self.coefs

        # цели: латенты следующих наблюдений и TD-цели по EMA-копии ансамбля
        with torch.no_grad():
            z_next_target = self.model.encode(obs[:, 1:])  # (B, T, L)
            a_next, _ = self.model.pi(z_next_target)
            idx = torch.randperm(len(self.qs_target))[:2]
            q_next = self.q_value(z_next_target, a_next, target=True)[idx].min(0).values
            y = rew + self.gamma * (1 - term) * q_next  # (B, T)

        # прокатка латента по модели
        z = self.model.encode(obs[:, 0])
        zs, loss_cons, loss_rew, loss_val = [], 0.0, 0.0, 0.0
        for t in range(t_len):
            zs.append(z)
            w = self.rho**t
            z_pred = self.model.next(z, act[:, t])
            loss_cons = loss_cons + w * ((z_pred - z_next_target[:, t]) ** 2).mean()
            loss_rew = (
                loss_rew
                + w * self.twohot.loss(self.model.reward_logits(z, act[:, t]), rew[:, t]).mean()
            )
            q_logits = self.model.q_logits(z, act[:, t])  # (K, B, bins)
            loss_val = (
                loss_val + w * self.twohot.loss(q_logits, y[:, t].expand(len(q_logits), -1)).mean()
            )
            z = z_pred
        loss_model = (c_cons * loss_cons + c_rew * loss_rew + c_val * loss_val) / t_len
        self.opt_model.zero_grad(set_to_none=True)
        loss_model.backward()
        grad_norm = nn.utils.clip_grad_norm_(self._model_params, self.grad_clip)
        self.opt_model.step()

        # policy prior на отсоединённых латентах
        zs_t = torch.stack(zs, 1).detach()  # (B, T, L)
        a_pi, logp = self.model.pi(zs_t)
        q_pi = self.q_value(zs_t, a_pi).mean(0)  # (B, T)
        self.scale.update(q_pi)
        weights = self.rho ** torch.arange(t_len, device=self.device, dtype=torch.float32)
        loss_pi = ((self.entropy_coef * logp - self.scale(q_pi)) * weights).mean()
        self.opt_pi.zero_grad(set_to_none=True)
        loss_pi.backward()
        nn.utils.clip_grad_norm_(self._policy_params, self.grad_clip)
        self.opt_pi.step()

        soft_update(self.qs_target, self.model.qs, self.tau)
        return {
            "loss_consistency": loss_cons.item() / t_len,
            "loss_reward": loss_rew.item() / t_len,
            "loss_value": loss_val.item() / t_len,
            "loss_pi": loss_pi.item(),
            "q": q_pi.mean().item(),
            "q_scale": self.scale.scale.item(),
            "entropy": -logp.mean().item(),
            "grad_norm": float(grad_norm),
        }

    # ------------------------------------------------------------------ сериализация
    def state_dict(self) -> dict:
        return {
            "model": self.model.state_dict(),
            "qs_target": self.qs_target.state_dict(),
            "scale": self.scale.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.model.load_state_dict(sd["model"])
        self.qs_target.load_state_dict(sd["qs_target"])
        self.scale.load_state_dict(sd["scale"])
