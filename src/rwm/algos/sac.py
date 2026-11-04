"""Soft Actor-Critic (Haarnoja et al., 2018) — алгоритм 1.1 из лекции 1.

Компоненты: актёр π_θ (MLP → TanhGaussian), два критика Q_φ1, Q_φ2 и их EMA-копии,
температура α с автонастройкой под целевую энтропию −dim(A).

Один вызов update() = один шаг по каждому из трёх лоссов:
    L_Q(φ_i) = E[(Q_φi(s,a) − y)²],  y = r + γ(1−d)[min_i Q_φ̄i(s',a') − α log π(a'|s')]
    L_π(θ)   = E[α log π_θ(ã|s) − min_i Q_φi(s,ã)],  ã = f_θ(s, ε)
    L(α)     = E[−α (log π_θ(ã|s) + H̄)]
Градиент по α берётся через log α, чтобы α оставалась положительной.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rwm.data.replay import ReplayBuffer
from rwm.models.nets import TanhGaussian, mlp, soft_update


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.trunk = mlp(obs_dim, (hidden,), hidden)
        self.head = TanhGaussian(hidden, act_dim)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        return self.head(torch.relu(self.trunk(obs)), deterministic)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, (hidden, hidden), 1)
        self.q2 = mlp(obs_dim + act_dim, (hidden, hidden), 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class SAC:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden: int = 256,
        gamma: float = 0.99,
        lr: float = 3e-4,
        tau: float = 0.005,
        batch_size: int = 256,
        init_alpha: float = 0.1,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma, self.tau, self.batch_size = gamma, tau, batch_size
        self.actor = Actor(obs_dim, act_dim, hidden).to(self.device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.log_alpha = torch.tensor(np.log(init_alpha), device=self.device, requires_grad=True)
        self.target_entropy = -float(act_dim)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.opt_alpha = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def reset(self) -> None:
        """SAC не имеет состояния между шагами; метод нужен для единого интерфейса агентов."""

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, _ = self.actor(obs_t, deterministic)
        return a.squeeze(0).cpu().numpy()

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        b = buffer.sample(self.batch_size)
        obs, act, rew, next_obs, term = (
            b["obs"].float(),
            b["action"],
            b["reward"],
            b["next_obs"].float(),
            b["terminated"],
        )
        alpha = self.alpha.detach()

        # --- критик
        with torch.no_grad():
            next_a, next_logp = self.actor(next_obs)
            q1_t, q2_t = self.critic_target(next_obs, next_a)
            y = rew + self.gamma * (1 - term) * (torch.min(q1_t, q2_t) - alpha * next_logp)
        q1, q2 = self.critic(obs, act)
        loss_q = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
        self.opt_critic.zero_grad(set_to_none=True)
        loss_q.backward()
        self.opt_critic.step()

        # --- актёр (критик заморожен: градиент идёт только в θ)
        a_new, logp = self.actor(obs)
        q1_pi, q2_pi = self.critic(obs, a_new)
        loss_pi = (alpha * logp - torch.min(q1_pi, q2_pi)).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss_pi.backward()
        self.opt_actor.step()

        # --- температура
        loss_alpha = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.opt_alpha.zero_grad(set_to_none=True)
        loss_alpha.backward()
        self.opt_alpha.step()

        soft_update(self.critic_target, self.critic, self.tau)
        return {
            "loss_q": loss_q.item(),
            "loss_pi": loss_pi.item(),
            "alpha": alpha.item(),
            "entropy": -logp.mean().item(),
            "q": q1.mean().item(),
        }

    # ------------------------------------------------------------------ сериализация
    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.critic_target.load_state_dict(sd["critic_target"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
