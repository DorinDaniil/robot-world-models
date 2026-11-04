"""Dreamer: латентная модель мира (RSSM) и актёр-критик в воображении — лекция 4.

Следует рецепту DreamerV3 (Hafner et al., 2023) в упрощённом виде.

Модель мира на батче последовательностей (B, T+1 наблюдений, T переходов):
    h_t = f(h_{t−1}, z_{t−1}, a_{t−1})             GRU, детерминированный путь
    prior  p(z_t | h_t),  posterior q(z_t | h_t, e(o_t))   категориальные латенты
    L_wm = Σ_t [ −log p(o_t | h_t, z_t) − log p(r_{t−1} | h_t, z_t) − log p(c_{t−1} | h_t, z_t)
                 + β_dyn max(1, KL(sg q ‖ p)) + β_rep max(1, KL(q ‖ sg p)) ]
Награда и продолжение c = 1 − terminated относятся к переходу, *приводящему* в состояние t
(как в DreamerV3), поэтому их головы читаются со сдвигом на один шаг.

Актёр-критик в воображении: от каждого апостериорного состояния батча прогон по prior на
горизонт H с действиями актёра; λ-доходность R^λ с бутстрапом EMA-копией критика;
критик — two-hot регрессия на sg(R^λ); актёр — градиент через динамику (репараметризация,
как в DreamerV3 для непрерывных действий) по нормированной R^λ плюс энтропийный бонус.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rwm.data.replay import ReplayBuffer
from rwm.models.nets import (
    CNNDecoder,
    CNNEncoder,
    RunningScale,
    TanhGaussian,
    TwoHot,
    mlp,
    soft_update,
    symlog,
)


def lambda_return(
    reward: torch.Tensor,
    value: torch.Tensor,
    cont: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """R^λ_τ = r_{τ+1} + γ c_{τ+1} [(1−λ) v_{τ+1} + λ R^λ_{τ+1}],  R^λ_H = v_H.

    reward, cont: (H, N) — награда и продолжение на входе в состояния 1..H;
    value: (H+1, N) — ценности состояний 0..H. Возвращает (H, N) для состояний 0..H−1.
    """
    out = []
    nxt = value[-1]
    for t in reversed(range(reward.shape[0])):
        nxt = reward[t] + gamma * cont[t] * ((1 - lam) * value[t + 1] + lam * nxt)
        out.append(nxt)
    return torch.stack(out[::-1])


class RSSM(nn.Module):
    """Рекуррентная модель состояния: GRU + категориальный латент stoch × classes."""

    def __init__(
        self,
        act_dim: int,
        embed_dim: int,
        deter: int,
        hidden: int,
        stoch: int,
        classes: int,
        unimix: float = 0.01,
    ):
        super().__init__()
        self.deter, self.stoch, self.classes, self.unimix = deter, stoch, classes, unimix
        self.inp = nn.Sequential(
            nn.Linear(stoch * classes + act_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()
        )
        self.gru = nn.GRUCell(hidden, deter)
        self.prior = mlp(deter, hidden, stoch * classes, act=nn.SiLU)
        self.post = mlp(deter + embed_dim, hidden, stoch * classes, act=nn.SiLU)

    @property
    def feat_dim(self) -> int:
        return self.deter + self.stoch * self.classes

    def initial(self, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(n, self.deter, device=device), torch.zeros(
            n, self.stoch * self.classes, device=device
        )

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        """softmax с примесью 1 % равномерного распределения (unimix): KL остаётся конечным."""
        p = torch.softmax(logits.view(*logits.shape[:-1], self.stoch, self.classes), -1)
        return (1 - self.unimix) * p + self.unimix / self.classes

    def sample(self, probs: torch.Tensor) -> torch.Tensor:
        """Straight-through: z = onehot(sample) + probs − sg(probs)."""
        idx = torch.multinomial(probs.flatten(0, -2), 1).view(*probs.shape[:-1])
        onehot = F.one_hot(idx, self.classes).float()
        return (onehot + probs - probs.detach()).flatten(-2)

    def img_step(self, h, z, a):
        h = self.gru(self.inp(torch.cat([z, a], -1)), h)
        prior = self.probs(self.prior(h))
        return h, prior, self.sample(prior)

    def obs_step(self, h, z, a, embed):
        h, prior, _ = self.img_step(h, z, a)
        post = self.probs(self.post(torch.cat([h, embed], -1)))
        return h, prior, post, self.sample(post)

    @staticmethod
    def kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """KL(p ‖ q) между наборами категориальных распределений, сумма по группам."""
        return (p * (p.log() - q.log())).sum(-1).sum(-1)


class Actor(nn.Module):
    def __init__(self, feat_dim: int, hidden: int, act_dim: int):
        super().__init__()
        self.trunk = mlp(feat_dim, hidden, hidden, act=nn.SiLU)
        self.head = TanhGaussian(hidden, act_dim)

    def forward(self, feat: torch.Tensor, deterministic: bool = False):
        return self.head(F.silu(self.trunk(feat)), deterministic)


class Dreamer:
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        act_dim: int,
        *,
        deter: int = 512,
        hidden: int = 512,
        stoch: int = 32,
        classes: int = 32,
        depth: int = 32,
        batch_size: int = 16,
        seq_len: int = 64,
        horizon: int = 15,
        gamma: float = 0.997,
        lam: float = 0.95,
        beta_dyn: float = 0.5,
        beta_rep: float = 0.1,
        free_bits: float = 1.0,
        entropy_coef: float = 3e-4,
        lr_model: float = 1e-4,
        lr_actor: float = 3e-5,
        lr_critic: float = 3e-5,
        critic_tau: float = 0.02,
        update_every: int | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.pixels = len(obs_shape) == 3
        self.act_dim = act_dim
        self.batch_size, self.seq_len, self.horizon = batch_size, seq_len, horizon
        self.gamma, self.lam = gamma, lam
        self.beta_dyn, self.beta_rep, self.free_bits = beta_dyn, beta_rep, free_bits
        self.entropy_coef, self.critic_tau = entropy_coef, critic_tau
        self.update_every = update_every or (2 if self.pixels else 1)
        self._calls, self._last_metrics = 0, {}

        if self.pixels:
            self.encoder = CNNEncoder(depth)
            embed_dim = self.encoder.out_dim
        else:
            self.encoder = mlp(obs_shape[0], hidden, hidden, act=nn.SiLU)
            embed_dim = hidden
        self.rssm = RSSM(act_dim, embed_dim, deter, hidden, stoch, classes)
        fd = self.rssm.feat_dim
        self.decoder = (
            CNNDecoder(fd, depth) if self.pixels else mlp(fd, hidden, obs_shape[0], act=nn.SiLU)
        )
        self.twohot = TwoHot(255, -20.0, 20.0)
        self.reward = mlp(fd, hidden, 255, act=nn.SiLU)
        self.cont = mlp(fd, hidden, 1, act=nn.SiLU)
        self.actor = Actor(fd, hidden, act_dim)
        self.critic = mlp(fd, hidden, 255, act=nn.SiLU)
        self.critic_target = mlp(fd, hidden, 255, act=nn.SiLU)
        self.return_scale = RunningScale()

        self.model = nn.ModuleList([self.encoder, self.rssm, self.decoder, self.reward, self.cont])
        self.modules = nn.ModuleList(
            [
                self.model,
                self.actor,
                self.critic,
                self.critic_target,
                self.twohot,
                self.return_scale,
            ]
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)
        self.opt_model = torch.optim.Adam(self.model.parameters(), lr=lr_model, eps=1e-8)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor, eps=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic, eps=1e-5)
        self.reset()

    # ------------------------------------------------------------------ взаимодействие
    def reset(self) -> None:
        self._h, self._z = self.rssm.initial(1, self.device)
        self._a = torch.zeros(1, self.act_dim, device=self.device)

    def _embed(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs) if self.pixels else self.encoder(symlog(obs.float()))

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(np.asarray(obs), device=self.device).unsqueeze(0)
        self._h, _, _, self._z = self.rssm.obs_step(self._h, self._z, self._a, self._embed(obs_t))
        a, _ = self.actor(torch.cat([self._h, self._z], -1), deterministic)
        self._a = a
        return a.squeeze(0).cpu().numpy()

    # ------------------------------------------------------------------ обучение
    def _observe(self, obs: torch.Tensor, action: torch.Tensor):
        """Posterior-прогон по последовательности: obs (B, L, ...), action (B, L−1, A).

        h, z инициализируются нулями в начале каждого окна (окна не пересекают границы
        эпизодов, поэтому это приближает механизм is_first DreamerV3). Возвращает
        feat (B, L, F), prior и posterior (B, L, stoch, classes).
        """
        B, L = obs.shape[:2]
        embed = self._embed(obs.flatten(0, 1)).view(B, L, -1)
        h, z = self.rssm.initial(B, self.device)
        a = torch.zeros(B, self.act_dim, device=self.device)
        feats, priors, posts = [], [], []
        for t in range(L):
            h, prior, post, z = self.rssm.obs_step(h, z, a, embed[:, t])
            feats.append(torch.cat([h, z], -1))
            priors.append(prior)
            posts.append(post)
            if t < L - 1:
                a = action[:, t]
        return torch.stack(feats, 1), torch.stack(priors, 1), torch.stack(posts, 1)

    def _recon_loss(self, feat: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """−log p(o | feat) с точностью до константы: MSE, сумма по компонентам наблюдения."""
        pred = self.decoder(feat.flatten(0, 1)).reshape(*feat.shape[:2], -1)
        target = obs.float().flatten(2) / 255.0 if self.pixels else symlog(obs.float())
        return ((pred - target) ** 2).sum(-1)

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        self._calls += 1
        if self._calls % self.update_every:
            return self._last_metrics
        b = buffer.sample_sequences(self.batch_size, self.seq_len)
        obs, action, reward, term = b["obs"], b["action"], b["reward"], b["terminated"]

        # --- модель мира
        feat, prior, post = self._observe(obs, action)
        loss_recon = self._recon_loss(feat, obs).mean()
        loss_rew = self.twohot.loss(self.reward(feat[:, 1:]), reward).mean()
        loss_cont = F.binary_cross_entropy_with_logits(self.cont(feat[:, 1:]).squeeze(-1), 1 - term)
        kl_dyn = self.rssm.kl(post.detach(), prior)
        kl_rep = self.rssm.kl(post, prior.detach())
        # free bits применяются поэлементно (к каждой паре (b, t)), затем усреднение — как в V3
        loss_kl = (
            self.beta_dyn * kl_dyn.clamp_min(self.free_bits)
            + self.beta_rep * kl_rep.clamp_min(self.free_bits)
        ).mean()
        loss_model = loss_recon + loss_rew + loss_cont + loss_kl
        self.opt_model.zero_grad(set_to_none=True)
        loss_model.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1000.0)
        self.opt_model.step()

        # --- воображение от всех апостериорных состояний батча
        h = feat[:, :, : self.rssm.deter].detach().flatten(0, 1)
        z = feat[:, :, self.rssm.deter :].detach().flatten(0, 1)
        feats, logps = [torch.cat([h, z], -1)], []
        for _ in range(self.horizon):
            a, logp = self.actor(feats[-1])
            h, _, z = self.rssm.img_step(h, z, a)
            feats.append(torch.cat([h, z], -1))
            logps.append(logp)
        feats = torch.stack(feats)  # (H+1, N, F)
        logps = torch.stack(logps)  # (H, N)
        r_hat = self.twohot.decode(self.reward(feats[1:]))
        c_hat = torch.sigmoid(self.cont(feats[1:]).squeeze(-1))
        v_target = self.twohot.decode(self.critic_target(feats))
        ret = lambda_return(r_hat, v_target, c_hat, self.gamma, self.lam)  # (H, N)
        weight = torch.cat(
            [torch.ones_like(c_hat[:1]), torch.cumprod(self.gamma * c_hat[:-1], 0)]
        ).detach()

        # --- актёр: градиент через динамику по нормированной доходности + энтропия
        self.return_scale.update(ret)
        loss_actor = -(weight * (self.return_scale(ret) + self.entropy_coef * (-logps))).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss_actor.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0)
        self.opt_actor.step()

        # --- критик: two-hot регрессия на sg(R^λ) + регуляризация к EMA-копии
        feats_d = feats[:-1].detach()
        logits = self.critic(feats_d)
        with torch.no_grad():
            target_probs = torch.softmax(self.critic_target(feats_d), -1)
        loss_critic = self.twohot.loss(logits, ret.detach()) - (
            target_probs * torch.log_softmax(logits, -1)
        ).sum(-1)
        loss_critic = (weight * loss_critic).mean()
        self.opt_critic.zero_grad(set_to_none=True)
        loss_critic.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 100.0)
        self.opt_critic.step()
        soft_update(self.critic_target, self.critic, self.critic_tau)

        self._last_metrics = {
            "loss_recon": loss_recon.item(),
            "loss_reward": loss_rew.item(),
            "loss_cont": loss_cont.item(),
            "kl_dyn": kl_dyn.mean().item(),
            "kl_rep": kl_rep.mean().item(),
            "loss_actor": loss_actor.item(),
            "loss_critic": loss_critic.item(),
            "imag_return": ret.mean().item(),
            "entropy": -logps.mean().item(),
            "return_scale": self.return_scale.scale.item(),
        }
        return self._last_metrics

    # ------------------------------------------------------------------ диагностика
    @torch.no_grad()
    def open_loop_error(
        self, buffer: ReplayBuffer, context: int = 5, horizon: int = 20, batch_size: int = 32
    ) -> dict[str, list[float]]:
        """Ошибка предсказания в открытом цикле (L4, §4.6).

        Posterior по первым `context` наблюдениям, затем `horizon` шагов по prior с реальными
        действиями. Возвращает средние |r̂ − r| и ошибку реконструкции по шагам горизонта.
        """
        b = buffer.sample_sequences(batch_size, context + horizon - 1)
        obs, action, reward = b["obs"], b["action"], b["reward"]
        feat, _, _ = self._observe(obs[:, :context], action[:, : context - 1])
        h, z = feat[:, -1, : self.rssm.deter], feat[:, -1, self.rssm.deter :]
        err_r, err_o = [], []
        for k in range(horizon):
            i = context - 1 + k  # индекс действия; приводит в состояние i+1
            h, _, z = self.rssm.img_step(h, z, action[:, i])
            f = torch.cat([h, z], -1)
            err_r.append((self.twohot.decode(self.reward(f)) - reward[:, i]).abs().mean().item())
            err_o.append(self._recon_loss(f.unsqueeze(1), obs[:, i + 1 : i + 2]).mean().item())
        return {"reward_abs_error": err_r, "recon_error": err_o}

    # ------------------------------------------------------------------ сериализация
    def state_dict(self) -> dict:
        return {"modules": self.modules.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.modules.load_state_dict(sd["modules"])
        self.reset()
