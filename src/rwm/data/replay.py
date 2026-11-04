"""Кольцевой буфер переходов с поддержкой эпизодов и выборкой подпоследовательностей.

Хранение: массивы длины `capacity` для obs, action, reward, terminated, episode id и флага
valid. Шаг t эпизода k занимает один слот: (o_t, a_t, r_t, term_t, ep=k, valid=True).
После последнего шага эпизода добавляется один «хвостовой» слот с финальным наблюдением
o_T и valid=False — так следующее наблюдение любого шага всегда лежит в слоте i+1, и
отдельный массив next_obs не нужен (для изображений это вдвое экономит память).

Выборка:
    sample(B)            — переходы (o, a, r, o', term) для SAC;
    sample_sequences(B,T) — окна (o_{0..T}, a_{0..T-1}, r, term) внутри одного эпизода
                           для TD-MPC2 и Dreamer.
Окно считается допустимым, если все T слотов valid и принадлежат одному эпизоду, а слот T
(следующее наблюдение) — тому же эпизоду. При перезаписи по кольцу старые слоты получают
новый ep id и автоматически выпадают из допустимых.

Усечение (truncated) и завершение (terminated) различаются: хранится только terminated,
по нему маскируется бутстрап; усечённый эпизод просто заканчивается хвостовым слотом.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        obs_shape: tuple[int, ...],
        act_dim: int,
        capacity: int,
        obs_dtype: np.dtype = np.float32,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ):
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(device)
        self.obs = np.zeros((capacity, *obs_shape), dtype=obs_dtype)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.term = np.zeros(capacity, dtype=np.float32)
        self.ep = np.full(capacity, -1, dtype=np.int64)
        self.valid = np.zeros(capacity, dtype=bool)
        self.ptr = 0
        self.size = 0
        self.num_steps = 0
        self._ep_id = 0

    def __len__(self) -> int:
        return self.num_steps

    def add(self, obs, action, reward: float, terminated: bool) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = action
        self.rew[i] = reward
        self.term[i] = float(terminated)
        self.ep[i] = self._ep_id
        self.valid[i] = True
        self._advance()
        self.num_steps += 1

    def end_episode(self, final_obs) -> None:
        """Записать финальное наблюдение эпизода в хвостовой слот и начать новый эпизод."""
        i = self.ptr
        self.obs[i] = final_obs
        self.ep[i] = self._ep_id
        self.valid[i] = False
        self._advance()
        self._ep_id += 1

    def _advance(self) -> None:
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        # слот, в который будем писать следующим, уже недействителен как источник выборки
        self.valid[self.ptr] = False

    # ------------------------------------------------------------------ выборка
    def _to(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, device=self.device)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        rng = self.rng
        idx = np.empty(0, dtype=np.int64)
        while idx.size < batch_size:
            cand = rng.integers(0, self.size, size=2 * batch_size)
            nxt = (cand + 1) % self.capacity
            ok = self.valid[cand] & (self.ep[nxt] == self.ep[cand])
            idx = np.concatenate([idx, cand[ok]])
        idx = idx[:batch_size]
        nxt = (idx + 1) % self.capacity
        return {
            "obs": self._to(self.obs[idx]),
            "action": self._to(self.act[idx]),
            "reward": self._to(self.rew[idx]),
            "next_obs": self._to(self.obs[nxt]),
            "terminated": self._to(self.term[idx]),
        }

    def sample_sequences(self, batch_size: int, horizon: int) -> dict[str, torch.Tensor]:
        """Окна длины horizon: obs (B, T+1, ...), action/reward/terminated (B, T)."""
        rng = self.rng
        starts = np.empty(0, dtype=np.int64)
        offsets = np.arange(horizon + 1)
        while starts.size < batch_size:
            cand = rng.integers(0, self.size, size=4 * batch_size)
            win = (cand[:, None] + offsets[None, :]) % self.capacity  # (N, T+1)
            same_ep = (self.ep[win] == self.ep[cand][:, None]).all(1)
            ok = same_ep & self.valid[win[:, :-1]].all(1)
            starts = np.concatenate([starts, cand[ok]])
        win = (starts[:batch_size, None] + offsets[None, :]) % self.capacity
        steps = win[:, :-1]
        return {
            "obs": self._to(self.obs[win]),
            "action": self._to(self.act[steps]),
            "reward": self._to(self.rew[steps]),
            "terminated": self._to(self.term[steps]),
        }
