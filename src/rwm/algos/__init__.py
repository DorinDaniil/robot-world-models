"""Алгоритмы обучения: SAC (L1), TD-MPC2 (L5), Dreamer (L4). Фабрика make_agent."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces


def make_agent(name: str, obs_space: spaces.Space, act_space: spaces.Space, cfg: dict, device: str):
    act_dim = int(np.prod(act_space.shape))
    if name == "sac":
        from rwm.algos.sac import SAC

        if len(obs_space.shape) != 1:
            raise ValueError("SAC в курсе работает только с векторным наблюдением (obs_mode=state)")
        return SAC(obs_space.shape[0], act_dim, device=device, **cfg)
    if name == "tdmpc2":
        from rwm.algos.tdmpc2 import TDMPC2

        return TDMPC2(obs_space.shape, act_dim, device=device, **cfg)
    if name == "dreamer":
        from rwm.algos.dreamer import Dreamer

        return Dreamer(obs_space.shape, act_dim, device=device, **cfg)
    raise ValueError(f"unknown algo {name!r}")


__all__ = ["make_agent"]
