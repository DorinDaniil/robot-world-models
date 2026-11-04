"""Единый цикл обучения для off-policy агентов (SAC, TD-MPC2, Dreamer).

Все агенты реализуют интерфейс
    reset()                                  — начало эпизода (сброс плана / скрытого состояния)
    act(obs, deterministic=False) -> action  — действие по наблюдению
    update(buffer) -> dict                   — один шаг обучения, сам выбирает батч из буфера
    state_dict() / load_state_dict(sd)
поэтому цикл один: сбор перехода → запись в буфер → UTD шагов update → периодическая оценка.

Конфиг (YAML, см. configs/): секции env, train, agent, log; алгоритм задаётся ключом algo.
"""

from __future__ import annotations

import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import rwm.envs  # noqa: F401  (регистрация сред)
from rwm.algos import make_agent
from rwm.data.replay import ReplayBuffer
from rwm.utils.logger import Logger
from rwm.utils.seeding import set_seed


def make_env(env_cfg: dict, **kwargs) -> gym.Env:
    return gym.make(env_cfg["id"], **{**env_cfg.get("kwargs", {}), **kwargs})


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


@torch.no_grad()
def evaluate(agent, env: gym.Env, episodes: int, seed: int, record: bool = False) -> dict:
    """Детерминированная политика на непересекающихся с обучением сидах."""
    returns, successes, frames = [], [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        agent.reset()
        ret, success, done = 0.0, False, False
        while not done:
            obs, r, terminated, truncated, info = env.step(agent.act(obs, deterministic=True))
            ret += r
            success = success or bool(info.get("success", False))
            done = terminated or truncated
            if record and ep == 0:
                frames.append(env.unwrapped.render_rgb(240, 320))
        returns.append(ret)
        successes.append(success)
    out = {"eval/return": float(np.mean(returns)), "eval/success": float(np.mean(successes))}
    if record:
        out["frames"] = frames
    return out


def train(cfg: dict) -> Path:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    tc = cfg["train"]
    device = resolve_device(tc.get("device", "auto"))

    env = make_env(cfg["env"])
    eval_env = make_env(cfg["env"])
    obs_space, act_space = env.observation_space, env.action_space
    act_dim = int(np.prod(act_space.shape))

    buffer = ReplayBuffer(
        obs_space.shape, act_dim, tc["buffer_size"], obs_space.dtype, device=device, seed=seed
    )
    agent = make_agent(cfg["algo"], obs_space, act_space, cfg.get("agent", {}), device)

    run_dir = Path(cfg["log"]["dir"]) / cfg["log"]["name"] / f"seed{seed}"
    logger = Logger(run_dir, cfg, tensorboard=cfg["log"].get("tensorboard", False))
    print(f"run: {run_dir}  device: {device}  obs: {obs_space.shape}  act: {act_dim}")

    obs, _ = env.reset(seed=seed)
    agent.reset()
    ep_return, ep_count, train_metrics = 0.0, 0, {}
    t0 = time.time()
    for step in range(1, tc["total_steps"] + 1):
        if step <= tc["seed_steps"]:
            action = act_space.sample()
        else:
            action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, terminated)
        ep_return += reward
        obs = next_obs
        if terminated or truncated:
            buffer.end_episode(obs)
            ep_count += 1
            logger.log(
                step,
                {"train/return": ep_return, "train/success": float(info.get("success", False))},
                echo=False,
            )
            obs, _ = env.reset()
            agent.reset()
            ep_return = 0.0

        if step > tc["seed_steps"]:
            for _ in range(tc.get("utd", 1)):
                train_metrics = agent.update(buffer)

        if step % tc["log_every"] == 0 and train_metrics:
            sps = tc["log_every"] / (time.time() - t0)
            t0 = time.time()
            logger.log(step, {**train_metrics, "sps": sps})

        if step % tc["eval_every"] == 0:
            metrics = evaluate(agent, eval_env, tc["eval_episodes"], seed=10_000 + seed)
            logger.log(step, metrics)
            torch.save({"cfg": cfg, "agent": agent.state_dict()}, run_dir / "ckpt.pt")

    logger.close()
    env.close()
    eval_env.close()
    return run_dir
