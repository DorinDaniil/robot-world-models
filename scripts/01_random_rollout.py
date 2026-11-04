"""Санити-проверка среды без окна: случайная политика, статистика по эпизодам, gif с камеры.

    python scripts/01_random_rollout.py --episodes 20
    python scripts/01_random_rollout.py --gif runs/random.gif

Случайная политика должна давать низкий success rate (ориентир ~0–5 %). Если он высокий,
задача слишком простая или success_radius слишком большой.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rwm.envs import ReachEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--gif", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env = ReachEnv(render_mode="rgb_array")
    returns, successes, frames = [], [], []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        ret, done = 0.0, False
        while not done:
            obs, r, terminated, truncated, info = env.step(env.action_space.sample())
            ret += r
            done = terminated or truncated
            if args.gif and ep == 0:
                frames.append(env.render())
        returns.append(ret)
        successes.append(info["success"])

    print(f"episodes: {args.episodes}")
    print(f"return   : {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"success  : {100 * np.mean(successes):.1f} %")
    print(f"obs shape: {obs.shape}, action shape: {env.action_space.shape}")

    if args.gif:
        import imageio

        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.gif, frames, fps=env.metadata["render_fps"])
        print(f"saved {args.gif} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
