"""Интерактивное окно MuJoCo: рука под синусоидальными или случайными действиями.

    python scripts/00_view_arm.py             # arm5 (5 DoF + захват), синусоида
    python scripts/00_view_arm.py --arm3      # учебная 3-DoF рука из ReachEnv
    python scripts/00_view_arm.py --random    # случайные действия

В окне: ЛКМ — вращать камеру, ПКМ — двигать, колесо — зум, Space — пауза,
Ctrl+ЛКМ по телу — тянуть, клавиша 3 — показать оси шарниров (arm5).
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from rwm.envs import PickEnv, ReachEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm3", action="store_true")
    p.add_argument("--random", action="store_true")
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()

    env = (
        ReachEnv(render_mode="human", max_steps=200)
        if args.arm3
        else PickEnv(render_mode="human", max_steps=300)
    )
    n = env.action_space.shape[0]
    for ep in range(args.episodes):
        env.reset(seed=ep)
        t = 0
        while True:
            if args.random:
                a = env.action_space.sample()
            else:
                a = 0.5 * np.sin(0.05 * t + np.arange(n))
            _, _, terminated, truncated, info = env.step(a)
            t += 1
            time.sleep(env.dt)
            if terminated or truncated:
                print(f"episode {ep}: {info}")
                break
    env.close()


if __name__ == "__main__":
    main()
