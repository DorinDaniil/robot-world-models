"""Открыть интерактивное окно MuJoCo с рукой и погонять её случайными/синусоидальными действиями.

    python scripts/00_view_arm.py            # синусоида, цель меняется каждый эпизод
    python scripts/00_view_arm.py --random   # случайные действия

В окне: ЛКМ — вращать камеру, ПКМ — двигать, колесо — зум, Space — пауза, Ctrl+ЛКМ по телу — тянуть.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from rwm.envs import ReachEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--random", action="store_true")
    p.add_argument("--episodes", type=int, default=5)
    args = p.parse_args()

    env = ReachEnv(render_mode="human", max_steps=200)
    dt = env.dt
    for ep in range(args.episodes):
        env.reset(seed=ep)
        t = 0
        while True:
            if args.random:
                a = env.action_space.sample()
            else:
                a = np.sin(0.05 * t + np.array([0.0, 1.0, 2.0]))
            _, _, terminated, truncated, info = env.step(a)
            t += 1
            time.sleep(dt)
            if terminated or truncated:
                print(f"episode {ep}: final dist={info['dist']:.3f} success={info['success']}")
                break
    env.close()


if __name__ == "__main__":
    main()
