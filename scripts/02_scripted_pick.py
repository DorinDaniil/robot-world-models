"""Аналитический контроллер захвата на PickEnv: доля успехов и gif.

python scripts/02_scripted_pick.py --episodes 20
python scripts/02_scripted_pick.py --episodes 3 --gif runs/pick.gif
python scripts/02_scripted_pick.py --view          # интерактивное окно
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from rwm.control import ScriptedPick, run_episode
from rwm.envs import PickEnv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--gif", type=str, default=None)
    p.add_argument("--view", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.view:
        env = PickEnv(render_mode="human")
        ctrl = ScriptedPick(env)
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep)
            ctrl.reset()
            while True:
                _, _, terminated, truncated, info = env.step(ctrl.act())
                time.sleep(env.dt)
                if terminated or truncated:
                    print(f"episode {ep}: lift={info['lift']:.3f} success={info['success']}")
                    break
        env.close()
        return

    env = PickEnv()
    results, frames = [], []
    for ep in range(args.episodes):
        ok, f = run_episode(env, seed=args.seed + ep, render=args.gif is not None)
        results.append(ok)
        frames += f
    print(f"episodes: {args.episodes}  success: {100 * np.mean(results):.1f} %")
    if args.gif:
        import imageio

        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.gif, frames, fps=25, loop=0)
        print(f"saved {args.gif} ({len(frames)} frames)")
    env.close()


if __name__ == "__main__":
    main()
