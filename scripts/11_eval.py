"""Оценка чекпоинта: доля успехов, доходность, gif или интерактивное окно.

python scripts/11_eval.py runs/sac_reach/seed0/ckpt.pt --episodes 20
python scripts/11_eval.py runs/sac_reach/seed0/ckpt.pt --gif runs/sac_reach.gif
python scripts/11_eval.py runs/sac_reach/seed0/ckpt.pt --view
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from rwm.algos import make_agent
from rwm.train import evaluate, make_env, resolve_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=20_000)
    p.add_argument("--gif", type=str, default=None)
    p.add_argument("--view", action="store_true")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    env = make_env(cfg["env"], render_mode="human" if args.view else None)
    agent = make_agent(
        cfg["algo"],
        env.observation_space,
        env.action_space,
        cfg.get("agent", {}),
        resolve_device(args.device),
    )
    agent.load_state_dict(ck["agent"])

    if args.view:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=args.seed + ep)
            agent.reset()
            done = False
            while not done:
                obs, _, terminated, truncated, info = env.step(agent.act(obs, deterministic=True))
                time.sleep(env.unwrapped.dt)
                done = terminated or truncated
            print(f"episode {ep}: {info}")
        env.close()
        return

    out = evaluate(agent, env, args.episodes, args.seed, record=args.gif is not None)
    print(
        f"episodes: {args.episodes}  return: {out['eval/return']:.2f}  "
        f"success: {100 * out['eval/success']:.0f} %"
    )
    if args.gif:
        import imageio

        Path(args.gif).parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(args.gif, out["frames"], fps=25, loop=0)
        print(f"saved {args.gif}")
    env.close()


if __name__ == "__main__":
    main()
