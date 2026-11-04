"""Обучение агента по конфигу.

    python scripts/10_train.py -c configs/sac_reach.yaml
    python scripts/10_train.py -c configs/sac_reach.yaml seed=1 train.total_steps=200000
    python scripts/10_train.py -c configs/tdmpc2_pick.yaml log.tensorboard=true

Переопределения — в точечной нотации key=value. Результат: runs/<name>/seed<k>/metrics.csv,
ckpt.pt, config.yaml.
"""

from __future__ import annotations

import argparse

from rwm.train import train
from rwm.utils.config import load_config


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", required=True)
    p.add_argument("overrides", nargs="*", help="key.subkey=value")
    args = p.parse_args()
    train(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
