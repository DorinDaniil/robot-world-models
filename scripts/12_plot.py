"""Кривые обучения по нескольким запускам: среднее ± стандартная ошибка по сидам.

    python scripts/12_plot.py runs/sac_reach runs/tdmpc2_reach --metric eval/success -o curves.png

Каждый аргумент — каталог с подкаталогами seed*/metrics.csv. Требуется matplotlib.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_runs(run_dir: Path, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (steps, values[seeds, steps]); сиды выравниваются по общим шагам."""
    series = []
    for f in sorted(run_dir.glob("seed*/metrics.csv")):
        with open(f, encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get(metric)]
        series.append({int(r["step"]): float(r[metric]) for r in rows})
    if not series:
        raise FileNotFoundError(f"no seed*/metrics.csv with {metric!r} in {run_dir}")
    steps = sorted(set.intersection(*(set(s) for s in series)))
    return np.array(steps), np.array([[s[t] for t in steps] for s in series])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--metric", default="eval/success")
    p.add_argument("-o", "--out", default=None)
    args = p.parse_args()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for run in args.runs:
        steps, vals = load_runs(Path(run), args.metric)
        mean, sem = vals.mean(0), vals.std(0, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        ax.plot(steps, mean, label=f"{Path(run).name} (n={len(vals)})")
        ax.fill_between(steps, mean - sem, mean + sem, alpha=0.25)
    ax.set_xlabel("шаги среды")
    ax.set_ylabel(args.metric)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=150)
        print(f"saved {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
