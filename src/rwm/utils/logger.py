"""Логгер эксперимента: CSV + консоль, опционально TensorBoard.

Один запуск — один каталог runs/<name>/seed<k>/ с файлами:
    config.yaml   — конфиг запуска (для воспроизведения);
    metrics.csv   — одна строка на вызов log(): step и все метрики (пустые, если нет);
    ckpt.pt       — чекпоинт агента.
Кривые по нескольким сидам строятся из metrics.csv скриптом scripts/12_plot.py.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import yaml


class Logger:
    def __init__(self, run_dir: str | Path, config: dict | None = None, tensorboard: bool = False):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        if config is not None:
            with open(self.dir / "config.yaml", "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        self._csv_path = self.dir / "metrics.csv"
        self._fields: list[str] = []
        self._rows: list[dict] = []
        self._tb = None
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(str(self.dir))
        self._t0 = time.time()

    def log(self, step: int, metrics: dict[str, float], echo: bool = True) -> None:
        row = {"step": step, "time": round(time.time() - self._t0, 1), **metrics}
        self._rows.append(row)
        for k in row:
            if k not in self._fields:
                self._fields.append(k)
        self._flush()
        if self._tb is not None:
            for k, v in metrics.items():
                self._tb.add_scalar(k, v, step)
        if echo:
            items = "  ".join(
                f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()
            )
            print(f"[{step:>8d}] {items}", flush=True)

    def _flush(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._fields)
            w.writeheader()
            w.writerows(self._rows)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
