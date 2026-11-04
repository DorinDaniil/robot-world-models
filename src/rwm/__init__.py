"""rwm — Robot World Models.

Учебный курс и код: обучение управлению манипулятором в симуляции (SAC, TD-MPC2, Dreamer),
кинематика в формулировке произведения экспонент, языковые команды. Лекции — docs/lectures/.
"""

from pathlib import Path

__version__ = "0.1.0"

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
CONFIGS = ROOT / "configs"
