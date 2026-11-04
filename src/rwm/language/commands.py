"""Разбор команд на естественном языке в формальное задание для манипулятора.

Команда — текст на русском или английском. Результат — `Command`: глагол из конечного
набора и параметры цели. Разбор детерминированный (грамматика на ключевых словах и
числах), поэтому поведение полностью проверяемо; это уровень «заземления» (grounding)
языка в символьное задание, который в VLA-моделях заменяется обученной сетью (L7 §7.3).

Поддерживаемые команды (примеры):
    «подними кубик», «возьми куб»                          → pick
    «положи кубик слева / справа / ближе / дальше»          → place, цель относительно кубика
    «положи кубик в точку 0.3 -0.1», «перенеси в 30 -10 см» → place, абсолютная цель (м)
    «подними на 10 см»                                      → pick с высотой подъёма
    «открой захват», «закрой захват»                        → open / close
    «домой», «исходное положение»                           → home
Английские эквиваленты: pick up / lift / grab, put / place / move ... left / right /
closer / farther / to x y, open / close gripper, home.

Система координат — робота: x — от основания вперёд, y — влево (если смотреть от
основания), z — вверх. «Ближе/дальше» — вдоль x к основанию / от него.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

Vec = tuple[float, float]

# Порядок важен: «положи» включает захват, поэтому place проверяется раньше pick.
# Стемы с границей слова: «closer» не должен попадать в close.
VERBS: dict[str, tuple[str, ...]] = {
    "place": (
        r"\bполож",
        r"\bпостав",
        r"\bперенес",
        r"\bперемест",
        r"\bпередвин",
        r"\bput\b",
        r"\bplace\b",
        r"\bmove\b",
    ),
    "pick": (
        r"\bподним",
        r"\bподн",
        r"\bвозьм",
        r"\bвзять",
        r"\bсхват",
        r"\bpick\b",
        r"\blift\b",
        r"\bgrab\b",
        r"\btake\b",
        r"\braise\b",
    ),
    "home": (r"\bдомой", r"\bисходн", r"\bhome\b"),
    "open": (r"\bоткр", r"\bразожм", r"\bразомкн", r"\bopen\b", r"\brelease\b"),
    "close": (r"\bзакр", r"\bсожм", r"\bсомкн", r"\bclose\b"),
}

DIRECTIONS: dict[str, Vec] = {
    "слева": (0.0, 1.0),
    "налево": (0.0, 1.0),
    "влево": (0.0, 1.0),
    "left": (0.0, 1.0),
    "справа": (0.0, -1.0),
    "направо": (0.0, -1.0),
    "вправо": (0.0, -1.0),
    "right": (0.0, -1.0),
    "ближе": (-1.0, 0.0),
    "к себе": (-1.0, 0.0),
    "назад": (-1.0, 0.0),
    "closer": (-1.0, 0.0),
    "back": (-1.0, 0.0),
    "дальше": (1.0, 0.0),
    "вперёд": (1.0, 0.0),
    "вперед": (1.0, 0.0),
    "farther": (1.0, 0.0),
    "forward": (1.0, 0.0),
    "away": (1.0, 0.0),
}

DEFAULT_SHIFT = 0.10  # м, сдвиг «слева/справа/ближе/дальше» без явного расстояния
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


@dataclass
class Command:
    verb: str  # pick | place | open | close | home
    target: tuple[float, float] | None = None  # абсолютная цель (x, y), м
    offset: tuple[float, float] | None = None  # сдвиг относительно текущего положения кубика, м
    height: float | None = None  # высота подъёма, м (для pick)
    text: str = field(default="", repr=False)

    def resolve_target(self, cube_xy) -> tuple[float, float] | None:
        """Абсолютная цель с учётом относительного сдвига от положения кубика."""
        if self.target is not None:
            return self.target
        if self.offset is not None:
            return (float(cube_xy[0]) + self.offset[0], float(cube_xy[1]) + self.offset[1])
        return None


class ParseError(ValueError):
    pass


def _numbers(text: str) -> list[float]:
    return [float(n.replace(",", ".")) for n in NUMBER.findall(text)]


def _to_meters(values: list[float], text: str) -> list[float]:
    """Единицы: «см»/cm → делим на 100; «мм»/mm → на 1000; иначе — метры."""
    if re.search(r"\bсм\b|\bcm\b|сантиметр|centimet", text):
        return [v / 100 for v in values]
    if re.search(r"\bмм\b|\bmm\b|миллиметр|millimet", text):
        return [v / 1000 for v in values]
    return values


def parse(text: str) -> Command:
    t = text.lower().strip()
    if not t:
        raise ParseError("пустая команда")

    verb = next((v for v, pats in VERBS.items() if any(re.search(p, t) for p in pats)), None)
    if verb is None:
        raise ParseError(f"не распознан глагол в команде: {text!r}")

    nums = _to_meters(_numbers(t), t)
    cmd = Command(verb=verb, text=text)

    if verb == "place":
        direction = next((d for d in DIRECTIONS if d in t), None)
        if direction is not None:
            dist = nums[0] if nums else DEFAULT_SHIFT
            dx, dy = DIRECTIONS[direction]
            cmd.offset = (dx * dist, dy * dist)
        elif len(nums) >= 2:
            cmd.target = (nums[0], nums[1])
        else:
            raise ParseError("для «положи» нужна цель: направление или координаты x y")
    elif verb == "pick" and nums:
        cmd.height = nums[0]
    return cmd
