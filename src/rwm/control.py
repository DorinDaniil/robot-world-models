"""Аналитический контроллер манипулятора arm5 на основе обратной кинематики.

Используется как (а) проверка согласованности модели, актуаторов и физики контакта,
(б) верхняя оценка доли успехов для обучаемых методов, (в) источник демонстраций,
(г) исполнитель символьных команд из `rwm.language` (L7 §7.3).

План — список фаз (q_des, g, min_steps): целевые углы, команда захвата (−1 открыт,
+1 закрыт) и минимальная длительность фазы. На каждом шаге действие агента — ограниченный
шаг q_ref к q_des, то есть контроллер работает через тот же интерфейс действий, что и
обучаемая политика. Переход к следующей фазе — когда q_ref достиг q_des и рука пришла
(‖θ − q_des‖_∞ < 0.05) либо по таймауту 60 шагов.

Захват сверху: ось инструмента направлена вниз, θ5 задаётся отдельно функцией `grasp_yaw`,
так как при вертикальной оси инструмента оси шарниров 1 и 5 параллельны и IK не определяет
θ5 (L3 §3.4, п. 3).
"""

from __future__ import annotations

import numpy as np

from rwm.envs.pick import PickEnv
from rwm.kinematics import SerialChain, arm5_chain
from rwm.language.commands import Command

DOWN = np.array([0.0, 0.0, -1.0])
OPEN, CLOSED = -1.0, 1.0


def grasp_yaw(chain: SerialChain, theta: np.ndarray, cube_yaw: float) -> float:
    """Угол j5, при котором ось смыкания пальцев (ось y захвата) параллельна грани кубика.

    Кубик симметричен с периодом π/2, поэтому из всех решений берётся ближайшее к нулю.
    """
    best = None
    for k in range(-4, 5):
        cand = theta.copy()
        cand[4] = 0.0
        y_axis = chain.fk(cand)[:3, :3] @ np.array([0.0, 1.0, 0.0])
        phi = np.arctan2(y_axis[1], y_axis[0])
        delta = (phi - cube_yaw - k * np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
        if best is None or abs(delta) < abs(best):
            best = delta
    return float(best)


class ScriptedPick:
    """Контроллер pick / pick-and-place / open / close / home для PickEnv."""

    def __init__(self, env: PickEnv, hover: float = 0.10, grasp_z: float = 0.02):
        self.env = env
        self.chain = arm5_chain()
        self.hover = hover
        self.grasp_z = grasp_z
        self.limits = (env.q_low, env.q_high)
        self.plan: list[tuple[np.ndarray, float, int]] = []
        self.phase = 0
        self.steps_in_phase = 0

    # ------------------------------------------------------------------ планирование
    def _ik(self, pos: np.ndarray, theta0: np.ndarray, yaw: float) -> np.ndarray:
        q, _ = self.chain.ik(np.asarray(pos, float), theta0, DOWN, limits=self.limits)
        q[4] = yaw
        return q

    def _grasp_sequence(self, xy, yaw: float, theta0: np.ndarray, lift: float):
        """Фазы: над объектом → опуститься → сомкнуть → поднять на lift."""
        above = self._ik([xy[0], xy[1], self.grasp_z + self.hover], theta0, 0.0)
        above[4] = grasp_yaw(self.chain, above, yaw)
        down = self._ik([xy[0], xy[1], self.grasp_z], above, above[4])
        up = self._ik([xy[0], xy[1], self.grasp_z + lift], above, above[4])
        return [(above, OPEN, 0), (down, OPEN, 0), (down, CLOSED, 15), (up, CLOSED, 0)]

    def _release_sequence(self, xy, theta0: np.ndarray, yaw: float):
        """Фазы: над целью → опуститься → разомкнуть → подняться."""
        above = self._ik([xy[0], xy[1], self.grasp_z + self.hover], theta0, yaw)
        down = self._ik([xy[0], xy[1], self.grasp_z + 0.005], above, yaw)
        return [(above, CLOSED, 0), (down, CLOSED, 0), (down, OPEN, 10), (above, OPEN, 0)]

    def reset(self, command: Command | None = None) -> None:
        """Построить план по текущему состоянию среды; без команды — подъём кубика."""
        env = self.env
        command = command or Command("pick")
        cube, yaw, q0 = env.cube_pos, env.cube_yaw, env.q_arm
        grip_now = CLOSED if env.data.ctrl[5] > env.g_max / 2 else OPEN

        if command.verb == "pick":
            lift = command.height if command.height is not None else self.hover
            self.plan = self._grasp_sequence(cube[:2], yaw, q0, lift)
        elif command.verb == "place":
            target = command.resolve_target(cube[:2])
            self.plan = self._grasp_sequence(cube[:2], yaw, q0, self.hover)
            self.plan += self._release_sequence(target, self.plan[-1][0], self.plan[-1][0][4])
        elif command.verb == "home":
            self.plan = [(env.rest_pose.copy(), OPEN, 0)]
        elif command.verb in ("open", "close"):
            g = OPEN if command.verb == "open" else CLOSED
            self.plan = [(env.q_ref, g, 15)]
        else:
            raise ValueError(f"unknown verb {command.verb!r}")
        if self.plan[0][1] != grip_now and command.verb in ("pick", "place"):
            self.plan.insert(0, (q0, OPEN, 10))  # сначала раскрыть захват на месте
        self.phase, self.steps_in_phase = 0, 0

    # ------------------------------------------------------------------ исполнение
    @property
    def done(self) -> bool:
        return self.phase >= len(self.plan)

    def act(self) -> np.ndarray:
        if self.done:
            q_des, grip, min_steps = self.plan[-1]
        else:
            q_des, grip, min_steps = self.plan[self.phase]
        a = np.zeros(6)
        a[:5] = np.clip((q_des - self.env.q_ref) / self.env.action_scale, -1.0, 1.0)
        a[5] = grip
        if self.done:
            return a
        self.steps_in_phase += 1
        reached = np.abs(q_des - self.env.q_ref).max() < 1e-6 and (
            np.abs(q_des - self.env.q_arm).max() < 0.05
        )
        if (reached or self.steps_in_phase > 60) and self.steps_in_phase >= min_steps:
            self.phase += 1
            self.steps_in_phase = 0
        return a


def run_episode(env: PickEnv, seed: int, render: bool = False) -> tuple[bool, list]:
    ctrl = ScriptedPick(env)
    env.reset(seed=seed)
    ctrl.reset()
    frames, success = [], False
    while True:
        if render:
            frames.append(env.render_rgb(360, 480))
        _, _, terminated, truncated, info = env.step(ctrl.act())
        success = success or info["success"]
        if terminated or truncated:
            return success, frames


def execute(env: PickEnv, command: Command, max_steps: int = 400, sleep: float = 0.0) -> dict:
    """Выполнить команду в текущем состоянии среды (без reset); вернуть info последнего шага."""
    import time

    ctrl = ScriptedPick(env)
    ctrl.reset(command)
    info: dict = {}
    for _ in range(max_steps):
        _, _, terminated, truncated, info = env.step(ctrl.act())
        if sleep:
            time.sleep(sleep)
        if ctrl.done or terminated or truncated:
            break
    return info
