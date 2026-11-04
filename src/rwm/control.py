"""Аналитический контроллер захвата для PickEnv на основе обратной кинематики.

Используется как (а) проверка согласованности модели, актуаторов и физики контакта,
(б) верхняя оценка доли успехов для обучаемых методов, (в) источник демонстраций.

Фазы: подход сверху → опускание → смыкание пальцев → подъём. На каждом шаге действие
агента вычисляется как ограниченный шаг q_ref к решению IK, то есть контроллер работает
через тот же интерфейс действий, что и обучаемая политика.
"""

from __future__ import annotations

import numpy as np

from rwm.envs.pick import PickEnv
from rwm.kinematics import SerialChain, arm5_chain

DOWN = np.array([0.0, 0.0, -1.0])


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
    def __init__(self, env: PickEnv, hover: float = 0.10, grasp_z: float = 0.02):
        self.env = env
        self.chain = arm5_chain()
        self.hover = hover
        self.grasp_z = grasp_z
        self.limits = (env.q_low, env.q_high)
        self.plan: list[tuple[np.ndarray, float, int]] = []  # (q_des, gripper, min_steps)
        self.phase = 0
        self.steps_in_phase = 0

    def reset(self) -> None:
        env = self.env
        cube = env.cube_pos
        q0 = env.q_arm
        above, _ = self.chain.ik(cube + [0, 0, self.hover], q0, DOWN, limits=self.limits)
        above[4] = grasp_yaw(self.chain, above, env.cube_yaw)
        down, _ = self.chain.ik(
            np.array([cube[0], cube[1], self.grasp_z]), above, DOWN, limits=self.limits
        )
        down[4] = above[4]
        up = above.copy()
        self.plan = [(above, -1.0, 0), (down, -1.0, 0), (down, 1.0, 15), (up, 1.0, 10**6)]
        self.phase, self.steps_in_phase = 0, 0

    def act(self) -> np.ndarray:
        q_des, grip, min_steps = self.plan[self.phase]
        a = np.zeros(6)
        a[:5] = np.clip((q_des - self.env.q_ref) / self.env.action_scale, -1.0, 1.0)
        a[5] = grip
        self.steps_in_phase += 1
        reached = np.abs(q_des - self.env.q_ref).max() < 1e-6 and (
            np.abs(q_des - self.env.q_arm).max() < 0.05
        )
        done = (reached or self.steps_in_phase > 60) and self.steps_in_phase >= min_steps
        if done and self.phase < len(self.plan) - 1:
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
