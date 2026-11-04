"""Базовый класс для сред на MuJoCo.

Содержит только то, что одинаково для всех задач: загрузка модели, интегрирование
физики, рендеринг, освобождение ресурсов. Логика задачи (наблюдение, награда,
семплирование цели) определяется в подклассах через три метода:

    _get_obs()      -> наблюдение
    _reset_task()   -> расставить цели/объекты после сброса
    _task_step()    -> (reward, terminated, info) после шага физики
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

import gymnasium as gym
import mujoco
import numpy as np


class MujocoEnv(gym.Env, ABC):
    metadata: ClassVar[dict] = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str,
        frame_skip: int,
        max_steps: int,
        render_mode: str | None = None,
        camera: str | None = None,
        render_size: tuple[int, int] = (240, 320),
    ):
        super().__init__()
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"unknown render_mode {render_mode!r}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.render_size = render_size
        self._camera_id = (
            -1
            if camera is None
            else mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        )
        self._renderer: mujoco.Renderer | None = None
        self._viewer = None
        self._t = 0

    # -------------------------------------------------------------- task hooks
    @abstractmethod
    def _get_obs(self) -> np.ndarray: ...

    @abstractmethod
    def _reset_task(self) -> None: ...

    @abstractmethod
    def _task_step(self) -> tuple[float, bool, dict[str, Any]]: ...

    # -------------------------------------------------------------- properties
    @property
    def dt(self) -> float:
        """Шаг управления (с): шаг физики × frame_skip."""
        return self.model.opt.timestep * self.frame_skip

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        r = self.model.jnt_range
        return r[:, 0].copy(), r[:, 1].copy()

    def site_pos(self, name: str) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.data.site_xpos[sid].copy()

    def forward_kinematics(self, q: np.ndarray, site: str) -> np.ndarray:
        """Позиция site при конфигурации q. Состояние симулятора восстанавливается."""
        saved = self.data.qpos.copy()
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        pos = self.site_pos(site)
        self.data.qpos[:] = saved
        mujoco.mj_kinematics(self.model, self.data)
        return pos

    # -------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._t = 0
        self._reset_task()
        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self._sync_viewer()
        return self._get_obs(), self._task_step()[2]

    def step(self, action: np.ndarray):
        self._apply_action(np.asarray(action, dtype=np.float64))
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self._t += 1
        reward, terminated, info = self._task_step()
        truncated = self._t >= self.max_steps
        if self.render_mode == "human":
            self._sync_viewer()
        return self._get_obs(), reward, terminated, truncated, info

    @abstractmethod
    def _apply_action(self, action: np.ndarray) -> None:
        """Записать действие агента в data.ctrl."""

    # -------------------------------------------------------------- rendering
    def render_rgb(self, height: int | None = None, width: int | None = None) -> np.ndarray:
        h, w = height or self.render_size[0], width or self.render_size[1]
        if self._renderer is None or (self._renderer.height, self._renderer.width) != (h, w):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._renderer.update_scene(self.data, camera=self._camera_id)
        return self._renderer.render().copy()

    def _sync_viewer(self) -> None:
        import mujoco.viewer

        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self._viewer.is_running():
            self._viewer.sync()

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_rgb()
        if self.render_mode == "human":
            self._sync_viewer()
        return None

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
