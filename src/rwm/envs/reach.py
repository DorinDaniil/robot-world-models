"""Задача Reach: подвести конец 3-DoF руки (site "ee") к случайной цели.

Формально (см. docs/lectures/L1): эпизодический MDP с горизонтом T = max_steps.

Наблюдение, obs_mode="state", размерность 15:
    [q (3), q̇ (3), p_ee (3), p_target (3), p_target − p_ee (3)]
Наблюдение, obs_mode="pixels": RGB uint8 (H, W, 3) с камеры "front".

Действие a ∈ [−1, 1]^3: приращение целевых углов, q_ref ← clip(q_ref + a·action_scale).

Награда r = −‖p_target − p_ee‖ + bonus·1[‖p_target − p_ee‖ < success_radius].
terminated всегда False; truncated при t ≥ max_steps.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from rwm import ASSETS
from rwm.envs.base import MujocoEnv


class ReachEnv(MujocoEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        obs_mode: str = "state",
        max_steps: int = 100,
        frame_skip: int = 10,
        action_scale: float = 0.15,
        success_radius: float = 0.05,
        success_bonus: float = 1.0,
        image_size: int = 64,
        model_path: str | None = None,
    ):
        if obs_mode not in ("state", "pixels"):
            raise ValueError(f"obs_mode must be 'state' or 'pixels', got {obs_mode!r}")
        super().__init__(
            model_path=model_path or str(ASSETS / "arm3.xml"),
            frame_skip=frame_skip,
            max_steps=max_steps,
            render_mode=render_mode,
            camera="front",
        )
        self.obs_mode = obs_mode
        self.action_scale = action_scale
        self.success_radius = success_radius
        self.success_bonus = success_bonus
        self.image_size = image_size

        self.q_low, self.q_high = self.joint_limits
        target_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target_mocap = self.model.body_mocapid[target_body]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.model.nu,), dtype=np.float32)
        if obs_mode == "state":
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(15,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(
                0, 255, shape=(image_size, image_size, 3), dtype=np.uint8
            )

    # -------------------------------------------------------------- state
    @property
    def ee_pos(self) -> np.ndarray:
        return self.site_pos("ee")

    @property
    def target_pos(self) -> np.ndarray:
        return self.data.mocap_pos[self._target_mocap].copy()

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.target_pos - self.ee_pos))

    # -------------------------------------------------------------- task hooks
    def _reset_task(self) -> None:
        self.data.qpos[:] = self.np_random.uniform(-0.3, 0.3, size=self.model.nq)
        self.data.ctrl[:] = self.data.qpos
        # Цель = FK случайной конфигурации, поэтому она гарантированно достижима.
        q = self.np_random.uniform(self.q_low, self.q_high)
        target = self.forward_kinematics(q, "ee")
        target[2] = max(target[2], 0.08)
        self.data.mocap_pos[self._target_mocap] = target

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.clip(action, -1.0, 1.0)
        q_ref = self.data.ctrl + action * self.action_scale
        self.data.ctrl[:] = np.clip(q_ref, self.q_low, self.q_high)

    def _task_step(self) -> tuple[float, bool, dict[str, Any]]:
        d = self.distance
        success = d < self.success_radius
        reward = -d + (self.success_bonus if success else 0.0)
        return reward, False, {"dist": d, "success": success}

    def _get_obs(self) -> np.ndarray:
        if self.obs_mode == "pixels":
            return self.render_rgb(self.image_size, self.image_size)
        ee, target = self.ee_pos, self.target_pos
        return np.concatenate([self.data.qpos, self.data.qvel, ee, target, target - ee]).astype(
            np.float32
        )
