"""Задача Pick: рука arm5 должна схватить кубик и поднять его на заданную высоту.

Наблюдение, obs_mode="state", размерность 23:
    q_arm (5), g (1, раскрытие захвата), q̇_arm (5), p_ee (3), p_cube (3),
    p_cube − p_ee (3), (cos ψ, sin ψ) (2, рыскание кубика), h (1, высота подъёма)
Наблюдение, obs_mode="pixels": RGB uint8 (H, W, 3) с камеры "front".

Действие a ∈ [−1, 1]^6: a[:5] — приращения целевых углов, q_ref ← clip(q_ref + a·scale);
a[5] — абсолютная команда захвата, g_ref = g_max·(1 + a[5])/2 (−1 открыт, +1 закрыт).

Награда: r = −‖p_cube − p_ee‖ + 0.5·1[захват] + 10·min(h, h_max),
где h = z_cube − z_cube,0, захват — контакт обоих пальцев с кубиком.
Успех: h > lift_height. terminated всегда False; truncated при t ≥ max_steps.
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces

from rwm import ASSETS
from rwm.envs.base import MujocoEnv

ARM_JOINTS = ["j1", "j2", "j3", "j4", "j5"]


class PickEnv(MujocoEnv):
    def __init__(
        self,
        render_mode: str | None = None,
        obs_mode: str = "state",
        max_steps: int = 150,
        frame_skip: int = 10,
        action_scale: float = 0.1,
        lift_height: float = 0.05,
        image_size: int = 64,
        model_path: str | None = None,
    ):
        if obs_mode not in ("state", "pixels"):
            raise ValueError(f"obs_mode must be 'state' or 'pixels', got {obs_mode!r}")
        super().__init__(
            model_path=model_path or str(ASSETS / "arm5.xml"),
            frame_skip=frame_skip,
            max_steps=max_steps,
            render_mode=render_mode,
            camera="front",
        )
        self.obs_mode = obs_mode
        self.action_scale = action_scale
        self.lift_height = lift_height
        self.image_size = image_size

        m = self.model
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self._q_arm = np.array([m.jnt_qposadr[j] for j in jid])
        self._v_arm = np.array([m.jnt_dofadr[j] for j in jid])
        self.q_low, self.q_high = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "finger_left")
        self._q_grip = m.jnt_qposadr[gid]
        self.g_max = float(m.jnt_range[gid, 1])
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube")
        self._q_cube = m.jnt_qposadr[cid]
        self._cube_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        self._finger_geoms = {
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in ("finger_left", "finger_right")
        }
        self._target_mocap = m.body_mocapid[
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target")
        ]
        self._z0 = 0.015  # высота центра кубика на полу
        self.rest_pose = np.array([0.0, 0.5, 1.2, 1.4, 0.0])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
        if obs_mode == "state":
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(23,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(
                0, 255, shape=(image_size, image_size, 3), dtype=np.uint8
            )

    # -------------------------------------------------------------- state
    @property
    def q_arm(self) -> np.ndarray:
        return self.data.qpos[self._q_arm].copy()

    @property
    def q_ref(self) -> np.ndarray:
        return self.data.ctrl[:5].copy()

    @property
    def ee_pos(self) -> np.ndarray:
        return self.site_pos("ee")

    @property
    def cube_pos(self) -> np.ndarray:
        return self.data.qpos[self._q_cube : self._q_cube + 3].copy()

    @property
    def cube_yaw(self) -> float:
        w, x, y, z = self.data.qpos[self._q_cube + 3 : self._q_cube + 7]
        return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    @property
    def lift(self) -> float:
        return float(self.cube_pos[2] - self._z0)

    def grasped(self) -> bool:
        """Оба пальца в контакте с кубиком."""
        touching = set()
        for c in self.data.contact[: self.data.ncon]:
            pair = {c.geom1, c.geom2}
            if self._cube_geom in pair:
                touching |= pair & self._finger_geoms
        return touching == self._finger_geoms

    # -------------------------------------------------------------- task hooks
    def _reset_task(self) -> None:
        self.data.qpos[self._q_arm] = self.rest_pose
        self.data.ctrl[:5] = self.rest_pose
        self.data.ctrl[5] = 0.0  # захват открыт
        r = self.np_random.uniform(0.22, 0.34)
        phi = self.np_random.uniform(-0.6, 0.6)
        yaw = self.np_random.uniform(-np.pi, np.pi)
        q = self._q_cube
        self.data.qpos[q : q + 3] = [r * np.cos(phi), r * np.sin(phi), self._z0]
        self.data.qpos[q + 3 : q + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qvel[:] = 0.0
        self.data.mocap_pos[self._target_mocap] = self.data.qpos[q : q + 3] + [
            0,
            0,
            self.lift_height,
        ]

    def _apply_action(self, action: np.ndarray) -> None:
        action = np.clip(action, -1.0, 1.0)
        q_ref = self.data.ctrl[:5] + action[:5] * self.action_scale
        self.data.ctrl[:5] = np.clip(q_ref, self.q_low, self.q_high)
        self.data.ctrl[5] = self.g_max * (1.0 + action[5]) / 2.0

    def _task_step(self) -> tuple[float, bool, dict[str, Any]]:
        d = float(np.linalg.norm(self.cube_pos - self.ee_pos))
        grasped = self.grasped()
        h = self.lift
        reward = -d + 0.5 * grasped + 10.0 * min(max(h, 0.0), 0.1)
        info = {"dist": d, "grasped": grasped, "lift": h, "success": h > self.lift_height}
        return reward, False, info

    def _get_obs(self) -> np.ndarray:
        if self.obs_mode == "pixels":
            return self.render_rgb(self.image_size, self.image_size)
        ee, cube, yaw = self.ee_pos, self.cube_pos, self.cube_yaw
        return np.concatenate(
            [
                self.q_arm,
                [self.data.qpos[self._q_grip]],
                self.data.qvel[self._v_arm],
                ee,
                cube,
                cube - ee,
                [np.cos(yaw), np.sin(yaw)],
                [self.lift],
            ]
        ).astype(np.float32)
