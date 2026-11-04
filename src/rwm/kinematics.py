"""Кинематика открытой цепи в формулировке произведения экспонент (POE).

Обозначения (Lynch & Park, *Modern Robotics*, гл. 3–6). Твист $\\mathcal V = (\\omega, v) \\in
\\mathbb R^6$; матричное представление $[\\mathcal V] \\in se(3)$. Винтовая ось шарнира $i$ в
пространственной системе координат при нулевой конфигурации: $S_i = (\\omega_i, v_i)$,
$v_i = -\\omega_i \\times q_i$, где $q_i$ — точка на оси. Прямая кинематика:

    T(θ) = e^{[S_1]θ_1} e^{[S_2]θ_2} ··· e^{[S_n]θ_n} M,

M — поза рабочего органа при θ = 0. Пространственный якобиан:
J_s(θ) = [S_1, Ad_{T_1} S_2, ..., Ad_{T_1···T_{n-1}} S_n], где T_k = e^{[S_k]θ_k}.

Модуль не зависит от MuJoCo; `screw_axes_from_mujoco` строит (S, M) из модели для
сверки с аналитическими значениями.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- so(3) / se(3)
def skew(w: np.ndarray) -> np.ndarray:
    """[w] ∈ so(3): [w] x = w × x."""
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def exp_so3(w: np.ndarray, theta: float) -> np.ndarray:
    """Формула Родрига: e^{[w]θ} = I + sin θ [w] + (1 − cos θ)[w]², ‖w‖ = 1."""
    W = skew(w)
    return np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * W @ W


def exp_se3(S: np.ndarray, theta: float) -> np.ndarray:
    """e^{[S]θ} для единичного винта S = (ω, v), ‖ω‖ = 1 (или ω = 0 — призматический)."""
    w, v = S[:3], S[3:]
    T = np.eye(4)
    if np.allclose(w, 0):
        T[:3, 3] = v * theta
        return T
    W = skew(w)
    T[:3, :3] = exp_so3(w, theta)
    # G(θ) = Iθ + (1 − cos θ)[w] + (θ − sin θ)[w]²
    G = np.eye(3) * theta + (1 - np.cos(theta)) * W + (theta - np.sin(theta)) * W @ W
    T[:3, 3] = G @ v
    return T


def adjoint(T: np.ndarray) -> np.ndarray:
    """Ad_T ∈ R^{6×6}: [[R, 0], [[p]R, R]] — переносит твисты между системами координат."""
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[3:, :3] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


def log_so3(R: np.ndarray) -> np.ndarray:
    """Вектор поворота ωθ для R ∈ SO(3) (обратная к формуле Родрига)."""
    cos = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    theta = np.arccos(cos)
    if theta < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return w * theta


# --------------------------------------------------------------------------- цепь
@dataclass(frozen=True)
class SerialChain:
    """Открытая цепь: винтовые оси S (n×6) в пространственной системе и домашняя поза M."""

    S: np.ndarray  # (n, 6)
    M: np.ndarray  # (4, 4)

    @property
    def n(self) -> int:
        return self.S.shape[0]

    def fk(self, theta: np.ndarray) -> np.ndarray:
        """T(θ) = Π_i e^{[S_i]θ_i} · M."""
        T = np.eye(4)
        for S_i, th in zip(self.S, theta):
            T = T @ exp_se3(S_i, th)
        return T @ self.M

    def jacobian_space(self, theta: np.ndarray) -> np.ndarray:
        """J_s(θ) ∈ R^{6×n}: столбец i = Ad_{e^{[S_1]θ_1}···e^{[S_{i−1}]θ_{i−1}}} S_i."""
        J = np.zeros((6, self.n))
        T = np.eye(4)
        for i, (S_i, th) in enumerate(zip(self.S, theta)):
            J[:, i] = adjoint(T) @ S_i
            T = T @ exp_se3(S_i, th)
        return J

    def jacobian_point(self, theta: np.ndarray) -> np.ndarray:
        """Якобиан линейной скорости точки рабочего органа: ṗ = J_p(θ) θ̇, J_p ∈ R^{3×n}.

        Из пространственного твиста: ṗ = ω_s × p + v_s = (−[p] J_ω + J_v) θ̇.
        """
        J = self.jacobian_space(theta)
        p = self.fk(theta)[:3, 3]
        return -skew(p) @ J[:3] + J[3:]

    def ik(
        self,
        target_pos: np.ndarray,
        theta0: np.ndarray,
        target_axis: np.ndarray | None = None,
        tool_axis: np.ndarray | None = None,
        limits: tuple[np.ndarray, np.ndarray] | None = None,
        damping: float = 0.05,
        step: float = 0.5,
        iters: int = 200,
        tol: float = 1e-4,
    ) -> tuple[np.ndarray, bool]:
        """Обратная кинематика методом демпфированных наименьших квадратов.

        Требуемая скорость рабочего органа: линейная ṗ = p* − p; при заданном направлении
        оси инструмента z* — угловая ω = z × z* (ось поворота, переводящего z в z*,
        с модулем sin угла). Якобиан: [J_p; J_ω], J_ω — угловая часть J_s.
        Шаг: Δθ = step · Jᵀ (J Jᵀ + λ² I)⁻¹ e. Возвращает (θ, сошлось ли).
        """
        theta = theta0.astype(float).copy()
        tool_axis = np.array([0.0, 0.0, 1.0]) if tool_axis is None else tool_axis
        for _ in range(iters):
            T = self.fk(theta)
            e = target_pos - T[:3, 3]
            J = self.jacobian_point(theta)
            if target_axis is not None:
                z = T[:3, :3] @ tool_axis
                e_rot = np.cross(z, target_axis)
                if np.linalg.norm(e_rot) < 1e-6 and z @ target_axis < 0:
                    # антипараллельные оси: z × z* = 0; берём любое перпендикулярное направление
                    e_rot = np.cross(z, np.array([1.0, 0.0, 0.0]))
                    if np.linalg.norm(e_rot) < 1e-6:
                        e_rot = np.cross(z, np.array([0.0, 1.0, 0.0]))
                e = np.concatenate([e, e_rot])
                J = np.vstack([J, self.jacobian_space(theta)[:3]])
            if np.linalg.norm(e) < tol:
                return theta, True
            dtheta = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), e)
            theta = theta + step * dtheta
            if limits is not None:
                theta = np.clip(theta, limits[0], limits[1])
        return theta, False


# --------------------------------------------------------------------------- из модели MuJoCo
def screw_axes_from_mujoco(model, data, joint_names: list[str], site: str) -> SerialChain:
    """Строит (S, M) по модели MuJoCo в нулевой конфигурации.

    Для вращательного шарнира ω — ось в мировой системе, q — точка оси (anchor);
    v = −ω × q. M — поза site при θ = 0.
    """
    import mujoco

    qpos0 = data.qpos.copy()
    data.qpos[:] = 0
    mujoco.mj_kinematics(model, data)
    S = []
    for name in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        w = data.xaxis[jid].copy()
        q = data.xanchor[jid].copy()
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
            S.append(np.concatenate([np.zeros(3), w]))
        else:
            S.append(np.concatenate([w, -np.cross(w, q)]))
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
    M = np.eye(4)
    M[:3, :3] = data.site_xmat[sid].reshape(3, 3)
    M[:3, 3] = data.site_xpos[sid]
    data.qpos[:] = qpos0
    mujoco.mj_kinematics(model, data)
    return SerialChain(np.array(S), M)


# --------------------------------------------------------------------------- arm5 аналитически
def arm5_chain() -> SerialChain:
    """Винтовые оси и домашняя поза arm5 (см. assets/arm5.xml, docs/lectures/L3_kinematics_screw_theory.md).

    Высоты осей: h1 = 0.06, h2 = 0.12, h3 = 0.32, h4 = 0.50, h5 = 0.56; ee при θ = 0 — 0.635.
    """
    z, y = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])

    def rev(w, h):
        q = np.array([0.0, 0.0, h])
        return np.concatenate([w, -np.cross(w, q)])

    S = np.array([rev(z, 0.06), rev(y, 0.12), rev(y, 0.32), rev(y, 0.50), rev(z, 0.56)])
    M = np.eye(4)
    M[2, 3] = 0.635
    return SerialChain(S, M)
