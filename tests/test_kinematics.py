"""Аналитическая кинематика arm5 против MuJoCo."""

import mujoco
import numpy as np
import pytest

from rwm import ASSETS
from rwm.kinematics import arm5_chain, exp_se3, exp_so3, log_so3, screw_axes_from_mujoco

JOINTS = ["j1", "j2", "j3", "j4", "j5"]


@pytest.fixture(scope="module")
def model():
    m = mujoco.MjModel.from_xml_path(str(ASSETS / "arm5.xml"))
    return m, mujoco.MjData(m)


def test_rodrigues_is_rotation():
    rng = np.random.default_rng(0)
    for _ in range(10):
        w = rng.normal(size=3)
        w /= np.linalg.norm(w)
        R = exp_so3(w, rng.uniform(-3, 3))
        assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)


def test_log_inverts_exp():
    rng = np.random.default_rng(1)
    for _ in range(10):
        w = rng.normal(size=3)
        w /= np.linalg.norm(w)
        theta = rng.uniform(0.1, 3.0)
        assert np.allclose(log_so3(exp_so3(w, theta)), w * theta)


def test_exp_se3_pure_rotation_keeps_axis_point():
    q = np.array([0.1, 0.2, 0.3])
    S = np.concatenate([[0, 0, 1], -np.cross([0, 0, 1.0], q)])
    T = exp_se3(S, 1.3)
    assert np.allclose(T @ np.append(q, 1.0), np.append(q, 1.0))


def test_closed_form_matches_poe():
    """Замкнутая форма из L3 §3.3: p = (ρ cos θ1, ρ sin θ1, z),
    ρ = l2 sin φ2 + l3 sin φ3 + l45 sin φ4, z = h2 + l2 cos φ2 + l3 cos φ3 + l45 cos φ4,
    φk = θ2 + … + θk; R = Rz(θ1) Ry(φ4) Rz(θ5)."""
    chain = arm5_chain()
    h2, l2, l3, l45 = 0.12, 0.20, 0.18, 0.06 + 0.075
    rng = np.random.default_rng(4)
    for _ in range(20):
        t = rng.uniform(-1.5, 1.5, 5)
        phi2, phi3, phi4 = t[1], t[1] + t[2], t[1] + t[2] + t[3]
        rho = l2 * np.sin(phi2) + l3 * np.sin(phi3) + l45 * np.sin(phi4)
        z = h2 + l2 * np.cos(phi2) + l3 * np.cos(phi3) + l45 * np.cos(phi4)
        p = np.array([rho * np.cos(t[0]), rho * np.sin(t[0]), z])
        R = exp_so3([0, 0, 1.0], t[0]) @ exp_so3([0, 1.0, 0], phi4) @ exp_so3([0, 0, 1.0], t[4])
        T = chain.fk(t)
        assert np.allclose(T[:3, 3], p, atol=1e-12)
        assert np.allclose(T[:3, :3], R, atol=1e-12)


def test_screw_axes_match_model(model):
    m, d = model
    analytic = arm5_chain()
    from_model = screw_axes_from_mujoco(m, d, JOINTS, "ee")
    assert np.allclose(analytic.S, from_model.S, atol=1e-12)
    assert np.allclose(analytic.M, from_model.M, atol=1e-12)


def test_fk_and_jacobian_match_mujoco(model):
    m, d = model
    chain = arm5_chain()
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "ee")
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
    qadr, dofs = m.jnt_qposadr[jids], m.jnt_dofadr[jids]
    rng = np.random.default_rng(2)
    for _ in range(30):
        theta = rng.uniform(m.jnt_range[jids, 0], m.jnt_range[jids, 1])
        d.qpos[:] = 0
        d.qpos[qadr] = theta
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        T = chain.fk(theta)
        assert np.allclose(T[:3, 3], d.site_xpos[sid], atol=1e-10)
        assert np.allclose(T[:3, :3], d.site_xmat[sid].reshape(3, 3), atol=1e-10)
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, None, sid)
        assert np.allclose(chain.jacobian_point(theta), jacp[:, dofs], atol=1e-10)


def test_ik_point_down(model):
    m, _ = model
    chain = arm5_chain()
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINTS]
    limits = (m.jnt_range[jids, 0], m.jnt_range[jids, 1])
    rng = np.random.default_rng(3)
    for _ in range(10):
        r, phi = rng.uniform(0.22, 0.34), rng.uniform(-0.6, 0.6)
        target = np.array([r * np.cos(phi), r * np.sin(phi), rng.uniform(0.02, 0.12)])
        theta, ok = chain.ik(
            target, np.array([0, 0.5, 1.2, 1.4, 0]), np.array([0, 0, -1.0]), limits=limits
        )
        T = chain.fk(theta)
        assert ok
        assert np.linalg.norm(T[:3, 3] - target) < 1e-3
        assert T[2, 2] < -0.999  # ось захвата направлена вниз
