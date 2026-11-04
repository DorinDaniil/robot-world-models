import gymnasium as gym
import numpy as np

import rwm.envs  # noqa: F401  (регистрирует rwm/Reach-v0)
from rwm.envs import ReachEnv


def test_spaces_and_reset():
    env = ReachEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape == (15,)
    assert env.observation_space.contains(obs)
    assert set(info) == {"dist", "success"}
    env.close()


def test_step_contract():
    env = ReachEnv(max_steps=20)
    env.reset(seed=1)
    for _ in range(20):
        obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
        assert obs.shape == (15,)
        assert np.isfinite(r)
        assert terminated is False
    assert truncated is True
    env.close()


def test_target_is_reachable_and_above_floor():
    env = ReachEnv()
    reach = 0.25 + 0.30 + 0.25 + 0.10  # длины звеньев + высота основания
    for s in range(20):
        env.reset(seed=s)
        t = env.target_pos
        assert t[2] >= 0.08
        assert np.linalg.norm(t) <= reach + 1e-6
    env.close()


def test_determinism():
    e1, e2 = ReachEnv(), ReachEnv()
    o1, _ = e1.reset(seed=42)
    o2, _ = e2.reset(seed=42)
    assert np.allclose(o1, o2)
    a = np.array([0.5, -0.3, 0.2])
    for _ in range(10):
        o1, o2 = e1.step(a)[0], e2.step(a)[0]
    assert np.allclose(o1, o2)
    e1.close()
    e2.close()


def test_greedy_controller_reaches_target():
    """Градиентный спуск по расстоянию в пространстве целевых углов.

    Проверяет согласованность актуаторов, кинематики и награды: если всё верно,
    простейший контроллер должен достигать цели.
    """
    env = ReachEnv(max_steps=80, action_scale=0.1)
    env.reset(seed=3)
    d0 = env.distance
    eps = 1e-3

    def dist_at(q: np.ndarray) -> float:
        return float(np.linalg.norm(env.target_pos - env.forward_kinematics(q, "ee")))

    for _ in range(env.max_steps):
        q_ref = env.data.ctrl.copy()
        d = dist_at(q_ref)
        grad = np.array([(dist_at(q_ref + eps * e) - d) / eps for e in np.eye(env.model.nu)])
        _, _, _, _, info = env.step(-np.clip(5.0 * grad, -1, 1))
        if info["success"]:
            break
    assert info["dist"] < d0
    assert info["success"], f"controller did not reach target, dist={info['dist']:.3f}"
    env.close()


def test_pixels_mode():
    env = ReachEnv(obs_mode="pixels", image_size=32)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (32, 32, 3) and obs.dtype == np.uint8
    assert obs.std() > 1.0  # не пустой кадр
    env.close()


def test_registered_id():
    env = gym.make("rwm/Reach-v0")
    obs, _ = env.reset(seed=0)
    assert obs.shape == (15,)
    env.close()
