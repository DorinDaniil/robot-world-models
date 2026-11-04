import gymnasium as gym
import numpy as np

import rwm.envs  # noqa: F401
from rwm.control import run_episode
from rwm.envs import PickEnv


def test_spaces_and_contract():
    env = PickEnv(max_steps=10)
    obs, info = env.reset(seed=0)
    assert obs.shape == (23,) and env.observation_space.contains(obs)
    assert set(info) == {"dist", "grasped", "lift", "success"}
    for _ in range(10):
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (23,) and np.isfinite(r) and terminated is False
    assert truncated is True
    env.close()


def test_cube_starts_on_floor_in_workspace():
    env = PickEnv()
    for s in range(10):
        env.reset(seed=s)
        c = env.cube_pos
        assert abs(c[2] - 0.015) < 1e-6
        assert 0.22 <= np.linalg.norm(c[:2]) <= 0.34
        assert env.lift == 0.0 and not env.grasped()
    env.close()


def test_determinism():
    e1, e2 = PickEnv(), PickEnv()
    o1, _ = e1.reset(seed=7)
    o2, _ = e2.reset(seed=7)
    assert np.allclose(o1, o2)
    a = np.array([0.3, -0.2, 0.1, 0.0, 0.2, 1.0])
    for _ in range(10):
        o1, o2 = e1.step(a)[0], e2.step(a)[0]
    assert np.allclose(o1, o2)
    e1.close()
    e2.close()


def test_scripted_controller_lifts_cube():
    """Аналитический контроллер должен поднимать кубик: проверка модели и физики контакта."""
    env = PickEnv()
    results = [run_episode(env, seed=s)[0] for s in range(5)]
    assert all(results), f"successes: {results}"
    env.close()


def test_pixels_and_registry():
    env = PickEnv(obs_mode="pixels", image_size=32)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (32, 32, 3) and obs.dtype == np.uint8
    env.close()
    env = gym.make("rwm/Pick-v0")
    assert env.reset(seed=0)[0].shape == (23,)
    env.close()
