"""Dreamer: формы, конечность лоссов, λ-доходность, открытый цикл, сериализация."""

import numpy as np
import torch

from rwm.algos.dreamer import Dreamer, lambda_return
from rwm.data.replay import ReplayBuffer

TINY = {
    "deter": 32,
    "hidden": 32,
    "stoch": 4,
    "classes": 4,
    "depth": 8,
    "batch_size": 4,
    "seq_len": 8,
    "horizon": 5,
}


def make(obs_shape, act_dim=3, **kw):
    dtype = np.uint8 if len(obs_shape) == 3 else np.float32
    agent = Dreamer(obs_shape, act_dim, update_every=1, **TINY, **kw)
    buf = ReplayBuffer(obs_shape, act_dim, capacity=400, obs_dtype=dtype, seed=0)
    rng = np.random.default_rng(0)

    def obs():
        return (
            rng.integers(0, 255, obs_shape, dtype=np.uint8)
            if dtype == np.uint8
            else rng.normal(size=obs_shape).astype(np.float32)
        )

    for _ in range(6):
        for t in range(40):
            buf.add(obs(), rng.uniform(-1, 1, act_dim), rng.normal(), t == 39)
        buf.end_episode(obs())
    return agent, buf, obs


def test_act_and_update():
    agent, buf, obs = make((15,))
    a = agent.act(obs())
    assert a.shape == (3,) and np.all(np.abs(a) <= 1)
    assert not torch.all(agent._h == 0)
    agent.reset()
    assert torch.all(agent._h == 0) and torch.all(agent._a == 0)
    m = agent.update(buf)
    assert m and all(np.isfinite(v) for v in m.values())
    a_det = agent.act(obs(), deterministic=True)
    assert a_det.shape == (3,)


def test_pixels_update():
    agent, buf, obs = make((64, 64, 3))
    assert agent.act(obs()).shape == (3,)
    m = agent.update(buf)
    assert all(np.isfinite(v) for v in m.values())
    err = agent.open_loop_error(buf, context=3, horizon=4, batch_size=2)
    assert len(err["reward_abs_error"]) == 4 and len(err["recon_error"]) == 4


def test_lambda_return_limits():
    r = torch.tensor([[1.0], [2.0], [3.0]])  # r_1, r_2, r_3
    v = torch.tensor([[10.0], [20.0], [30.0], [40.0]])  # v_0..v_3
    c = torch.ones(3, 1)
    g = 0.9
    mc = lambda_return(r, v, c, g, 1.0)  # Монте-Карло с бутстрапом в конце
    expected0 = 1 + g * (2 + g * (3 + g * 40))
    assert torch.isclose(mc[0, 0], torch.tensor(expected0))
    td = lambda_return(r, v, c, g, 0.0)  # одношаговый TD
    assert torch.allclose(td, r + g * v[1:])
    c2 = torch.tensor([[1.0], [0.0], [1.0]])  # обрыв после второго шага
    mc2 = lambda_return(r, v, c2, g, 1.0)
    assert torch.isclose(mc2[0, 0], torch.tensor(1 + g * 2.0))


def test_open_loop_and_state_dict():
    agent, buf, obs = make((15,))
    agent.update(buf)
    err = agent.open_loop_error(buf, context=5, horizon=6, batch_size=4)
    assert len(err["reward_abs_error"]) == 6 and all(np.isfinite(err["reward_abs_error"]))
    sd = agent.state_dict()
    other = Dreamer((15,), 3, **TINY)
    other.load_state_dict(sd)
    o = obs()
    torch.manual_seed(0)  # posterior семплируется, поэтому сид фиксируется перед каждым вызовом
    a1 = agent.act(o, deterministic=True)
    torch.manual_seed(0)
    a2 = other.act(o, deterministic=True)
    assert np.allclose(a1, a2, atol=1e-6)
