"""TD-MPC2: формы, конечность лоссов, сериализация, планировщик. Только CPU, малые размеры."""

import numpy as np
import pytest
import torch

from rwm.algos.tdmpc2 import TDMPC2
from rwm.data.replay import ReplayBuffer

OBS, ACT = 7, 3
TINY = {
    "latent_dim": 16,
    "hidden": 32,
    "num_bins": 21,
    "num_samples": 32,
    "num_pi_trajs": 4,
    "iterations": 2,
    "num_elites": 8,
    "batch_size": 16,
}


def _agent(**kw) -> TDMPC2:
    torch.manual_seed(0)
    return TDMPC2((OBS,), ACT, **{**TINY, **kw})


def _buffer(steps: int = 200, seed: int = 0) -> ReplayBuffer:
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer((OBS,), ACT, capacity=1000, seed=seed)
    t = 0
    while t < steps:
        for i in range(20):
            buf.add(rng.normal(size=OBS), rng.uniform(-1, 1, ACT), rng.normal(), i == 19)
            t += 1
        buf.end_episode(rng.normal(size=OBS))
    return buf


@pytest.mark.parametrize("mpc", [True, False])
def test_act_shape_and_range(mpc):
    agent = _agent(mpc=mpc)
    agent.reset()
    obs = np.random.default_rng(0).normal(size=OBS).astype(np.float32)
    for det in (False, True):
        a = agent.act(obs, deterministic=det)
        assert a.shape == (ACT,) and np.all(np.abs(a) <= 1.0) and np.all(np.isfinite(a))
    if mpc:
        assert agent._prev_mean is not None and agent._prev_mean.shape == (agent.horizon, ACT)
        agent.reset()
        assert agent._prev_mean is None


def test_update_metrics_finite_and_losses_move():
    agent = _agent()
    buf = _buffer()
    first = agent.update(buf)
    assert all(np.isfinite(v) for v in first.values())
    for _ in range(30):
        last = agent.update(buf)
    assert all(np.isfinite(v) for v in last.values())
    assert last["loss_consistency"] < first["loss_consistency"]


def test_state_dict_round_trip():
    a1, a2 = _agent(), _agent(mpc=False)
    a1.update(_buffer(100))
    a2.load_state_dict(a1.state_dict())
    z = torch.randn(4, TINY["latent_dim"]).softmax(-1)
    a = torch.zeros(4, ACT)
    assert torch.allclose(a1.q_value(z, a), a2.q_value(z, a))
    assert torch.allclose(a1.q_value(z, a, target=True), a2.q_value(z, a, target=True))


def test_pixels_encoder():
    torch.manual_seed(0)
    agent = TDMPC2((64, 64, 3), ACT, cnn_depth=4, **TINY)
    obs = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert agent.act(obs).shape == (ACT,)


def test_planner_with_constant_value_is_finite():
    """Если награда и Q не зависят от действия, MPPI сводится к усреднению кандидатов."""
    agent = _agent()
    agent.twohot.decode = lambda logits: torch.zeros(logits.shape[:-1])
    z = agent.model.encode(torch.zeros(1, OBS))
    a = agent.plan(z, deterministic=True)
    assert a.shape == (ACT,) and torch.isfinite(a).all() and a.abs().max() < 0.5
