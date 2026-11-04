import numpy as np
import pytest

from rwm.control import execute
from rwm.envs import PickEnv
from rwm.language import ParseError, parse


@pytest.mark.parametrize(
    "text, verb",
    [
        ("подними кубик", "pick"),
        ("Возьми куб!", "pick"),
        ("pick up the cube", "pick"),
        ("открой захват", "open"),
        ("close the gripper", "close"),
        ("домой", "home"),
    ],
)
def test_verbs(text, verb):
    assert parse(text).verb == verb


def test_pick_height_units():
    assert parse("подними кубик на 10 см").height == pytest.approx(0.10)
    assert parse("lift the cube 0.08").height == pytest.approx(0.08)
    assert parse("подними кубик").height is None


def test_place_relative_and_absolute():
    c = parse("положи кубик слева")
    assert c.verb == "place" and c.offset == (0.0, 0.10) and c.target is None
    assert c.resolve_target((0.3, 0.0)) == pytest.approx((0.3, 0.10))
    c = parse("put the cube 5 cm closer")
    assert c.offset == pytest.approx((-0.05, 0.0))
    c = parse("перенеси кубик в точку 0.28 -0.12")
    assert c.target == pytest.approx((0.28, -0.12))
    with pytest.raises(ParseError):
        parse("положи кубик")
    with pytest.raises(ParseError):
        parse("сделай что-нибудь")


def test_place_moves_cube_to_target():
    env = PickEnv(max_steps=10**6)
    env.reset(seed=1)
    cmd = parse("положи кубик справа на 8 см")
    target = np.array(cmd.resolve_target(env.cube_pos[:2]))
    execute(env, cmd)
    assert np.linalg.norm(env.cube_pos[:2] - target) < 0.02
    assert env.cube_pos[2] < 0.03  # кубик стоит на полу
    env.close()
