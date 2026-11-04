from gymnasium.envs.registration import register

from rwm.envs.pick import PickEnv
from rwm.envs.reach import ReachEnv

register(id="rwm/Reach-v0", entry_point="rwm.envs.reach:ReachEnv", max_episode_steps=100)
register(id="rwm/Pick-v0", entry_point="rwm.envs.pick:PickEnv", max_episode_steps=150)

__all__ = ["PickEnv", "ReachEnv"]
