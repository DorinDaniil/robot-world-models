from gymnasium.envs.registration import register

from rwm.envs.reach import ReachEnv

register(
    id="rwm/Reach-v0",
    entry_point="rwm.envs.reach:ReachEnv",
    max_episode_steps=100,
)

__all__ = ["ReachEnv"]
