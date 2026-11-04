# robot-world-models

Learning to control a robot arm with world models, implemented from scratch in MuJoCo and
PyTorch. Course notes (in Russian), a minimal arm environment, and reference implementations
of SAC, TD-MPC2 and an RSSM/Dreamer-style agent.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,train]"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pytest
```

## Usage

```bash
python scripts/00_view_arm.py                        # interactive MuJoCo viewer
python scripts/01_random_rollout.py --gif runs/random.gif
```

```python
import gymnasium as gym
import rwm.envs

env = gym.make("rwm/Reach-v0")          # 3-DoF arm, reach a random target
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Layout

```
assets/      MJCF models
configs/     experiment configs (YAML)
docs/        syllabus, lectures L1–L5, labs, reading list
scripts/     entry points, numbered by lab
src/rwm/     envs/  models/  algos/  utils/
tests/
```

## Docs

| | |
|---|---|
| [Syllabus](docs/00_syllabus.md) | course outline, prerequisites, reporting requirements |
| [Lectures](docs/lectures/) | L1 MDP & SAC · L2 dynamics & simulation · L3 latent world models · L4 latent planning / TD-MPC2 · L5 imitation & sim2real |
| [Labs](docs/03_labs.md) | six graded labs, from the environment to a physical SO-101 arm |
| [Architecture](docs/01_system_architecture.md) · [Stack](docs/02_stack.md) · [Reading](docs/04_reading_list.md) · [Code conventions](docs/05_code_conventions.md) | |

## Status

- [x] Lab 0 — `ReachEnv` (state / pixels), viewer, tests
- [ ] Lab 1 — `PushEnv`
- [ ] Lab 2 — SAC
- [ ] Lab 3 — TD-MPC2
- [ ] Lab 4 — RSSM / Dreamer
- [ ] Lab 5 — SO-101 model, domain randomization
- [ ] Lab 6 — physical arm, imitation learning

## License

MIT
