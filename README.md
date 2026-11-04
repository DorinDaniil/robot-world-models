# robot-world-models

Learning to control a robot arm with world models, implemented from scratch in MuJoCo and
PyTorch. Course notes (in Russian), a 5-DoF arm with a parallel gripper described by screw
theory, reach and pick environments, and reference implementations of SAC, TD-MPC2 and an
RSSM/Dreamer-style agent.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,train]"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pytest
```

## Usage

```bash
python scripts/00_view_arm.py                        # interactive MuJoCo viewer, 3-DoF arm
python scripts/01_random_rollout.py --gif runs/random.gif
python scripts/02_scripted_pick.py --view            # 5-DoF arm picks up a cube (IK controller)
```

```python
import gymnasium as gym
import rwm.envs

env = gym.make("rwm/Pick-v0")           # 5-DoF arm + gripper, lift a cube
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

```python
from rwm.kinematics import arm5_chain

chain = arm5_chain()                    # screw axes S_i and home pose M
T = chain.fk(theta)                     # product of exponentials
J = chain.jacobian_space(theta)
theta, ok = chain.ik(target_pos, theta0, target_axis=[0, 0, -1])
```

## Layout

```
assets/      MJCF models: arm3 (3-DoF), arm5 (5-DoF + gripper)
configs/     experiment configs (YAML)
docs/        syllabus, lectures L1–L5, labs, arm geometry, reading list
scripts/     entry points, numbered by lab
src/rwm/     envs/  kinematics.py  control.py  models/  algos/  utils/
tests/
```

## Docs

| | |
|---|---|
| [Syllabus](docs/00_syllabus.md) | course outline, prerequisites, reporting requirements |
| [Lectures](docs/lectures/) | L1 MDP & SAC · L2 dynamics & simulation · L3 latent world models · L4 latent planning / TD-MPC2 · L5 imitation & sim2real |
| [Labs](docs/03_labs.md) | six graded labs, from the environment to a physical SO-101 arm |
| [Arm geometry](docs/06_arm_geometry.md) | screw axes, closed-form FK, Jacobian singularities, IK, gripper and dynamics of `arm5` |
| [Architecture](docs/01_system_architecture.md) · [Stack](docs/02_stack.md) · [Reading](docs/04_reading_list.md) · [Code conventions](docs/05_code_conventions.md) | |

## Status

- [x] Lab 0 — `ReachEnv` (state / pixels), viewer, tests
- [x] Lab 1 — `arm5` model, POE kinematics verified against MuJoCo, `PickEnv`, scripted pick (20/20)
- [ ] Lab 2 — SAC
- [ ] Lab 3 — TD-MPC2
- [ ] Lab 4 — RSSM / Dreamer
- [ ] Lab 5 — SO-101 model, domain randomization
- [ ] Lab 6 — physical arm, imitation learning

## License

MIT
