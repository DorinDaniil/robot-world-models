# robot-world-models

Learning to control a robot arm with world models, implemented from scratch in MuJoCo and
PyTorch. A seven-lecture course (in Russian) with its code: a 5-DoF arm with a parallel
gripper described by screw theory, reach and pick environments, SAC, TD-MPC2 and a
DreamerV3-style agent, and a language layer that turns spoken or typed commands into
arm motion.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,train,plot]"
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU; or /whl/cpu
pytest
```

Voice commands additionally need `pip install -e ".[voice]"` (faster-whisper, sounddevice).

## Usage

```bash
python scripts/00_view_arm.py                        # interactive viewer (key 3: joint axes)
python scripts/02_scripted_pick.py --view            # analytic IK controller picks up a cube
python scripts/40_command_arm.py --text "подними кубик" --text "положи кубик слева"
python scripts/40_command_arm.py --voice             # microphone → Whisper → command → arm

python scripts/10_train.py -c configs/sac_reach.yaml seed=0
python scripts/10_train.py -c configs/tdmpc2_pick.yaml
python scripts/10_train.py -c configs/dreamer_reach_pixels.yaml
python scripts/11_eval.py runs/sac_reach/seed0/ckpt.pt --gif runs/sac_reach.gif
python scripts/12_plot.py runs/sac_reach runs/tdmpc2_reach --metric eval/success
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
assets/      MJCF models: arm3 (3-DoF), arm5 (5-DoF + gripper, visual + axis layers)
configs/     one YAML per (algorithm, task)
docs/        course programme and lectures L1–L7
scripts/     00 view · 01 random rollout · 02 scripted pick · 10 train · 11 eval · 12 plot · 40 commands
src/rwm/     envs/  kinematics.py  control.py  algos/{sac,tdmpc2,dreamer}.py  models/nets.py
             data/replay.py  language/{commands,speech}.py  train.py  utils/
tests/
```

## Course

| | |
|---|---|
| [Programme](docs/README.md) | prerequisites, lecture–practicum map, reporting requirements |
| [L1](docs/lectures/L1_mdp_and_sac.md) | MDP, Bellman operators, policy gradient, SAC |
| [L2](docs/lectures/L2_dynamics_and_simulation.md) | rigid-body dynamics, contacts, actuators, MJCF, environment contract |
| [L3](docs/lectures/L3_kinematics_screw_theory.md) | screw axes, product of exponentials, Jacobian singularities, IK, gripper of `arm5` |
| [L4](docs/lectures/L4_world_models_rssm.md) | simulation lemma, sequence ELBO, RSSM, imagination, DreamerV3 |
| [L5](docs/lectures/L5_latent_planning_tdmpc2.md) | MPPI as variational inference, TD-MPC2 |
| [L6](docs/lectures/L6_imitation_and_sim2real.md) | behaviour cloning bounds, ACT / diffusion / flow policies, domain randomization |
| [L7](docs/lectures/L7_vision_language_action.md) | goal-conditioned policies, visual features, language grounding, VLA, speech |

## Status

- [x] `arm5` model, POE kinematics verified against MuJoCo, `Reach` and `Pick` environments
- [x] Analytic pick / pick-and-place controller (20/20), text and voice commands
- [x] SAC, TD-MPC2, Dreamer implemented and unit-tested; training configs for `Reach` and `Pick`
- [ ] Trained checkpoints and learning curves (practicums 1, 4, 5)
- [ ] `PushEnv`, `TableEnv` with task vector, domain randomization, SO-101 model
- [ ] Imitation learning from controller demonstrations; VLA fine-tuning

## License

MIT
