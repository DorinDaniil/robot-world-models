# Литература

Ссылки сгруппированы по модулям курса. Обязательная литература отмечена символом ●.

## Основы обучения с подкреплением (L1)

- ● Sutton R., Barto A. *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press, 2018.
  http://incompleteideas.net/book/the-book-2nd.html — гл. 3, 6, 8, 11, 13.
- ● Achiam J. *Spinning Up in Deep RL*. OpenAI, 2018. https://spinningup.openai.com
- Haarnoja T. et al. Soft Actor-Critic. ICML 2018. https://arxiv.org/abs/1801.01290;
  расширенная версия https://arxiv.org/abs/1812.05905
- Fujimoto S., van Hoof H., Meger D. Addressing Function Approximation Error in Actor-Critic
  Methods. ICML 2018. https://arxiv.org/abs/1802.09477
- Henderson P. et al. Deep Reinforcement Learning That Matters. AAAI 2018.
  https://arxiv.org/abs/1709.06560
- Эталонные реализации: CleanRL, https://github.com/vwxyzjn/cleanrl

## Механика и симуляция (L2)

- ● MuJoCo documentation. https://mujoco.readthedocs.io — Overview, Modeling, Computation,
  XML Reference.
- Todorov E. Convex and analytically-invertible dynamics with contacts and constraints.
  ICRA 2014.
- Lynch K., Park F. *Modern Robotics: Mechanics, Planning, and Control*. Cambridge UP, 2017.
  http://modernrobotics.org — гл. 4, 5, 8, 11.
- MuJoCo Menagerie. https://github.com/google-deepmind/mujoco_menagerie
- MJX. https://mujoco.readthedocs.io/en/stable/mjx.html
- Gymnasium. https://gymnasium.farama.org; Gymnasium-Robotics. https://robotics.farama.org
- Andrychowicz M. et al. Hindsight Experience Replay. NeurIPS 2017.
  https://arxiv.org/abs/1707.01495

## Model-based RL и модели мира (L3)

- Sutton R. Integrated architectures for learning, planning, and reacting based on
  approximating dynamic programming. ICML 1990.
- Janner M. et al. When to Trust Your Model: Model-Based Policy Optimization. NeurIPS 2019.
  https://arxiv.org/abs/1906.08253
- ● Ha D., Schmidhuber J. World Models. 2018. https://arxiv.org/abs/1803.10122
- Hafner D. et al. Learning Latent Dynamics for Planning from Pixels. ICML 2019.
  https://arxiv.org/abs/1811.04551
- Hafner D. et al. Dream to Control. ICLR 2020. https://arxiv.org/abs/1912.01603
- Hafner D. et al. Mastering Atari with Discrete World Models. ICLR 2021.
  https://arxiv.org/abs/2010.02193
- ● Hafner D. et al. Mastering Diverse Domains through World Models. 2023; Nature 640, 2025.
  https://arxiv.org/abs/2301.04104. Код: https://github.com/danijar/dreamerv3;
  PyTorch: https://github.com/NM512/dreamerv3-torch
- Wu P. et al. DayDreamer: World Models for Physical Robot Learning. CoRL 2022.
  https://arxiv.org/abs/2206.14176
- Zhou G. et al. DINO-WM. 2024. https://arxiv.org/abs/2411.04983
- Assran M. et al. V-JEPA 2. 2025. https://arxiv.org/abs/2506.09985
- NVIDIA. Cosmos World Foundation Model Platform. 2025. https://arxiv.org/abs/2501.03575
- Kingma D., Welling M. Auto-Encoding Variational Bayes. ICLR 2014.
  https://arxiv.org/abs/1312.6114

## Планирование в латентном пространстве (L4)

- Williams G. et al. Information Theoretic MPC for Model-Based Reinforcement Learning.
  ICRA 2017.
- Hansen N., Wang X., Su H. Temporal Difference Learning for Model Predictive Control.
  ICML 2022. https://arxiv.org/abs/2203.04955
- ● Hansen N., Su H., Wang X. TD-MPC2. ICLR 2024. https://arxiv.org/abs/2310.16828.
  Код: https://github.com/nicklashansen/tdmpc2
- Bellemare M., Dabney W., Munos R. A Distributional Perspective on Reinforcement Learning.
  ICML 2017. https://arxiv.org/abs/1707.06887

## Обучение по демонстрациям и перенос (L5)

- Ross S., Bagnell D. Efficient Reductions for Imitation Learning. AISTATS 2010.
- Ross S., Gordon G., Bagnell D. A Reduction of Imitation Learning and Structured Prediction
  to No-Regret Online Learning. AISTATS 2011. https://arxiv.org/abs/1011.0686
- Zhao T. et al. Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. RSS 2023.
  https://arxiv.org/abs/2304.13705
- Chi C. et al. Diffusion Policy. RSS 2023. https://arxiv.org/abs/2303.04137
- Black K. et al. π0. 2024. https://arxiv.org/abs/2410.24164
- Shukor M. et al. SmolVLA. 2025. https://arxiv.org/abs/2506.01844
- Tobin J. et al. Domain Randomization for Transferring Deep Neural Networks from Simulation
  to the Real World. IROS 2017. https://arxiv.org/abs/1703.06907
- LeRobot. https://github.com/huggingface/lerobot; документация
  https://huggingface.co/docs/lerobot
- SO-ARM100/101. https://github.com/TheRobotStudio/SO-ARM100

## Курсы

- Levine S. CS285: Deep Reinforcement Learning, UC Berkeley.
  https://rail.eecs.berkeley.edu/deeprlcourse/ — лекции по model-based RL.
- Hugging Face Deep RL Course. https://huggingface.co/learn/deep-rl-course
- Hugging Face Robotics Course. https://huggingface.co/learn/robotics-course
