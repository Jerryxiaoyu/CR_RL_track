from gym.envs.registration import registry, register, make, spec

from core.my_envs.cartpole_swingup.cartpole_swingup_env import CartPoleSwingUpEnv
from core.my_envs.cartpole_swingup.pendulum_custom_env import PendulumCustomEnv


register(
    id='CartpoleSwingUp-v0',
    entry_point='core.my_envs.cartpole_swingup.cartpole_swingup_env:CartPoleSwingUpEnv',
    max_episode_steps=50
)

register(
    id='PendulumCustom-v0',
    entry_point='core.my_envs.cartpole_swingup.pendulum_custom_env:PendulumCustomEnv',
    max_episode_steps=200,
)


