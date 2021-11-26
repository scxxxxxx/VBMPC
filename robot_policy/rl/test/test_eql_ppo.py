from icecream import ic
from tqdm import trange
import numpy as np
import gym
import torch

from equation_learner.eql_common import Layer, EQLConfig
from rl.ppo.ppo import PPO, AgentConfig, PPOTrain
from robot_policy.rl.common import ActorCriticPolicyConfig
from robot_policy.rl.common import EqlNetworkConfig

# env = gym.make('Pendulum-v0')
env = gym.make('CartPole-v1')

struct = [
    Layer(ide=10, sin=10, cos=10, mul=10, mul_mode=1),
    Layer(ide=10, sin=10, cos=10, mul=0, mul_mode=1),
    # Layer(ide=10, div=1, div_mode=2),
    # Layer(ide=2)
]
c = AgentConfig(
    env=env,
    seed=None,
    use_gpu=True,
    policy_type="MlpPolicy",
    policy_config=ActorCriticPolicyConfig(
        learning_rate=1e-4,
        log_std_init=0.0,
        squash_output=False,
        network_config=EqlNetworkConfig(
            eql_config=EQLConfig(structs=struct),
            value_net_struct=[64, 64],
            value_activation=torch.nn.ReLU,
        )
    )
)
# # for pendulum
# t = PPOTrain(
#     learning_rate=3e-4,
#     rollout_steps=1024,
#     # max_steps=2048,
#     max_steps=10000,
#     batch_size=64,
#     epochs=10,
#     max_grad_norm=0.5,
#     # on policy train
#     gamma=0.99,
#     gae_lambda=0.95,
#     ent_coef=0.0,
#     val_coef=0.5,
#     use_sde=False,
#     sde_sample_freq=-1,
#     num_env=1,
#     # ppo train
#     clip_range=0.2,
#     clip_range_vf=None,
#     target_kl=0.01
# )

# for cart-pole
t = PPOTrain(
    learning_rate=1e-3,
    rollout_steps=1024,
    # max_steps=2048,
    max_steps=int(1e4),
    batch_size=256,
    epochs=20,
    max_grad_norm=0.5,
    # on policy train
    gamma=0.98,
    gae_lambda=0.8,
    ent_coef=0.0,
    val_coef=0.5,
    use_sde=False,
    sde_sample_freq=-1,
    num_env=1,
    # ppo train
    clip_range=0.2,
    clip_range_vf=None,
    target_kl=0.01
)

model = PPO(c=c, t=t)
model.learn()

obs = env.reset()
for i in trange(5000):
    action, _states = model.predict(obs, deterministic=True)
    # ic(action)
    # action = np.array([action])
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
