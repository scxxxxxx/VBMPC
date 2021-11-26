import click
from tqdm import trange
import gym
import torch

from rl.ppo.ppo import PPO, AgentConfig, PPOTrain
from robot_policy.rl.common import ActorCriticPolicyConfig
from robot_policy.rl.common import MlpNetworkConfig


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--train", "-t", default=0, help="1: training, 0: testing, 2: retrain")
@click.option("--env_name", "-e", default="pendulum", help="pendulum, cartpole")
def main(train, env_name):
    if env_name == "pendulum":
        env = gym.make('Pendulum-v0')
        max_steps = 500000
    else:
        env = gym.make('CartPole-v1')
        max_steps = 50000

    c = AgentConfig(
        env=env,
        seed=None,
        use_gpu=True,
        policy_type="MlpPolicy",
        policy_config=ActorCriticPolicyConfig(
            learning_rate=1e-4,
            log_std_init=0.0,
            squash_output=False,
            network_config=MlpNetworkConfig(
                net_struct={"shared": [], "policy": [64, 64], "value": [64, 64]},
                activation_fn=torch.nn.ReLU,  # torch.nn.Tanh
            )
        )
    )
    t = PPOTrain(
        learning_rate=1e-4,
        rollout_steps=1024,
        max_steps=max_steps,
        batch_size=64,
        epochs=10,
        max_grad_norm=0.5,
        # on policy train
        gamma=0.99,
        gae_lambda=0.97,
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

    model_path = "./saved_model/ppo_{}".format(env_name)
    if train:
        model = PPO(c=c, t=t, model_path=model_path)
        model.print(param=False, info=True)
        model.learn()
        model.save_model()
    else:
        model = PPO(None, None, model_path=model_path)
        model.print(param=False, info=True)

    obs = env.reset()
    for i in trange(5000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
          obs = env.reset()

    env.close()


if __name__ == "__main__":
    main()
