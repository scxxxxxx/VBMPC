import click
from robot_policy.rl.ppo.ppo import PPO, PPOConfig


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--train",    "-t", is_flag=True,                 help="1: training, 0: testing")
@click.option("--env_name", "-e", default="pendulum",           help="pendulum, cartpole")
@click.option("--config",   "-c", default="./config/ppo.yaml",  help="configuration file")
def main(train, env_name, config):
    model_path = f"./saved_model/ppo_{env_name}"

    config = config if train else None
    model = PPO(config=config, model_path=model_path)
    model.print(param=False, info=True)

    if train:
        model.learn()
        model.save_model()
    else:
        model.load_model()

    model.play(10, progress=True)
    model.env.close()


if __name__ == "__main__":
    main()
