"""
This script test the IMPPI approach with learnable dynamics and ground truth reward function
from the environment. MPC policy is learned by the i-mppi approach.
"""
import click
from robot_policy.mpc.imppi import IMPPI, IMPPI1
from robot_utils.py.utils import load_dict_from_yaml


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--train", "-t", is_flag=True, help="1: training, 0: testing, 2: retrain")
@click.option("--env_name", "-e", default="pendulum", help="pendulum, cartpole")
def main(train, env_name):
    config_file = "./config/imppi_model_fnn.yaml"
    config = load_dict_from_yaml(config_file)
    model_path = "./saved_model/imppi_{}".format(config['agent']['env_name'])
    if True:
        model = IMPPI1(config=config_file, model_path=model_path)
        model.print(param=False, info=True)
        model.learn()
        print("learn")
        model.save_model()
    # else:
    #     model = IMPPI1(None, model_path=model_path)
    #     model.load_model()
    #     model.print(param=False, info=True)

    # model.play(12, rollout_steps=200, progress=True)
    model.start(12, rollout_steps=200, progress=True)


if __name__ == "__main__":
    main()
