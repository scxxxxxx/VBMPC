"""
This script test the ActorCritic IMPPI approach with learnable dynamics and learnable reward function.
The MPC policy is learned by the i-mppi approach.
"""
import click
from robot_policy.mpc.imppi import ActorCriticIMPPI
from robot_utils.py.utils import load_dict_from_yaml


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--train", "-t", is_flag=True, help="1: training, 0: testing, 2: retrain")
def main(train):
    config_file = "./config/imppi_model_fnn_critic_fnn.yaml"
    config = load_dict_from_yaml(config_file)
    #ic(config)
    model_path = "./saved_model/ac_imppi_{}".format(config['agent']['env_name'])
    if True:
        model = ActorCriticIMPPI(config=config_file, model_path=model_path)
        model.print(param=False, info=True)
        model.learn()
        model.save_model()
    # else:
    #     model = ActorCriticIMPPI(None, model_path=model_path)
    #     model.load_model()
    #     model.print(param=False, info=True)

        model.play(12, rollout_steps=200, progress=True)


if __name__ == "__main__":
    main()
