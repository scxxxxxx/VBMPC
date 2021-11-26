import pickle
import logging
import argparse
import numpy as np
import torch
from icecream import ic

import robot_env
from robot_env.utils import NormalizeActionEnv

# TODO
from robot_policy.mpc.mpc_common.archive.mppi import MPPI
from robot_policy.mpc.mpc_common.ilqr import iLQR
from robot_policy.mpc.mpc_common.shooting import Shooting

from robot_policy.mpc.mpc_common.twin_env import Model, EQLModel, MDNModel
from robot_policy.mpc.mpc_common.dynamic_optimizer import ModelOptimizer, EqlModelOptimizer, MDNModelOptimizer
from robot_policy.mpc.mpc_common.archive.buffer import ReplayBuffer

from robot_utils.py.utils import load_dict_from_yaml
from robot_utils.gym.gym_utils import get_space_dim
from robot_utils.py.utils import create_path
from robot_utils.torch.torch_utils import init_torch
device = init_torch()


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, help=robot_env.get_env_list())
parser.add_argument('--max_steps',  type=int,   default=200)
parser.add_argument('--max_frames', type=int,   default=10000)
parser.add_argument('--frame_skip', type=int,   default=2)
parser.add_argument('--model_lr',   type=float, default=3e-4)
parser.add_argument('--policy_lr',  type=float, default=3e-4)

parser.add_argument('--seed',       type=int,   default=666)

parser.add_argument('--horizon',    type=int,   default=5)
parser.add_argument('--model_iter', type=int,   default=2)

parser.add_argument('--method',     type=str,   default='mppi')

parser.add_argument("--model",      type=str,   default='mlp')

parser.add_argument('--done_util',      dest='done_util',   action='store_true')
parser.add_argument('--no_done_util',   dest='done_util',   action='store_false')
parser.set_defaults(done_util=True)

parser.add_argument('--render',         dest='render',      action='store_true')
parser.add_argument('--no_render',      dest='render',      action='store_false')
parser.set_defaults(render=False)

parser.add_argument('--log',            dest='log',         action='store_true')
parser.add_argument('--no-log',         dest='log',         action='store_false')
parser.set_defaults(log=False)
args = parser.parse_args()


def main():
    env_name = args.env
    try:
        env = NormalizeActionEnv(robot_env.env_list[env_name](render=args.render))
    except TypeError as err:
        env = NormalizeActionEnv(robot_env.env_list[env_name]())
    env.reset()
    # set_random_seeds(args.seed)

    if args.log:
        dir_name = 'seed_{}/'.format(args.seed)
        path = './saved_models/' + args.method + '/' + args.env + '/' + dir_name
        create_path(path)

    a_dim = get_space_dim(env.action_space)
    s_dim = get_space_dim(env.observation_space)
    ic(a_dim, s_dim)

    replay_buffer_size = 100000
    model_replay_buffer = ReplayBuffer(replay_buffer_size)

    if args.model == "mlp":
        model = Model(s_dim, a_dim, def_layers=[200]).to(device)
        optimizer = ModelOptimizer(model, model_replay_buffer, lr=args.model_lr)
    elif args.model == 'mdn':
        model = MDNModel(s_dim, a_dim, def_layers=[200, 200])
        optimizer = MDNModelOptimizer(model, model_replay_buffer, lr=args.model_lr)
    elif args.model == "eql":
        ic("args.model")
        config_file = f"config/eql.yaml"
        config = load_dict_from_yaml(config_file)

        model = EQLModel(s_dim, a_dim, config, device, env_name=env).to(device)
        optimizer = EqlModelOptimizer(model, model_replay_buffer, lr=args.model_lr, max_steps=args.max_frames)
    else:
        raise NotImplementedError

    methods = {'ilqr': iLQR, 'shooting': Shooting, 'mppi': MPPI}
    mpc_planner = methods[args.method](model, horizon=args.horizon, eps=1, device=device)

    max_frames = args.max_frames
    max_steps = 1000
    frame_skip = args.frame_skip

    frame_idx = 0
    rewards = []
    batch_size = 256

    ep_num = 0
    while frame_idx < max_frames:
        state = env.reset()
        mpc_planner.reset()

        action = mpc_planner.update(state)

        episode_reward = 0
        done = False
        for step in range(max_steps):
            for _ in range(frame_skip):
                next_state, reward, done, _ = env.step(action.copy())

            next_action = mpc_planner.update(next_state)

            if args.method == 'ilqr' or args.method == 'shooting':
                eps = 1.0 * (0.995 ** frame_idx)
                next_action = next_action + np.random.normal(0., eps, size=(a_dim,))

            model_replay_buffer.push(state, action, reward, next_state, next_action, done)

            if len(model_replay_buffer) > batch_size:
                optimizer.update_model(frame_idx, batch_size, mini_iter=args.model_iter)

            state = next_state
            action = next_action
            episode_reward += reward
            frame_idx += 1

            if (not args.render and frame_idx > 0.8 * max_frames) or args.render:
                env.render()

            if frame_idx % 100 == 0:
                last_reward = rewards[-1][1] if len(rewards)>0 else 0
                logging.info('frame : {}/{}: \t last rew: {}'.format(frame_idx, max_frames, last_reward))

                if args.log:
                    pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
                    torch.save(model.state_dict(), path + 'model_' + str(frame_idx) + '.pt')

            if args.done_util:
                if done:
                    break

        logging.info('episodic rew: {} \t {}'.format(ep_num, episode_reward))
        rewards.append([frame_idx, episode_reward])
        ep_num += 1

    if args.log:
        pickle.dump(rewards, open(path + 'reward_data' + '.pkl', 'wb'))
        torch.save(model.state_dict(), path + 'model_' + 'final' + '.pt')

# python train_mpc.py --env PendulumEnv --max_frames 6000 --frame_skip 4 --render --model eql --method mppi --model_iter 2
# python train_mpc.py --env Walker2DEnv --max_frames 6000 --frame_skip 4 --render --model eql --method mppi --model_iter 20
# python train_mpc.py --env AntBulletEnv --max_frames 6000 --frame_skip 5 --render --model eql --method mppi --model_iter 20
# python train_mpc.py --env AntBulletEnv --max_frames 6000 --frame_skip 5 --render --model eql --method mppi --model_iter 20
if __name__ == '__main__':
    main()
