from robot_utils.math import TrajNormalizer
from robot_policy.mp.vmp import VMP
from robot_policy.mp.tvmp import TVMP
import numpy as np


def train_vmp_from_demo(motion_list, space='ts', kernel=100, plot_flag=False):
    """
    create a VMP instance and train it with the motion data, it depends on the control space
    (joint space or task space)
    :param motion_list: a list of np array of the motions with dimension [TimeSteps x (action_dim + 1)],
    with first column the time steps
    :param space: ts for task space and js for joint space
    :param kernel: kernel size
    :param plot_flag: True for plot and compare of the mp generated traj. with the first demonstrated traj.
    :return: trained VMP or TVMP instance with start and goal pose read from the motion recordings
    """
    trajs = []
    for motion in motion_list:
        traj_normalizer = TrajNormalizer(motion)
        trajs.append(traj_normalizer.normalize_timestamp())
    trajs = np.stack(trajs, axis=0)

    if space == 'ts':
        vmp = TVMP(kernel_num=int(kernel))
    elif space == 'js':
        vmp = VMP(dim=trajs.shape[1], kernel_num=int(kernel))
    vmp.train(trajs)
    # weight = vmp.get_flatten_weights()
    start = motion_list[0][0, 1:]
    goal = motion_list[0][-1, 1:]
    print("-------------- start ----------------")
    print(start)
    print("-------------- goal ----------------")
    print(goal)

    # trajectory_weights.append(weight)
    # trajectory_starts.append(y0)
    # trajectory_goals.append(g)
    # traj_js = vmp_js.roll(y0, g)
    # vmp.set_start(start)
    # vmp.set_goal(goal)
    vmp.set_start_goal(start, goal)

    if plot_flag and space == 'ts':
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 4, sharex='col')

        t = np.linspace(0, 1, 100)
        ttraj = []
        vtraj = []
        for i in range(100):
            transl, rot = vmp.get_target(t[i])
            translv, rotv = vmp.get_vel(t[i])
            # print(translv, rotv)

            ttraj.append(np.concatenate([np.array([[t[i]]]), transl, rot], axis=1))
            vtraj.append(np.concatenate([np.array([[t[i]]]), translv, rotv], axis=1))

        ttraj = np.stack(ttraj)
        # print('ttraj has shape: {}'.format(np.shape(ttraj)))

        for i in range(2):
            for j in range(4):
                if (i * 4 + j) == 7:
                    break
                ax[i, j].plot(trajs[0, :, 0], trajs[0, :, i * 4 + j + 1], 'k-.')
                ax[i, j].plot(ttraj[:, 0], ttraj[:, i * 4 + j + 1], 'r-')


        plt.show()
    return vmp


# def train_tvmp_from_demo(motion_data):
#     traj_normalizer = TrajNormalizer(motion_data)
#     trajs = traj_normalizer.normalize_timestamp()
#     trajs = np.expand_dims(trajs, axis=0)
#
#     vmp = TVMP(kernel_num=int(20))
#     vmp.train(trajs)
#     # weight = vmp.get_flatten_weights()
#     start = motion_data[0, 1:]
#     goal = motion_data[-1, 1:]
#     # trajectory_weights.append(weight)
#     # trajectory_starts.append(y0)
#     # trajectory_goals.append(g)
#     # traj_js = vmp_js.roll(y0, g)
#     vmp.set_start_goal(start, goal)
#     # print("new traj js: \n", traj_js)
#
#     t = np.linspace(0, 1, 100)
#     ttraj = []
#     vtraj = []
#     for i in range(100):
#         transl, rot = vmp.get_target(t[i])
#         translv, rotv = vmp.get_vel(t[i])
#         print(translv, rotv)
#
#         ttraj.append(np.concatenate([np.array([[t[i]]]), transl, rot], axis=1))
#         vtraj.append(np.concatenate([np.array([[t[i]]]), translv, rotv], axis=1))
#
#     ttraj = np.stack(ttraj)
#     print('ttraj has shape: {}'.format(np.shape(ttraj)))
#     import matplotlib.pyplot as plt
#     for i in range(7):
#         plt.figure(i)
#         plt.plot(trajs[0, :, 0], trajs[0, :, i + 1], 'k-.')
#         plt.plot(ttraj[:, 0], ttraj[:, i + 1], 'r-')
#
#     plt.show()
#     return vmp

