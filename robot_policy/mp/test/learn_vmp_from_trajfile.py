import click
import numpy as np

import matplotlib.pyplot as plt
from robot_policy.mp.vmp import VMP


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--motion_file", "-i", type=str, help="the absolute path of the motion recording, csv format")
@click.option("--weight_file", "-w", type=str, default="weights.csv", help="the file name to store the weights of the learned MP")
def main(motion_file, result_file):
    trajectory = np.loadtxt(motion_file, delimiter=',')
    vmp = VMP(2, kernel_num=10)
    vmp.train(trajectory)
    np.savetxt(result_file, vmp.get_flatten_weights().transpose(), delimiter=',')

    traj = vmp.roll(vmp.y0, vmp.goal)
    dim = np.size(traj, axis=1) - 1

    fig, axes = plt.subplots(nrows=dim, ncols=1)
    for i in range(dim):
        axes[i].plot(traj[:,0], traj[:,i+1])

    if dim == 2:
        fig2, ax = plt.subplots()
        ax.plot(traj[:, 1], traj[:, 2])

    plt.show()


if __name__ == "__main__":
    main()
