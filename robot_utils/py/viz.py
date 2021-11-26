import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def plot_mesh(x, y, ax=None, up_down_dir=True, cmap='Spectral', linewidths=0.5, **kwargs):
    """
    x, y are the x-/y-coordinates of the 2d grid. Both have dimensionality M x N
    Args:
        x (M x N): x coordinates
        y (M x N): y coordinates
        up_down_dir: set the color spectrum to change in the up-down direction. Set false will use left-right dir.
        ax (): axis to plot the data
        **kwargs ():

    Returns:

    """
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)            # (M, N, 2), each entry in the MxN matrix is a 2d coordinate (x, y)
    segs2 = segs1.transpose(1, 0, 2)            # (N, M, 2)

    # extract N-1 segments from each row and in total (M, N-1, 2, 2). each entry in the (M x N-1) matric is a line
    # segments containing two points [(x1, y1), (x2, y2)] (horizontal lines)
    segs1 = np.expand_dims(segs1, axis=-2)                          # (M, N, 1, 2)
    segs1 = np.concatenate([segs1[:, :-1], segs1[:, 1:]], axis=-2)  # (M, N-1, 2, 2)
    # this is for the vertical lines, similar as above
    segs2 = np.expand_dims(segs2, axis=-2)
    segs2 = np.concatenate([segs2[:, :-1], segs2[:, 1:]], axis=-2)

    # now, we concern the order of all the line segments to assign a correct color maps.
    if up_down_dir:
        # we consider the 2 horizontally neighboring segments are close and should use similar colors. So we
        # concatenate the vertical lines right after each row of horizontal lines.
        segs = np.concatenate([segs1[:-1], segs2.transpose(1, 0, 2, 3)], axis=1)
        segs = segs.reshape(-1, 2, 2)
        # append the last row of segs1 (so that the dimensionality of the matrices match)
        segs = np.concatenate([segs, segs1[-1]])
    else:
        segs = np.concatenate([segs1.transpose(1, 0, 2, 3), segs2[:-1]], axis=1)
        segs = segs.reshape(-1, 2, 2)
        segs = np.concatenate([segs, segs2[-1]])

    cm = plt.get_cmap(cmap)
    c = cm(np.linspace(0, 1, segs.shape[0]))

    ax.add_collection(LineCollection(segs, colors=c, linewidths=linewidths, **kwargs))
    ax.autoscale()


if __name__ == "__main__":
    fig, ax = plt.subplots()
    grid_x, grid_y = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 40))
    plot_mesh(grid_x, grid_y, ax=ax, up_down_dir=False)
    plt.show()
