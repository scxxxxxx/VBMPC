import numpy as np
from mujoco_py.generated import const


class Arrow(object):
    __slots__ = ["pos", "size", "mat", "rgba", "geom_type", "label"]

    def __init__(self):
        self.pos = np.zeros(3)
        self.size = np.zeros(3)
        self.mat = np.zeros((3, 3))
        self.rgba = np.ones(4)
        self.geom_type = const.GEOM_ARROW,
        self.label = "arrow"
