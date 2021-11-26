import numpy as np
import matplotlib.pyplot as plt


class TrajNormalizer:
    def __init__(self, traj):
        self.orig_traj = traj

    def normalize_timestamp(self):
        traj = self.orig_traj
        timestamps = (traj[:,0] - traj[0,0])/(traj[-1,0] - traj[0,0])
        traj[:,0] = timestamps

        return traj

