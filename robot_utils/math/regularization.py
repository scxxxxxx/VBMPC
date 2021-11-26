import numpy as np


def nmse(target, pred):
    # (target - pred)/target = 1 - pred/target
    ratio = pred / target
    return mse(1., ratio)


def mse(target, pred):
    return np.mean((target - pred) ** 2)
