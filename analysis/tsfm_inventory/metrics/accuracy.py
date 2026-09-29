import numpy as np


def mase(y, f, history, m):
    return np.abs(y - f).mean(1) / np.nanmean(np.abs(history[:, m:] - history[:, :-m]), axis=1)
