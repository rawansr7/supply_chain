import numpy as np


def mase(y, f, history, m):
    errors = np.abs(y - f)
    mae = errors.mean(axis=1)

    naive_errors = np.abs(history[:, m:] - history[:, :-m])
    scale = np.nanmean(naive_errors, axis=1)

    return mae / scale
