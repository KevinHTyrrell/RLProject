import numpy as np


def concat(to_append, arr):
    if arr is None:
        return np.expand_dims(to_append, axis=0)
    else:
        to_append_expand = np.expand_dims(to_append, axis=0)
        to_return = np.concatenate([arr, to_append_expand], axis=0)
        return to_return