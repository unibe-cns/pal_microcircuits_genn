import numpy as np


def vstack(invar, newval):
    return (
        np.copy(newval) if invar is None
        else np.vstack((invar, newval))
    )
