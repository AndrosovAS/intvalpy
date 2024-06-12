import numpy as np

from ..kernel.real_intervals import Interval


def Neumaier(A, b, y, w = None):
    """
        Calculation of the estimates of the tolerable solution sets of ISLAE
            Ax = b
        using Neumaier method.

        Parameters:
            A: Interval
                The input interval matrix of ISLAE

            b: Interval
                The interval vector of the right part of the ISLAE.

            y: np.array
                The center around which the estimation takes place

            w: np.array, optional
                Weights of each direction
        Returns:
            out: Interval
                Returns an optimal interval vector, which means an external estimate of the united solution set.
    """
    if(A.shape[0] != len(b)):
        raise Exception(f"Inconsistent dimensions matrix ({A.shape}) and right side ({len(b)})")
    if (A.shape[1] != len(y)):
        raise Exception(f"Inconsistent dimensions matrix ({A.shape}) and center point ({len(y)})")

    if w is not None:
        if (len(y) != len(w)):
            raise Exception(f"Inconsistent dimensions center point {len(b)}  and weights ({len(w)})")
        D = np.diag(w)
        A_w = A @ D
        estimation_box = D @ np.ones_like(y)
    else:
        A_w = A
        estimation_box = np.ones_like(y)
    r = (b.rad - (b.mid - A_w @ y).mag) / np.sum(A_w.mag, axis=1)
    return y + estimation_box * r * Interval(-1, 1)