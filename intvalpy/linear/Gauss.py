import numpy as np

from ..kernel.new_objects import zeros


def Gauss(A, b):
    """
    Procedure Gauss.

    Parameters:

        A: Interval
            The input interval matrix of ISLAE, which can be either only square.

        b: Interval
            The interval vector of the right part of the ISLAE.

    Returns:

        out: Interval
            Returns an interval vector, which means an external estimate of the united solution set.

    """

    A, b = A.copy(), b.copy()
    n, _ = A.shape

    r = zeros((n, n))
    x = zeros(n)

    for j in range(n-1):
        mig = A[j:, j].mig
        argmax = np.argmax(mig)
        max_mig = mig[argmax]

        assert max_mig > 0, 'All subdiagonal elements of the column contain zeros.'

        k = argmax + j
        # swap rows, if necessary
        if k != j:
            A[[j, k], j:] = A[[k, j], j:]
            b[[j, k]] = b[[k, j]]

        # transformations of the forward step of the Gauss method
        r[j+1:, j] = A[j+1:, j]/A[j, j]
        A[j+1:, j+1:] = A[j+1:, j+1:] - r[j+1:, j] * A[j, j+1:]
        b[j+1:] = b[j+1:] - r[j+1:, j] * b[j]

    # backward step of the Gauss method
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - sum(A[i, i:] * x[i:])) / A[i, i]
    return x