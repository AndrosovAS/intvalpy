import numpy as np

from ..kernel.utils import compmat, dist, isnan
from ..kernel.new_objects import zeros, full
from ..kernel.preprocessing import intersection


def Gauss_Seidel(A, b, x0=None, C=None, tol=1e-12, maxiter=2000):
    """
    The iterative Gauss-Seidel method for obtaining external evaluations of the united solution set
    for an interval system of linear algebraic equations (ISLAE).

    Parameters:

        A: Interval
            The input interval matrix of ISLAE, which can be either only square.

        b: Interval
            The interval vector of the right part of the ISLAE.

        X: Interval, optional
            An initial guess within which to search for external evaluation is suggested.
            By default, X is an interval vector consisting of the elements [-1000, 1000].

        C: np.array, Interval
            A matrix for preconditioning the system. By default, C = inv(mid(A)).

        tol: float, optional
            The error that determines when further crushing of the bars is unnecessary,
            i.e. their width is "close enough" to zero, which can be considered exactly zero.

        maxiter: int, optional
            The maximum number of iterations.

    Returns:

        out: Interval
            Returns an interval vector, which means an external estimate of the united solution set.
    """

    n, m = A.shape
    assert n == m, 'Matrix is not square'
    assert n == len(b), 'Inconsistent dimensions of matrix and right-hand side vector'

    A, b = A.copy(), b.copy()
    C = np.linalg.inv(A.to_float().mid) if C is None else C
    A = C @ A
    b = C @ b

    # проверим, что A является H-матрицей
    B = np.linalg.inv(np.array(compmat(A), dtype=np.float64))
    v = abs(B @ np.ones(n))
    u = A @ v
    assert (u > 0).any(), 'Matrix of the system not an H-matrix'


    distance = np.inf
    result = zeros(n)
    pre_result = full(n, -1000, 1000) if x0 is None else x0

    nit = 0
    while distance >= tol and nit <= maxiter:
        for k in range(n):
            new_bar = (b[k] - sum(A[k, :k] * result[:k]) - sum(A[k, k+1:] * pre_result[k+1:])) / A[k, k]
            result[k] = intersection(pre_result[k], new_bar)

            if isnan(result[k]):
                raise Exception("The united solution set does not intersect the bar X.")

        distance = dist(result, pre_result)
        pre_result = result.copy()
        nit += 1
    return result