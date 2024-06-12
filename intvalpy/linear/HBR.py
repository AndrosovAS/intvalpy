import numpy as np

from ..kernel.real_intervals import Interval
from ..kernel.utils import diag, compmat


def HBR(A, b):
    """
    Procedure Hansen-Bliek-Rohn.

    Parameters:

        A: Interval
            The input interval matrix of ISLAE, which can be either only square.

        b: Interval
            The interval vector of the right part of the ISLAE.

    Returns:

        out: Interval
            Returns an interval vector, which means an external estimate of the united solution set.

    """

    n, m = A.shape
    assert n == m, 'Matrix is not square'
    assert n == len(b), 'Inconsistent dimensions of matrix and right-hand side vector'

    # создадим глубокие копии и сделаем предобуславливание
    A, b = A.copy(), b.copy()
    C = np.linalg.inv(A.to_float().mid)
    A = C @ A
    b = C @ b

    # проверим, что A является H-матрицей
    dA = diag(A)
    A = compmat(A)
    B = np.linalg.inv(np.array(A, dtype=np.float64))
    v = abs(B @ np.ones(n))
    u = A @ v
    assert (u > 0).any(), 'Matrix of the system not an H-matrix'

    # проводим процедуру Хансена-Блика-Рона
    dAc = np.diag(A)
    A = A @ B - np.eye(n)
    w = np.max(-A / np.outer(u, np.ones(n)), axis=0)
    dlow = -(v*w - np.diag(B))
    B = B + v @ w
    u = B @ b.mag
    d = np.diag(B)
    alpha = dAc + (-1)/d
    if len(b.shape) == 1:
        beta = u / dlow - b.mag
        return (b + Interval(-beta, beta)) / (dA + Interval(-alpha, alpha))
    else:
        v = np.ones(n)
        beta = u / (d @ v) - b.mag
        return (b + Interval(-beta, beta)) / ( (dA + Interval(-alpha, alpha)) @ v )