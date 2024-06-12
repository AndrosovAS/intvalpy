import numpy as np

from ..kernel.real_intervals import Interval


def Subdiff(A, b, tol=1e-12, maxiter=500, tau=1, norm_min_val=1e-12):
    """
    Subdifferential Newton method.

    Parameters:

        A: Interval
            The input interval matrix of ISLAE, which can be either square or rectangular.

        b: Interval
            The interval vector of the right part of the ISLAE.

        tol: float, optional
            An error that determines when further iterations of the algorithm are not required,
            i.e. their distance between the solution at iteration k and the solution at iteration k+1
            is "close enough" to zero.


        maxiter: int, optional
            The maximum number of iterations.

        ...

    Returns:

        out: Interval
            Returns an interval vector, which, after substituting into the system of equations
            and performing all operations according to the rules of arithmetic and analysis,
            turns the equations into true equalities.
    """


    def superMatrix(A):
        Amid = A.mid
        index = Amid >= 0
        A_plus = np.zeros(A.shape)
        A_minus = np.zeros(A.shape)
        A_plus[index] = Amid[index]
        A_minus[~index] = Amid[~index]

        result = np.zeros((2*n, 2*m))
        result[:n, :m], result[:n, m:2*m] = A_plus, A_minus
        result[n:2*n, :m], result[n:2*n, m:2*m] = A_minus, A_plus
        return result


    def calcSubgrad(F, i, j, a, b):
        n = int(F.shape[0] / 2)

        if np.sign(a.a) * np.sign(a.b) > 0:
            k = 0 if np.sign(a.a) > 0 else 2
        else:
            k = 1 if a.a < a.b else 3

        if np.sign(b.a) * np.sign(b.b) > 0:
            m = 1 if np.sign(b.a) > 0 else 3
        else:
            m = 2 if b.a <= b.b else 4

        cause = 4*k + m
        if cause == 1:
            F[i, j] = a.a
            F[i + n, j + n] = a.b
        elif cause == 2:
            F[i, j] = a.b
            F[i + n, j + n] = a.b
        elif cause == 3:
            F[i, j] = a.b
            F[i + n, j + n] = a.a
        elif cause == 4:
            F[i, j] = a.a
            F[i + n, j + n] = a.a
        elif cause == 5:
            F[i, j + n] = a.a
            F[i + n, j + n] = a.b
        elif cause == 6:
            if a.a*b.b < a.b*b.a:
                F[i, j + n] = a.a
            else:
                F[i, j] = a.b

            if a.a*b.a > a.b*b.b:
                F[i + n, j] = a.a
            else:
                F[i + n, j + n] = a.b
        elif cause == 7:
            F[i, j] = a.b
            F[i + n, j] = a.a
        elif cause == 9:
            F[i, j + n] = a.a
            F[i + n, j] = a.b
        elif cause == 10:
            F[i, j + n] = a.a
            F[i + n, j] = a.a
        elif cause == 11:
            F[i, j + n] = a.b
            F[i + n, j] = a.a
        elif cause == 12:
            F[i, j + n] = a.b
            F[i + n, j] = a.b
        elif cause == 13:
            F[i, j] = a.a
            F[i + n, j] = a.b
        elif cause == 15:
            F[i, j + n] = a.b
            F[i + n, j + n] = a.a
        elif cause == 16:
            if a.a*b.a > a.b*b.b:
                F[i, j] = a.a
            else:
                F[i, j + n] = -a.b

            if a.a*b.b < a.b*b.a:
                F[i + n, j + n] = a.a
            else:
                F[i + n, j] = a.b

        return F

    n, m = A.shape

    assert n == m, "matrix is not square"
    assert m == b.shape[0], "mismatch of matrix and vector dimensions"

    F = superMatrix(A)
    xx = np.zeros(2*n)
    xx[:n], xx[n:2*n] = b.a, b.b

    xx = np.linalg.solve(F, xx)
    r = float('inf')
    q = 1
    nit = 0
    while nit <= maxiter and r / q > tol:
        r = 0
        x = np.copy(xx)
        F = np.zeros((2*n, 2*n))

        for i in range(n):
            s = Interval(0, 0)

            for j in range(n):
                g = A[i, j]
                h = Interval(x[j], x[j+n], sortQ=False)
                t = g * h
                s = s + t
                F = calcSubgrad(F, i, j, g, h)

            t = s + b[i].opp
            xx[i] = t.a
            xx[i + n] = t.b

            r = r + t.mag

        xx = np.linalg.solve(F, xx)
        xx = x - xx * tau

        q = np.linalg.norm(xx, 1)
        if q <= norm_min_val:
            q = 1

        nit += 1
    return Interval(xx[:n], xx[n:], sortQ=False)