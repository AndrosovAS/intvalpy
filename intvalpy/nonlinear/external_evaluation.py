import numpy as np

from intvalpy.linear import Gauss_Seidel, PSS
from intvalpy.utils import asinterval, intersection, dist, infinity, isnan
from intvalpy.RealInterval import ARITHMETICS


def HansenSengupta(func, J, x0, maxiter=2000, tol=1e-12):

    def HS(X, c):
        L = asinterval(J(X))

        if isinstance(L, ARITHMETICS):
            LAMBDA = 1/L.mid
            A = LAMBDA * L
            b = -LAMBDA * func(c)

            if 0 in A:
                raise Exception('Диагональный элемент матрицы содержит нуль!')
            else:
                GS = b/A
            return c + GS

        else:
            LAMBDA = np.linalg.inv(L.to_float().mid)
            A = LAMBDA @ L
            b = -LAMBDA @ func(c)

            pss = PSS(A, b, tol=tol)
            return c + pss

    if not (0 in func(x0)):
        raise Exception('Брус не содержит решений!')

    result = x0
    pre_result = result.copy
    c = asinterval(result.mid)

    error = infinity
    nit = 0
    while nit < maxiter and error > tol:
        result = intersection(result, HS(result, c))
        if isnan(result).any():
            return result
        c = asinterval(result.mid)
        error = dist(result, pre_result)
        pre_result = result.copy
        nit += 1
    return result


def Krawczyk(func, J, x0, maxiter=2000, tol=1e-12):

    def K(X, c):
        L = asinterval(J(X))

        if isinstance(L, ARITHMETICS):
            LAMBDA = 1/L.mid

            B = 1 - LAMBDA * L
            return c - LAMBDA * asinterval(func(c)) + B * (X-c)

        else:
            n, m = L.shape
            LAMBDA = np.linalg.inv(L.to_float().mid)

            B = np.eye(n) - LAMBDA @ L
            w, _ = np.linalg.eigh(B.to_float().mag)
            return c - LAMBDA @ func(c) + B @ (X-c)

    if not (0 in func(x0)):
        raise Exception('Брус не содержит решений!')

    result = x0
    pre_result = result.copy
    c = asinterval(result.mid)

    error = infinity
    nit = 0
    while nit <= maxiter and error > tol:
        result = intersection(result, K(result, c))
        if isnan(result).any():
            return result
        c = asinterval(result.mid)
        error = dist(result, pre_result)
        pre_result = result.copy
        nit += 1

    return result
