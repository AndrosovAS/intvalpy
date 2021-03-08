import numpy as np

from intvalpy.RealInterval import Interval
from intvalpy.linear import Gauss_Seidel
from intvalpy.intoper import asinterval, intersection, dist


def HansenSengupta(func, J, x0, maxiter=2000, tol=1e-12):

    def HS(X, c):
        L = asinterval(J(X))

        if L.shape == ():
            LAMBDA = 1/L.mid
            A = LAMBDA * L
            b = -LAMBDA * func(c)

            if 0 in A:
                raise Exception('Диагональный элемент матрицы содержит нуль!')
            else:
                GS = b/A
            return c + GS

        else:
            LAMBDA = np.linalg.inv(L.mid)
            A = LAMBDA @ L
            b = -LAMBDA @ func(c)

            GS = Gauss_Seidel(A, b, X-c, tol=tol)
            return c + GS

    if not (0 in func(x0)):
        raise Exception('Брус не содержит решений!')

    result = x0
    pre_result = result.copy
    c = result.mid

    error = float('inf')
    nit = 0
    while nit < maxiter and error > tol:
        result = intersection(result, HS(result, c))
        if Interval(float('-inf'), float('-inf'), sortQ=False) in result:
            return result
        c = result.mid
        error = dist(result, pre_result)
        pre_result = result.copy
        nit += 1
    return result


def Krawczyk(func, J, x0, maxiter=2000, tol=1e-12):

    def K(X, c):
        L = asinterval(J(X))

        if L.shape == ():
            LAMBDA = 1/L.mid

            B = 1 - LAMBDA * L
            return c - LAMBDA * asinterval(func(c)) + B * (X-c)

        else:
            n, m = L.shape
            LAMBDA = np.linalg.inv(L.mid)

            B = np.eye(n) - LAMBDA @ L
            w, _ = np.linalg.eigh(abs(B))
            return c - LAMBDA @ func(c) + B @ (X-c)

    if not (0 in func(x0)):
        raise Exception('Брус не содержит решений!')

    result = x0
    pre_result = result.copy
    c = result.mid

    error = float('inf')
    nit = 0
    while nit < maxiter and error > tol:
        result = intersection(result, K(result, c))
        if Interval(float('-inf'), float('-inf'), sortQ=False) in result:
            return result
        c = result.mid
        error = dist(result, pre_result)
        pre_result = result.copy
        nit += 1

    return result
