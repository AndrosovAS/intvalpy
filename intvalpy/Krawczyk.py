import numpy as np
from copy import deepcopy

from .MyClass import Interval
from .intoper import asinterval, intersection, dist


def Krawczyk(func, J, x0, maxiter=10**3, tol=1e-12):
    msg = ''

    def K(X, c):
        nonlocal msg
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
            if msg == '' and max(abs(w)) > 1:
                msg = 'Спектральный радиус матрицы ρ(|I - Λ·L|) = {:.2f} больше единицы!'.format(max(abs(w)))
            return c - LAMBDA @ func(c) + B @ (X-c)

    if not (0 in func(x0)):
        raise Exception('Брус не содержит решений!')

    result = x0
    pre_result = deepcopy(result)
    c = result.mid

    error = float('inf')
    nit = 0
    while nit < maxiter and error > tol:
        result = intersection(result, K(result, c))
        if Interval(float('-inf'), float('-inf'), sortQ=False) in result:
            return result
        c = result.mid
        error = dist(result, pre_result)
        pre_result = deepcopy(result)
        nit += 1

    if msg != '':
        print(msg)
    return result
