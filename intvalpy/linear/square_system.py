import numpy as np

from intvalpy.RealInterval import Interval
from intvalpy.utils import asinterval, zeros, dist, intersection, diag


def Gauss(A, b):
    """
    Метод Гаусса для решения ИСЛАУ.

    Parameters:
                A: Interval
                    Матрица ИСЛАУ.

                b: Interval
                    Вектор правой части ИСЛАУ.

    Returns:
                out: Interval
                    Возвращается интервальный вектор решений.
    """

    WorkListA = asinterval(A).copy
    WorkListb = asinterval(b).copy

    n, _ = WorkListA.shape
    mignitude = diag(A).mig
    _abs_A = A.mag
    for k in range(n):
        if mignitude[k] < sum(_abs_A[k]) - _abs_A[k, k]:
            raise Exception('Матрица А не является H матрицей!')

    r = zeros((n, n))
    x = zeros(n)

    for j in range(n-1):
        r[j+1:, j] = WorkListA[j+1:, j]/WorkListA[j, j]
        WorkListA[j+1:, j+1:] = WorkListA[j+1:, j+1:] - r[j+1:, j] * WorkListA[j, j+1:]
        WorkListb[j+1:] = WorkListb[j+1:] - r[j+1:, j] * WorkListb[j]

    for i in range(n-1, -1, -1):
        x[i] = (WorkListb[i] - sum(WorkListA[i, i:] * x[i:])) / WorkListA[i, i]
    return x


def Gauss_Seidel(A, b, x0=None, P=True, tol=1e-8, maxiter=10**3):
    """
    Итерационный метод Гаусса-Зейделя для решения ИСЛАУ.

    Parameters:
                A: Interval
                    Матрица ИСЛАУ.

                b: Interval
                    Вектор правой части ИСЛАУ.

    Optional Parameters:
                x0: Interval
                    Начальный брус, в котором ищется решение.

                P: Interval
                    Матрица предобуславливания.
                    В случае, если параметр не задан, то берётся обратное среднее.

                tol: float
                    Погрешность для остановки итерационного процесса.

                maxiter: int
                    Максимальное количество итераций.

    Returns:
                out: Interval
                    Возвращается интервальный вектор решений.
    """

    A = asinterval(A).copy
    b = asinterval(b).copy

    if A.shape == () or A.shape == (1, 1):
        if 0 in A:
            raise Exception('Диагональный элемент матрицы содержит нуль!')
        return b/A

    if P:
        P = np.linalg.inv(np.array(A.mid, dtype=np.float64))
        A = P @ A
        b = P @ b

    n, _ = A.shape
    mignitude = diag(A).mig
    _abs_A = A.mag
    for k in range(n):
        if mignitude[k] < sum(_abs_A[k]) - _abs_A[k, k]:
            raise Exception('Матрица А не является H матрицей!')

    error = float("inf")
    result = zeros(n)

    if x0 is None:
        pre_result = zeros(n) + Interval(-1000, 1000, sortQ=False)
    else:
        pre_result = x0.copy

    nit = 0
    while error >= tol and nit <= maxiter:
        for k in range(n):
            tmp = 0
            for l in range(n):
                if l != k:
                    tmp += A[k, l] * pre_result[l]
            result[k] = intersection(pre_result[k], (b[k]-tmp)/A[k, k])

            if float('-inf') in result[k]:
                raise Exception("Интервалы не пересекаются!")

        error = dist(result, pre_result)
        pre_result = result.copy
        nit += 1

    return result


def Subdiff(A, b, tol=1e-12, maxiter=500, tau=1, norm_min_val=1e-12):

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
