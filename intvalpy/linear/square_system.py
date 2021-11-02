import numpy as np

from intvalpy.RealInterval import Interval
from intvalpy.intoper import asinterval, zeros, dist, intersection, diag


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
    _abs_A = abs(A)
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
    _abs_A = abs(A)
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
