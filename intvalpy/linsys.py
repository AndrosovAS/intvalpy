import numpy as np
from copy import deepcopy

from bisect import bisect_left
from joblib import Parallel, delayed

from .MyClass import Interval
from .intoper import *
from .recfunc import Tol


def ive(A, b, N=40):
    """
    Вычисление меры вариабельности оценки параметров.

    Parameters:
                A: Interval
                    Матрица ИСЛАУ.

                b: Interval
                    Вектор правой части ИСЛАУ.

    Optional Parameters:
                N: int
                    Количество угловых матриц для которых вычисляется обусловленность.

    Returns:
                out: float
                    Возвращается мера вариабельности IVE.
    """

    success, _arg_max, _max = Tol(A, b, maxQ=True)
    if not success:
        print('Оптимизация функционала Tol завершена некорректно!')

    _inf = A.a
    _sup = A.b
    cond = float('inf')
    angle_A = np.zeros(A.shape, dtype='float64')
    for _ in range(N):
        for k in range(A.shape[0]):
            for l in range(A.shape[1]):
                angle_A[k, l] = np.random.choice([_inf[k,l], _sup[k,l]])
        tmp = np.linalg.cond(angle_A)
        cond = tmp if tmp<cond else cond

    return np.sqrt(A.shape[1]) * _max * cond * \
           (np.linalg.norm(_arg_max, ord=2)/np.sqrt(sum(abs(b)**2)))


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

    WorkListA = deepcopy(asinterval(A))
    WorkListb = deepcopy(asinterval(b))

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
        WorkListA[j+1:, j+1:] -= r[j+1:, j] * WorkListA[j, j+1:]
        WorkListb[j+1:] -= r[j+1:, j] * WorkListb[j]

    for i in range(n-1, -1, -1):
        x[i] = (WorkListb[i] - sum(WorkListA[i, i:] * x[i:])) / WorkListA[i, i]

    return x


def Gauss_Seidel(A, b, x0=Interval(-10**3, 10**3), P=True, tol=1e-8, maxiter=10**3):
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

    A = deepcopy(asinterval(A))
    b = deepcopy(asinterval(b))

    if A.shape == () or A.shape == (1, 1):
        if 0 in A:
            raise Exception('Диагональный элемент матрицы содержит нуль!')
        return b/A

    if P:
        P = np.linalg.inv(A.mid)
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
    pre_result = zeros(n) + x0

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
        pre_result = deepcopy(result)
        nit += 1
    return result


def overdetermined(A, b, tol=1e-8, maxiter=10**3):
    n, m = A.shape
    assert m <= n, 'Количество строк должно быть не меньше, чем количество столбцов!'

    R = np.linalg.inv(A.mid.T @ A.mid) @ A.mid.T
    x0 = R @ b.mid

    G = abs(np.eye(m) - R @ A.mid) + abs(R) @ A.rad
    g = abs(R @ (A.mid @ x0 - b.mid)) + abs(R) @ (A.rad @ abs(x0) + b.rad)

    result = np.zeros(m)
    error = float('inf')
    nit = 0
    while error >= tol and nit <= maxiter:
        d = deepcopy(result)
        result = G @ d + g + tol
        error = np.amax(abs(result-d))
        nit += 1
    return Interval(x0-result, x0+result)


class KeyWrapper:
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)


def PSS(A, b, V=None, tol=1e-12, maxiter=10**3):
    """
    Метод дробления решений для оптимального внешнего оценивания
    объединённого множества решений.

    Parameters:
                A: Interval
                    Матрица ИСЛАУ.

                b: Interval
                    Вектор правой части ИСЛАУ.

    Optional Parameters:
                V: Interval
                    Начальный брус, в котором ищется решение.

                tol: float
                    Погрешность для остановки процесса дробления решений.

                maxiter: int
                    Максимальное количество итераций.

    Returns:
                out: Interval
                    Возвращается интервальный вектор решений.
    """

    if V is None:
        V = zeros(A.shape[1]) + Interval(-1e3, 1e3)

    def kdiv(a, b):
        """
        Деление в арифметике Кахана.
        """
        if 0 in a and 0 in b:
            return Interval(float('-inf'), float('inf'), sortQ=False)
        elif a.b<0 and b.a < b.b and b.b==0:
            return Interval(a.b/b.a, float('inf'), sortQ=False)
        elif a.b<0 and b.a<0 and 0<b.b:
            return Interval([float('-inf'), a.b/b.a],
                            [a.b/b.b, float('inf')], sortQ=False)
        elif a.b<0 and b.a==0 and b.a<b.b:
            return Interval(float('-inf'), a.b/b.b, sortQ=False)
        elif 0<a.a and b.a<b.b and b.b==0:
            return Interval(float('-inf'), a.a/b.a, sortQ=False)
        elif 0<a.a and b.a<0 and 0<b.b:
            return Interval([float('-inf'), a.a/b.b],
                            [a.a/b.a, float('inf')], sortQ=False)
        elif 0<a.a and 0==b.a and b.a<b.b:
            return Interval(a.a/b.b, float('inf'), sortQ=False)
        else:
            raise Exception('Деление не определено!')

    def Omega(A, b, r, nu, Anu):
        Ar = A @ r
        bAr = b - Ar
        index = np.where(Anu.a*Anu.b > 0)[0]
        div = bAr[index]/Anu[index]

        index = np.where(Anu.a*Anu.b <= 0)[0]
        pdiv = []
        for k in index:
            pdiv.append(kdiv(bAr[k], Anu[k]))

        result = intersection(div[0], V[nu])
        for el in div[1:]:
            result = intersection(result, el)
        for el in pdiv:
            result = intersection(result, el)

        if Interval(float('-inf'),float('-inf'),sortQ=False) in result:
            return float('inf')
        elif result.shape == ():
            return result.a
        else:
            return min(result.a)


    def algo(nu):
        WorkListA = deepcopy(A); del(WorkListA[:, nu])
        WorkListb = deepcopy(b*endint)
        Q = deepcopy(V); del(Q[nu])
        q = V[nu].a
        Anu = A[:, nu]
        L = [(Q, q)]

        nit = 0
        while np.amax(Q.wid) >= tol and nit <= maxiter:
            k = np.argmax(Q.wid)
            Q1 = deepcopy(L[0][0])
            Q2 = deepcopy(L[0][0])
            Q1[k], Q2[k] = Interval(Q[k].a, Q[k].mid, sortQ=False), Interval(Q[k].mid, Q[k].b, sortQ=False)

            q1, q2 = Omega(WorkListA, WorkListb, Q1, nu, Anu), Omega(WorkListA, WorkListb, Q2, nu, Anu)
            del L[0]

            newcol = (Q1, q1)
            bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
            L.insert(bslindex, newcol)

            newcol = (Q2, q2)
            bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
            L.insert(bslindex, newcol)

            Q, q = L[0]
            nit += 1
        return -L[0][1] if endint == -1 else L[0][1]

    for endint in [1, -1]:
        res = Parallel(n_jobs=-1)(delayed(algo)(nu) for nu in range(A.shape[1]))
        if endint == -1:
            sup = res
        else:
            inf = res
    return Interval(inf, sup, sortQ=False)
