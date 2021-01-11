import numpy as np

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

    A = asinterval(A).copy
    b = asinterval(b).copy

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
        pre_result = result.copy
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
        d = np.copy(result)
        result = G @ d + g + tol
        error = np.amax(abs(result-d))
        nit += 1
    return Interval(x0-result, x0+result)



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

    class KeyWrapper:
        def __init__(self, iterable, key):
            self.it = iterable
            self.key = key

        def __getitem__(self, i):
            return self.key(self.it[i])

        def __len__(self):
            return len(self.it)



    def kdiv(a, b, num_func):
        """
        Деление в арифметике Кахана.
        """
        func0 = lambda a, b: Interval(float('-inf'), a.b/b.b, sortQ=False)
        func1 = lambda a, b: Interval(a.b/b.a, float('inf'), sortQ=False)
        func2 = lambda a, b: Interval([float('-inf'), a.b/b.a],
                                      [a.b/b.b, float('inf')], sortQ=False)

        func3 = lambda a, b: Interval(a.a/b.b, float('inf'), sortQ=False)
        func4 = lambda a, b: Interval(float('-inf'), a.a/b.a, sortQ=False)
        func5 = lambda a, b: Interval([float('-inf'), a.a/b.b],
                                      [a.a/b.a, float('inf')], sortQ=False)

        kdiv_result = [func0, func1, func2, func3, func4, func5]

        if 0 in a:
            return Interval(float('-inf'), float('inf'), sortQ=False)
        elif a.b<0:
            return kdiv_result[num_func](a, b)
        elif 0<a.a:
            return kdiv_result[3+num_func](a, b)
        else:
            raise Exception('Деление не определено!')

    # def kdiv(a, b):
    #     """
    #     Деление в арифметике Кахана.
    #     """
    #
    #     if 0 in a:
    #         return Interval(float('-inf'), float('inf'), sortQ=False)
    #
    #     elif a.b<0:
    #         if not b.a:
    #             return Interval(float('-inf'), a.b/b.b, sortQ=False)
    #         elif not b.b:
    #             return Interval(a.b/b.a, float('inf'), sortQ=False)
    #         else:
    #             return Interval([float('-inf'), a.b/b.a],
    #                             [a.b/b.b, float('inf')], sortQ=False)
    #     elif 0<a.a:
    #         if not b.a:
    #             return Interval(a.a/b.b, float('inf'), sortQ=False)
    #         elif not b.b:
    #             return Interval(float('-inf'), a.a/b.a, sortQ=False)
    #         else:
    #             return Interval([float('-inf'), a.a/b.b],
    #                             [a.a/b.a, float('inf')], sortQ=False)
    #     else:
    #         raise Exception('Деление не определено!')


    def Omega(bAr, nu, Anu, index_classic_div, index_kahan_div, num_func):
        div = bAr[index_classic_div]/Anu[index_classic_div]
        pdiv = []

        for k in index_kahan_div:
            pdiv.append(kdiv(bAr[k], Anu[k], num_func[k]))

        result = intersection(div[0], V[nu])
        for el in div[1:]:
            result = intersection(result, el)
        for el in pdiv:
            result = intersection(result, el)

        if float('-inf') in result._b:
            return float('inf')
        elif result.shape == ():
            return result.a
        else:
            return min(result.a)


    def algo(nu):
        WorkListA = A.copy; del(WorkListA[:, nu])
        WorkListb = (b*endint).copy
        Q = V.copy; del(Q[nu])
        q = V[nu].a

        Anu = A[:, nu]
        index_classic_div = np.where(Anu.a*Anu.b > 0)[0]
        index_kahan_div = np.array([k for k in range(len(Anu.a)) if not k in index_classic_div])

        num_func = np.zeros(len(Anu), dtype='int32')
        for index in index_kahan_div:
            if not Anu[index].a:
                 num_func[index] = 0
            elif not Anu[index].b:
                 num_func[index] = 1
            else:
                 num_func[index] = 2

        L = [(Q, q)]
        Ar = WorkListA @ Q

        nit = 0
        while np.amax(Q.wid) >= tol and nit <= maxiter:
            k = np.argmax(Q.wid)
            Q1 = L[0][0].copy
            Q2 = L[0][0].copy
            Q1[k], Q2[k] = Interval(Q[k].a, Q[k].mid, sortQ=False), Interval(Q[k].mid, Q[k].b, sortQ=False)

            QA = Q[k] * WorkListA[:, k]
            Ar._a -= QA._a
            Ar._b -= QA._b

            Ar1 = Ar + Q1[k] * WorkListA[:, k]
            bAr = WorkListb - Ar1
            q1 = Omega(bAr, nu, Anu, index_classic_div, index_kahan_div, num_func)

            Ar2 = Ar + Q2[k] * WorkListA[:, k]
            bAr = WorkListb - Ar2
            q2 = Omega(bAr, nu, Anu, index_classic_div, index_kahan_div, num_func)

            del L[0]

            newcol = (Q1, q1, Ar1)
            bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
            L.insert(bslindex, newcol)

            newcol = (Q2, q2, Ar2)
            bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
            L.insert(bslindex, newcol)

            Q, q, Ar = L[0]
            nit += 1

        return -L[0][1] if endint == -1 else L[0][1]

    inf=[]
    sup=[]
    for endint in [1, -1]:
        # for nu in range(A.shape[1]):
        #     if endint==1:
        #         inf.append(algo(nu))
        #     else:
        #         sup.append(algo(nu))

        res = Parallel(n_jobs=-1)(delayed(algo)(nu) for nu in range(A.shape[1]))
        if endint == -1:
            sup = res
        else:
            inf = res

    return Interval(inf, sup, sortQ=False)
