import numpy as np
from mpmath import *

from bisect import bisect_left
from lpsolvers import solve_lp

from intvalpy.RealInterval import Interval
from intvalpy.intoper import intersection, infinity
from intvalpy.linear.square_system import Gauss


def Rohn(A, b, tol=1e-12, maxiter=2000):
    """
    Метод Дж. Рона для переопределённых ИСЛАУ.

    Parameters:
                A: Interval
                    Матрица ИСЛАУ.

                b: Interval
                    Вектор правой части ИСЛАУ.

    Optional Parameters:
                tol: float
                    Погрешность для остановки итерационного процесса.

                maxiter: int
                    Максимальное количество итераций.

    Returns:
                out: Interval
                    Возвращается интервальный вектор решений.
    """

    Amid = np.array(A.mid, dtype=np.float64)
    Ac_plus = np.linalg.inv(Amid.T @ Amid) @ Amid.T
    A = Ac_plus @ A.copy
    b = Ac_plus @ b.copy

    n, m = A.shape
    assert m <= n, 'Количество строк должно быть не меньше, чем количество столбцов!'

    Amid = np.array(A.mid, dtype=np.float64)
    R = np.linalg.inv(Amid.T @ Amid) @ Amid.T
    x0 = R @ b.mid

    G = abs(np.eye(m) - R @ Amid) + abs(R) @ A.rad
    g = abs(R @ (Amid @ x0 - b.mid)) + abs(R) @ (A.rad @ abs(x0) + b.rad)

    result = np.zeros(m)
    error = infinity
    nit = 0
    while error >= tol and nit <= maxiter:
        d = np.copy(result)
        result = G @ d + g + tol
        error = np.amax(abs(result-d))
        nit += 1
    return Interval(x0-result, x0+result)


def PSS(A, b, tol=1e-12, maxiter=2000, nu=None):
    """
    Метод дробления решений для оптимального внешнего оценивания
    объединённого множества решений.

    Parameters:
                A: Interval
                    Матрица ИСЛАУ.

                b: Interval
                    Вектор правой части ИСЛАУ.

    Optional Parameters:
                tol: float
                    Погрешность для остановки процесса дробления решений.

                maxiter: int
                    Максимальное количество итераций.

    Returns:
                out: Interval
                    Возвращается интервальный вектор решений.
    """


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
        func0 = lambda a, b: Interval(-infinity, a.b/b.b, sortQ=False)
        func1 = lambda a, b: Interval(a.b/b.a, infinity, sortQ=False)
        func2 = lambda a, b: Interval([-infinity, a.b/b.a],
                                      [a.b/b.b, infinity], sortQ=False)

        func3 = lambda a, b: Interval(a.a/b.b, infinity, sortQ=False)
        func4 = lambda a, b: Interval(-infinity, a.a/b.a, sortQ=False)
        func5 = lambda a, b: Interval([-infinity, a.a/b.b],
                                      [a.a/b.a, infinity], sortQ=False)

        kdiv_result = [func0, func1, func2, func3, func4, func5]

        if 0 in a:
            return Interval(-infinity, infinity, sortQ=False)
        elif a.b<0:
            return kdiv_result[num_func](a, b)
        elif 0<a.a:
            return kdiv_result[3+num_func](a, b)
        else:
            raise Exception('Деление не определено!')


    def Omega(bAr, _nu, Anu, index_classic_div, index_kahan_div, num_func):
        try:
            div = bAr[index_classic_div]/Anu[index_classic_div]
        except:
            div = []
        pdiv = []

        for k in index_kahan_div:
            pdiv.append(kdiv(bAr[k], Anu[k], num_func[k]))

        try:
            result = intersection(div[0], V[_nu])
            for el in div[1:]:
                result = intersection(result, el)
        except:
            result = intersection(pdiv[0], V[_nu])

        for el in pdiv:
            result = intersection(result, el)

        if 2 in num_func:
            # if np.isnan(result.b).all():
            if np.isnan(np.array(result.b, dtype=np.float64)).all():
                return infinity
            elif np.isnan(np.array(result.b, dtype=np.float64)).any():
                return np.max(result.a)
            else:
                return np.min(result.a)
        else:
            if np.isnan(np.array(result.b, dtype=np.float64)):
                return infinity
            else:
                return result.a


    n, m = A.shape
    midA = A.mid
    radA = A.rad

    def StartBar(A, b):
        if n > m:
            try:
                V = Rohn(A, b)
            except:
                V = Interval([-10**14]*m, [10**14]*m, sortQ=False)
        else:
            try:
                V = Gauss(A, b)
            except:
                V = Interval([-10**14]*m, [10**14]*m, sortQ=False)

        return V

    A_ub = np.zeros((2*(n+m)+m, m))
    A_ub[2*(n+m) : ] = -np.eye(m)

    b_ub = np.zeros(2*(n+m)+m)
    p = np.zeros(2*m)

    def ExactOmega(midA, radA, b, Q):
        S = np.eye(m) * np.insert(np.sign(Q.a), _nu, np.sign(V[_nu].a))
        midAS = midA @ S

        p[:m] = np.insert(-Q.a, _nu, 1e15)
        p[m:] = np.insert(Q.b, _nu, 1e15)

        c = np.zeros(m, dtype='float64')
        c[_nu] = S[_nu, _nu]

        A_ub[ : n] = midAS - radA
        A_ub[n : 2*n] = -midAS - radA
        A_ub[2*n : 2*n+m] = -S
        A_ub[2*n+m : 2*(n+m)] = S

        b_ub[ : n] = b.b
        b_ub[n : 2*n] = -b.a
        b_ub[2*n : 2*(n+m)] = p

        try:
            result = solve_lp(c, A_ub, b_ub)
            return result @ c
        except:
            pass

        S[_nu, _nu] = np.sign(V[_nu].b)
        midAS = midA @ S

        c[_nu] = S[_nu, _nu]
        A_ub[ : n] = midAS - radA
        A_ub[n : 2*n] = -midAS - radA
        A_ub[2*n : 2*n+m] = -S
        A_ub[2*n+m : 2*(n+m)] = S

        try:
            result = solve_lp(c, A_ub, b_ub)
            return result @ c
        except:
            return 10**15


    def algo(_nu):
        WorkListA = A.copy; del(WorkListA[:, _nu])
        WorkListb = b.copy

        Q = V.copy;
        q, omega = Q[_nu].a, Q[_nu].b
        del(Q[_nu])

        Anu = A[:, _nu]
        index_classic_div = np.where(Anu.a*Anu.b > 0)[0]
        index_kahan_div = np.array([k for k in range(len(Anu.a)) if not (k in index_classic_div)])

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

        item = 0
        nit = 0
        while np.amax(Q.wid) >= tol and nit <= maxiter:
            k = np.argmax(Q.wid)
            Q1 = L[item][0].copy
            Q2 = L[item][0].copy

            if Q[k].a:
                if -2 < Q[k].b / Q[k].a < -1/2:
                    Q1[k], Q2[k] = Interval(Q[k].a, 0, sortQ=False), Interval(0, Q[k].b, sortQ=False)
                else:
                    Q1[k], Q2[k] = Interval(Q[k].a, Q[k].mid, sortQ=False), Interval(Q[k].mid, Q[k].b, sortQ=False)
            else:
                Q1[k], Q2[k] = Interval(Q[k].a, Q[k].mid, sortQ=False), Interval(Q[k].mid, Q[k].b, sortQ=False)

            del L[item]

            matmul = True
            if 0 in Q1:
                QA = Q[k] * WorkListA[:, k]
                Ar = Ar + QA.opp
                matmul = False
                Ar1 = Ar + Q1[k] * WorkListA[:, k]
                bAr = WorkListb - Ar1
                q1 = Omega(bAr, _nu, Anu, index_classic_div, index_kahan_div, num_func)

                if q1 < omega:
                    newcol = (Q1, q1, Ar1)
                    bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
                    L.insert(bslindex, newcol)
                    eta1 = infinity
                else:
                    eta1 = infinity
            else:
                q1 = ExactOmega(midA, radA, b, Q1)

                if q1 < omega:
                    newcol = (Q1, q1)
                    bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
                    L.insert(bslindex, newcol)
                    eta1 = q1
                else:
                    eta1 = infinity

            if 0 in Q2:
                if matmul:
                    QA = Q[k] * WorkListA[:, k]
                    Ar = Ar + QA.opp

                Ar2 = Ar + Q2[k] * WorkListA[:, k]
                bAr = WorkListb - Ar2
                q2 = Omega(bAr, _nu, Anu, index_classic_div, index_kahan_div, num_func)

                if q2 < omega:
                    newcol = (Q2, q2, Ar2)
                    bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
                    L.insert(bslindex, newcol)
                    eta2 = infinity
                else:
                    eta2 = infinity
            else:
                q2 = ExactOmega(midA, radA, b, Q2)

                if q2 < omega:
                    newcol = (Q2, q2)
                    bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
                    L.insert(bslindex, newcol)
                    eta2 = q2
                else:
                    eta2 = infinity

            eta = eta1 if eta1 <= eta2 else eta2
            if omega > eta:
                omega = eta
                L = [l for l in L if l[1] <= omega]

            item = 0
            while True:
                try:
                    Q, q, Ar = L[item]
                    break
                except IndexError:
                    return L[0][1]
                except:
                    item += 1

            nit += 1

        return L[0][1]

    inf = []
    sup = []
    if nu is None:
        for endint in [1, -1]:
            b = endint * b.copy
            V = StartBar(A, b)
            for _nu in range(A.shape[1]):
                if endint == -1:
                    sup.append(endint * algo(_nu))
                else:
                    inf.append(algo(_nu))
    else:
        _nu = nu
        for endint in [1, -1]:
            b = endint * b.copy
            V = StartBar(A, b)
            if endint == -1:
                sup.append(endint * algo(_nu))
            else:
                inf.append(algo(_nu))

    return Interval(inf, sup, sortQ=False)
