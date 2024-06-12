import numpy as np

import cvxopt
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}

from ..kernel.real_intervals import Interval
from ..kernel.utils import infinity
from .Tol import Tol


def _Rohn_Tol(A, b, nu=None):

    def solve_lp(c, A_ub, b_ub):
        asmatrix = lambda A: A if isinstance(A, cvxopt.matrix) else cvxopt.matrix(A)
        args = [asmatrix(c), asmatrix(A_ub), asmatrix(b_ub)]

        result = cvxopt.solvers.lp(*args, solver='glpk')
        if 'optimal' not in result['status']:
            raise Exception("Неизвестная ошибка.")
        return np.array(result['x']).reshape(c.shape[0])

    def algo(A, b, nu):
        A_ub = np.zeros((2*(n+m), 2*m), dtype=np.float64)
        b_ub = np.zeros(2*(n+m), dtype=np.float64)

        A_ub[:n, :m] = A.b
        A_ub[:n, m:] = -A.a

        A_ub[n:2*n, :m] = -A.a
        A_ub[n:2*n, m:] = A.b
        A_ub[2*n:] = -np.eye(2*m)

        b_ub[:n] = b.b
        b_ub[n:2*n] = -b.a

        c = np.zeros(2*m, dtype=np.float64)
        c[nu], c[nu+m] = 1, -1

        inf = solve_lp(c, A_ub, b_ub) @ c
        sup = -(solve_lp(-c, A_ub, b_ub) @ -c)

        return inf, sup

    n, m = A.shape
    _, tolval, _, _, _ = Tol.maximize(A, b)
    if tolval < 0:
        raise Exception('Допусковое множество решений пусто!')

    _inf, _sup = [], []
    if nu is None:
        for k in range(m):
            inf, sup = algo(A, b, k)
            _inf.append(inf)
            _sup.append(sup)
    else:
        inf, sup = algo(A, b, nu)
        _inf.append(inf)
        _sup.append(sup)

    return Interval(_inf, _sup, sortQ=False)


def _Rohn_Uni(A, b, tol=1e-12, maxiter=2000):

    Amid = np.array(A.mid, dtype=np.float64)
    Ac_plus = np.linalg.inv(Amid.T @ Amid) @ Amid.T
    A = Ac_plus @ A.copy()
    b = Ac_plus @ b.copy()

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


def Rohn(A, b, tol=1e-12, maxiter=2000, consistency='uni'):
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

                consistency: str
                    Параметр указывает какое множество решений (объединённое или
                    допусковое) будет выведено в ответе.

    Returns:
                out: Interval
                    Возвращается интервальный вектор решений.
    """
    if consistency == 'uni':
        return _Rohn_Uni(A, b, tol=tol, maxiter=maxiter)
    elif consistency == 'tol':
        return _Rohn_Tol(A, b, nu=None)
    else:
        raise Exception('Нахождение данного множества решений не предусмотрено.')