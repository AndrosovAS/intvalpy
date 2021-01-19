import numpy as np
from scipy.optimize import minimize

from intvalpy.MyClass import Interval
from intvalpy.intoper import zeros


def Uni(A, b, x=None, maxQ=False, x0=None, tol=1e-12, maxiter=1e3):
    """
    Вычисление распознающего функционала Uni.
    В случае, если maxQ=True то находится максимум функционала.

    Parameters:
                A: Interval
                    Матрица ИСЛАУ.

                b: Interval
                    Вектор правой части ИСЛАУ.

    Optional Parameters:
                x: float, array_like
                    Точка в которой вычисляется распознающий функционал.
                    По умолчанию x равен массиву из нулей.

                maxQ: bool
                    Если значение параметра равно True, то производится
                    максимизация функционала.

                x0: float, array_like
                    Первоначальная догадка.

                tol: float
                    Погрешность для прекращения оптимизационного процесса.

                maxiter: int
                    Максимальное количество итераций.

    Returns:
                out: float, tuple
                    Возвращается значение распознающего функционала в точке x.
                    В случае, если maxQ=True, то возвращается кортеж, где
                    первый элемент -- корректность завершения оптимизации,
                    второй элемент -- точка оптимума,
                    третий элемент -- значение функции в этой точке.
    """

    __uni = lambda x: min(b.rad - (b.mid - A @ x).mig)
    __minus_uni = lambda x: -__uni(x)

    if maxQ==False:
        if x is None:
            x = np.zeros(A.shape[1])
        return __uni(x)
    else:
        from scipy.optimize import minimize

        if x0 is None:
            x0 = np.zeros(A.shape[1])+1
        maximize = minimize(__minus_uni, x0, method='Nelder-Mead', tol=tol,
                            options={'maxiter': maxiter})

        return maximize.success, maximize.x, -maximize.fun


def Tol(A, b, x=None, maxQ=False, x0=None, tol=1e-12, maxiter=1e3):
    """
    Вычисление распознающего функционала Tol.
    В случае, если maxQ=True то находится максимум функционала.

    Parameters:
                A: Interval
                    Матрица ИСЛАУ.

                b: Interval
                    Вектор правой части ИСЛАУ.

    Optional Parameters:
                x: float, array_like
                    Точка в которой вычисляется распознающий функционал.
                    По умолчанию x равен массиву из нулей.

                maxQ: bool
                    Если значение параметра равно True, то производится
                    максимизация функционала.

                x0: float, array_like
                    Первоначальная догадка.

                tol: float
                    Погрешность для прекращения оптимизационного процесса.

                maxiter: int
                    Максимальное количество итераций.

    Returns:
                out: float, tuple
                    Возвращается значение распознающего функционала в точке x.
                    В случае, если maxQ=True, то возвращается кортеж, где
                    первый элемент -- корректность завершения оптимизации,
                    второй элемент -- точка оптимума,
                    третий элемент -- значение функции в этой точке.
    """

    __tol = lambda x: min(b.rad - abs(b.mid - A @ x))
    __minus_tol = lambda x: -__tol(x)

    if maxQ==False:
        if x is None:
            x = np.zeros(A.shape[1])
        return __tol(x)
    else:
        from scipy.optimize import minimize

        if x0 is None:
            x0 = np.zeros(A.shape[1])+1
        maximize = minimize(__minus_tol, x0, method='Nelder-Mead', tol=tol,
                            options={'maxiter': maxiter})

        return maximize.success, maximize.x, -maximize.fun


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
