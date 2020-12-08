import numpy as np
from scipy.optimize import minimize

from .MyClass import Interval


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
