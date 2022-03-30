import numpy as np
from intvalpy.RealInterval import Interval
from intvalpy.utils import asinterval, infinity


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
    cond = infinity
    angle_A = np.zeros(A.shape, dtype='float64')
    for _ in range(N):
        for k in range(A.shape[0]):
            for l in range(A.shape[1]):
                angle_A[k, l] = np.random.choice([_inf[k, l], _sup[k, l]])
        tmp = np.linalg.cond(angle_A)
        cond = tmp if tmp<cond else cond

    return np.sqrt(A.shape[1]) * _max * cond * (np.linalg.norm(_arg_max, ord=2)/np.sqrt(sum(b.mag**2)))

# def outliers(A, b, functional='uni', x0=None, tol=1e-12, maxiter=2000, method='standard deviations'):
#
#     def interquartile(data):
#         q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
#         iqr = q75 - q25
#         cut_off = iqr * 1.5
#
#         lower, upper = q25 - cut_off, q75 + cut_off
#         return np.argwhere((data < lower) | (data > upper)).flatten()
#
#     def standard_deviations(data):
#         # Set upper and lower limit to 3 standard deviation
#         std, mean = np.std(data), np.mean(data)
#         cut_off = std * 3
#
#         lower, upper = mean - cut_off, mean + cut_off
#         return np.argwhere((data < lower) | (data > upper)).flatten()
#
#     WorkListA = asinterval(A).copy
#     WorkListb = asinterval(b).copy
#
#     if functional == 'uni':
#         _, x, _ = Uni(A, b, maxQ=True, x0=x0, tol=tol, maxiter=maxiter)
#         tt = WorkListb.rad - (WorkListb.mid - WorkListA @ x).mig
#
#     elif functional == 'tol':
#         _, _, _, tt = __tolsolvty(WorkListA.a, WorkListA.b, WorkListb.a, WorkListb.b, weight=x0, maxiter=maxiter,
#                                   tol_f=tol, tol_x=tol, tol_g=tol)
#     else:
#         Exception('Данный функционал не предусмотрен.')
#
#     if method == 'standard deviations':
#         outliers_index = standard_deviations(tt)
#     elif method == 'interquartile':
#         outliers_index = interquartile(tt)
#     else:
#         Exception('Данный метод не предусмотрен.')
#
#     index = np.delete(np.arange(WorkListA.shape[0]), outliers_index)
#     WorkListA = WorkListA[index]
#     WorkListb = WorkListb[index]
#
#     return WorkListA, WorkListb, outliers_index, tt
