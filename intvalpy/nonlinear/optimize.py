from bisect import bisect_left
import numpy as np

from intvalpy.RealInterval import Interval
from intvalpy.utils import asinterval


class KeyWrapper:
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)


def globopt(func, x0, tol=1e-12, maxiter=2000):
    Y = x0.copy
    y = func(Y).a
    L = [(Y, y)]

    nit = 0
    while func(Y).wid >= tol and nit <= maxiter:
        l = np.argmax(Y.wid)
        Y1 = L[0][0].copy
        Y2 = L[0][0].copy
        Y1[l], Y2[l] = Interval(Y[l].a, Y[l].mid, sortQ=False), Interval(Y[l].mid, Y[l].b, sortQ=False)

        v1, v2 = func(Y1).a, func(Y2).a
        del L[0]

        newcol = (Y1, v1)
        bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
        L.insert(bslindex, newcol)

        newcol = (Y2, v2)
        bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
        L.insert(bslindex, newcol)
        Y = L[0][0]
        nit += 1

    return L[0][0], func(L[0][0])


# def outliers(func, a, b, grad, functional='tol', weight=None, x0=None, tol=1e-12, maxiter=2000, method='standard deviations'):
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
#     WorkLista = asinterval(a).copy
#     WorkListb = asinterval(b).copy
#
#     if functional == 'tol':
#         _, _, _, tt = __tolsolvty(func, grad, a, b, weight=weight, x0=x0, maxiter=maxiter,
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
#     index = np.delete(np.arange(WorkLista.shape[0]), outliers_index)
#     WorkLista = WorkLista[index]
#     WorkListb = WorkListb[index]
#
#     return WorkLista, WorkListb, outliers_index, tt
