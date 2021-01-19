from bisect import bisect_left
import numpy as np

from intvalpy.MyClass import Interval


class KeyWrapper:
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)


def globopt(func, x0, tol=1e-12, maxiter=10**3):
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
