import numpy as np
from interval import interval, imath

from bisect import bisect_left
from time import perf_counter as pc


class KeyWrapper:
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)


def pyinterval_globopt(func, x0, tol=1e-12, maxiter=2000):

    wid = lambda a: a[0].sup - a[0].inf

    Y = x0[:]
    y = func(Y)[0].inf

    L = [(Y, y)]

    nit = 0
    while wid(func(Y)) >= tol and nit <= maxiter:
        l = np.argmax(np.array([wid(yy) for yy in Y]))
        Y1 = L[0][0][:]
        Y2 = L[0][0][:]
        Y1[l], Y2[l] = interval[Y[l][0].inf, Y[l].midpoint], interval[Y[l].midpoint, Y[l][0].sup]

        v1, v2 = func(Y1)[0].inf, func(Y2)[0].inf
        del L[0]

        newcol = (Y1, v1)
        bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
        L.insert(bslindex, newcol)

        newcol = (Y2, v2)
        bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
        L.insert(bslindex, newcol)
        Y = L[0][0][:]

        nit += 1

    return L[0][0], func(L[0][0]), wid(func(Y)), nit


def mag(x):
    if x[0].inf*x[0].sup < 0:
        return interval[0, max(np.abs(x[0].inf), np.abs(x[0].sup))]
    else:
        abs_xinf = np.abs(x[0].inf)
        abs_sxup = np.abs(x[0].sup)

        if abs_xinf < abs_sxup:
            return interval[abs_xinf, abs_sxup]
        else:
            return interval[abs_sxup, abs_xinf]

def branin(x):
    return (x[1] - 5.1/(4*np.pi**2)*x[0]**2 + 5/np.pi*x[0] - 6)**2 + \
            10*(1 - 1/(8*np.pi))*imath.cos(x[0]) + 10
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def bohachevsky(x):
    return x[0]**2 + 2*x[1]**2 - 0.3*imath.cos(3*np.pi*x[0]) -0.4*imath.cos(4*np.pi*x[1]) + 0.7
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def beale(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def booth(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def dejoung(x):
    return x[0]**2 + x[1]**2 + x[2]**2
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def easom(x):
    return -imath.cos(x[0]) * imath.cos(x[1]) * imath.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def eggholder(x):
    return -(x[1] + 47) * imath.sin(imath.sqrt(mag(x[0]/2 + (x[1] + 47)))) - \
            x[0]*imath.sin(imath.sqrt(mag(x[0] - (x[1]+47))))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def rastrigin(x):
    return 10*len(x) + sum([xx**2 - 10 * imath.cos(2 * np.pi * xx) for xx in x])
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def schaffer(x):
    return 0.5 + (imath.cos(imath.sin(mag(x[0]**2 - x[1]**2)))**2 - 0.5) / (1 + 0.001*(x[0]**2 + x[1]**2))**2
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def levy13(x):
    return imath.sin(3*np.pi*x[0])**2 + (x[0] - 1)**2 * (1 + imath.sin(3*np.pi*x[1])**2) + \
            (x[1] - 1)**2 * (1 + imath.sin(2*np.pi*x[1]))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+


# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+


test_opt = [
    [branin,          [interval[-5, 10], interval[0, 15]]],
    [bohachevsky,     [interval[-50, 100] for _ in range(2)]],
    [beale,           [interval[-4.5, 4.5] for _ in range(2)]],
    [booth,           [interval[-10, 10] for _ in range(2)]],
    [dejoung,         [interval[-2.56, 5.12] for _ in range(3)]],
    [easom,           [interval[-100, 100] for _ in range(2)]],
    [eggholder,       [interval[-512, 512] for _ in range(2)]],
    [rastrigin,       [interval[-2.56, 5.12] for _ in range(4)]],
    [schaffer,        [interval[-100, 100] for _ in range(2)]],
    [levy13,          [interval[-10, 10] for _ in range(2)]]
]

n = 50
for func, x0 in test_opt:

    t0 = pc()
    for _ in range(n):
        globopt = pyinterval_globopt(func, x0, tol=1e-14, maxiter=10000)
    print('function: ', func.__name__)
    print('x: ', globopt[0])
    print('func(x): ', globopt[1])
    print('tol: ', globopt[2])
    print('mean time: ', (pc() - t0) / n)
    print('+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+\n\n')
