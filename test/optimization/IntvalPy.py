from bisect import bisect_left
import numpy as np

from time import perf_counter as pc

import intvalpy as ip
from intvalpy.RealInterval import Interval
ip.precision.increasedPrecisionQ = False


class KeyWrapper:
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)


def intvalpy_globopt(func, x0, tol=1e-12, maxiter=2000):
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


def mag(x):
    if x.a*x.b < 0:
        return ip.Interval(0, max(np.abs(x.a), np.abs(x.b)))
    else:
        return ip.Interval(np.abs(x.a), np.abs(x.b))

def branin(x):
    return (x[1] - 5.1/(4*np.pi**2)*x[0]**2 + 5/np.pi*x[0] - 6)**2 + 10*(1 - 1/(8*np.pi))*np.cos(x[0]) + 10
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def trekkani(x):
    return x[0]**4 + 4*x[0]**3 + 4*x[0]**2 + x[1]**2
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def rosenbrock(x):
    return sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def dixonprice(x):
    return (x[0]-1)**2 + sum((k+1)*(2*x[k]**2-x[k-1])**2 for k in range(1, len(x)))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def levy(x):
    z = 1 + (x - 1) / 4
    t1 = np.sin( np.pi * z[0] )**2
    t2 = sum(((x - 1) ** 2 * (1 + 10 * np.sin(np.pi * x + 1) ** 2))[:-1])
    t3 = (z[-1] - 1) ** 2 * (1 + np.sin(2*np.pi * z[-1]) ** 2)
    return t1 + t2 + t3
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def rastrigin(x):
    return 10*len(x) + sum(x**2 - 10 * np.cos(2 * np.pi * x))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def bohachevsky(x):
    return x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) -0.4*np.cos(4*np.pi*x[1]) + 0.7
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def beale(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def booth(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def colville(x):
    return 100*(x[0]**2 - x[1])**2 + (x[0] - 1)**2 + (x[2] - 1)**2 + 90*(x[2]**2 - x[3])**2 + \
           10.1*((x[1] - 1)**2 + (x[3] - 1)**2) + 19.8*(x[1] - 1)*(x[3] - 1)
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def dejoung(x):
    return x[0]**2 + x[1]**2 + x[2]**2
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def easom(x):
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-(x[0] - np.pi)**2 - (x[1] - np.pi)**2)
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def hartmann(x):
    alpha = np.array([1, 1.2, 3, 3.2])
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]])
    P = 10**(-4) * np.array([[6890, 1170, 2673],
                             [4699, 4387, 7470],
                             [1091, 8732, 5547],
                             [381, 5743, 8828]])

    return -alpha @ np.exp(-sum((A * (x.T - P) ** 2).T))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def matyas(x):
    return 0.26*(x[0]**2 + x[1]**2) - 0.48*x[0]*x[1]
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def eggholder(x):
    return -(x[1] + 47) * np.sin(np.sqrt(mag(x[0]/2 + (x[1] + 47)))) - \
            x[0]*np.sin(np.sqrt(mag(x[0] - (x[1]+47))))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def schaffer(x):
    return 0.5 + (np.cos(np.sin(mag(x[0]**2 - x[1]**2)))**2 - 0.5) / (1 + 0.001*(x[0]**2 + x[1]**2))**2
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def holder(x):
    return -mag(np.sin(x[0])*np.cos(x[1]) * np.exp(mag(1 - np.sqrt(x[0]**2 + x[1]**2) / np.pi)))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

def levy13(x):
    return np.sin(3*np.pi*x[0])**2 + (x[0] - 1)**2 * (1 + np.sin(3*np.pi*x[1])**2) + \
            (x[1] - 1)**2 * (1 + np.sin(2*np.pi*x[1]))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+


# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+


test_opt = [
    [branin,        ip.Interval([[-5, 10], [0, 15]])],
    [bohachevsky,   ip.Interval([[-50, 100] for _ in range(2)])],
    [beale,         ip.Interval([[-4.5, 4.5] for _ in range(2)])],
    [booth,         ip.Interval([[-10, 10] for _ in range(2)])],
    [dejoung,       ip.Interval([[-2.56, 5.12] for _ in range(3)])],
    [easom,         ip.Interval([[-100, 100] for _ in range(2)])],
    [eggholder,     ip.Interval([[-512, 512] for _ in range(2)])],
    [rastrigin,     ip.Interval([[-2.56, 5.12] for _ in range(4)])],
    [schaffer,      ip.Interval([[-100, 100] for _ in range(2)])],
    [levy13,        ip.Interval([[-10, 10] for _ in range(2)])],
    [hartmann,      ip.Interval([[0, 1] for _ in range(3)])],
    [trekkani,      ip.Interval([[-5, 5], [-5, 5]])],
    [rosenbrock,    ip.Interval([[-5, 10] for _ in range(2)])],
    [dixonprice,    ip.Interval([[-10, 10] for _ in range(2)])],
    [levy,          ip.Interval([[-10, 10] for _ in range(2)])],
    [matyas,        ip.Interval([[-5, 10] for _ in range(2)])],
    [colville,      ip.Interval([[-10, 10] for _ in range(4)])],
    [holder,        ip.Interval([[-10, 10] for _ in range(2)])]
]


n = 50
for func, x0 in test_opt:

    t0 = pc()
    for _ in range(n):
        globopt = intvalpy_globopt(func, x0, tol=1e-14, maxiter=10000)
    print('function: ', func.__name__)
    print('x: ', globopt[0])
    print('func(x): ', globopt[1])
    print('tol: ', func(globopt[0]).wid)
    print('mean time: ', (pc() - t0) / n)
    print('+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+\n\n')
