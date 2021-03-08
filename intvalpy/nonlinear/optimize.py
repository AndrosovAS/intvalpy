from bisect import bisect_left
import numpy as np

from intvalpy.RealInterval import Interval


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


def __tolsolvty(func, grad, a, b, weight=None, x0=None, \
                tol_f=1e-12, tol_x=1e-12, tol_g=1e-12, maxiter=2000):

    nsims = 30
    alpha = 2.3
    hs = 1
    nh = 3
    q1 = 0.9
    q2 = 1.1

    if grad is None:
        raise Exception('Необходимо задать вектор градиент!')

    m = len(a)
    n = len(grad)
    if weight is None:
        weight = np.ones(m)

    bc = b.mid
    br = b.rad

    def calcfg(x):
        index = x >= 0
        infsup = bc - func(a, x)

        tt = weight * (br - np.maximum(np.abs(infsup.a), np.abs(infsup.b)))
        mc = np.argmin(tt)
        _grad = ip.asinterval([g(a[mc], x) for g in grad])

        if -infsup[mc].a <= infsup[mc].b:
            dd = weight[mc] * (_grad.a * index + _grad.b * (~index))
        else:
            dd = -weight[mc] * (_grad.b * index + _grad.a * (~index))

        return tt[mc], dd


    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0

    B = np.eye(n)
    vf = np.zeros(nsims) + float('inf')
    w = 1./alpha - 1

    f, g0 = calcfg(x)
    ff = f ;  xx = x;
    cal = 1;  ncals = 1;

    for nit in range(int(maxiter)):
        vf[nsims-1] = ff
        if np.linalg.norm(g0) < tol_g:
            ccode = True
            break

        g1 = B.T @ g0
        g = B @ (g1/np.linalg.norm(g1))
        normg = np.linalg.norm(g)

        r = 1
        cal = 0
        deltax = 0

        while r > 0 and cal <= 500:
            cal += 1
            x = x + hs * g
            deltax = deltax + hs * normg
            f, g1 = calcfg(x)

            if f > ff:
                ff = f
                xx = x

            if np.mod(cal, nh) == 0:
                hs = hs * q2
            r = g @ g1

        if cal > 500:
            ccode = False
            break

        if cal == 1:
            hs = hs * q1

        ncals = ncals + cal
        if deltax < tol_x:
            ccode = True
            break

        dg = B.T @ (g1 - g0)
        xi = dg / np.linalg.norm(dg)
        B = B + w * np.outer((B @ xi), xi)
        g0 = g1

        vf = np.roll(vf, 1)
        vf[0] = abs(ff - vf[0])

        if abs(ff) > 1:
            deltaf = np.sum(vf)/abs(ff)
        else:
            deltaf = np.sum(vf)

        if deltaf < tol_f:
            ccode = True
            break
        ccode = False

    return ccode, xx, ff


def Tol(func, a, b, x=None, maxQ=False, grad=None, weight=None, x0=None, tol=1e-12, maxiter=2000):

    if maxQ:
        return __tolsolvty(func, grad, a, b, weight=weight, x0=x0, maxiter=maxiter, \
                           tol_f=tol, tol_x=tol, tol_g=tol)

    else:
        br = b.rad
        bm = b.mid

        def __tol(x):
            data = func(a, x)
            return np.min(br - abs(bm - data))

        if x is None:
            x = np.zeros(len(a))

        return __tol(x)
