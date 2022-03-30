import numpy as np

from intvalpy.utils import asinterval, zeros, sgn
from intvalpy.RealInterval import Interval, INTERVAL_CLASSES
from intvalpy.ralgb5 import ralgb5

from bisect import bisect_left


class KeyWrapper:
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)


def recfunsolvty(model, grad, a, b, x0, consistency='uni', weight=None, tol=1e-12, maxiter=2000, alpha=2.3, nsims=30, h0=1, nh=3, q1=0.9, q2=1.1, tolx=1e-12, tolg=1e-12, tolf=1e-12):

    n, m = len(a), len(grad)
    if weight is None:
        weight = np.ones(n)
    x0 = np.copy(x0)

    bm = b.mid
    br = b.rad

    def mig(inf, sup):
        if inf*sup <= 0:
            return 0.0
        else:
            return min(abs(inf), abs(sup))

    if consistency=='uni':
        functional = lambda infs, sups: weight * (br - np.vectorize(mig)(infs, sups))
    else:
        functional = lambda infs, sups: weight * (br - np.maximum(np.abs(infs), np.abs(sups)))


    def calcfg(x):
        index = x >= 0
        infsup = bm - model(a, x)

        tt = functional(infsup.a, infsup.b)
        mc = np.argmin(tt)
        gg = asinterval([g(a[mc], x) for g in grad])

        if -infsup[mc].a <= infsup[mc].b:
            dd = weight[mc] * (gg.a * index + gg.b * (~index))
        else:
            dd = -weight[mc] * (gg.b * index + gg.a * (~index))

        return -tt[mc], -dd

    xr, fr, nit, ncalls, ccode = ralgb5(calcfg, x0, tol=tol, maxiter=maxiter, alpha=alpha, nsims=nsims, h0=h0, nh=nh, q1=q1, q2=q2, tolx=tolx, tolg=tolg, tolf=tolf)
    return xr, -fr, nit, ncalls, ccode


def _tol_tsopt(model, grad, a, b, x0, weight=None, tol=1e-12, maxiter=2000, alpha=2.3, nsims=30, h0=1, nh=3, q1=0.9, q2=1.1, tolx=1e-12, tolg=1e-12, tolf=1e-12):

    xr, fr, nit, ncalls, ccode = recfunsolvty(model, grad, a, b, x0, consistency='tol', weight=weight, tol=tol, maxiter=maxiter,
                                                alpha=alpha, nsims=nsims, h0=h0, nh=nh, q1=q1, q2=q2, tolx=tolx, tolg=tolg, tolf=tolf)
    ccode = False if (ccode==4 or ccode==5) else True
    return ccode, xr, fr


def _tol_iopt(model, a, b, grad, x0, tol, maxiter, stepwise):
    def tol_globopt(func, x0, grad, tol, maxiter):
        def insert(zeroS, Y, v, vmag, bmm, nit_mon):
            if zeroS or True:
                newcol = (Y, v.a)
                bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
                L.insert(bslindex, newcol)
            else:
                g = zeros(n)
                for k in range(m):
                    zeroG = False
                    for l in range(n):
                        g[l] = -grad[k](a[l], Y) * sgn(bmm[l])
                        if 0 in g[l]:
                            zeroG = True
                            break

                    if not zeroG:
                        nit_mon += 1
                        y1, y2 = Y.copy, Y.copy
                        y1[k], y2[k] = Y[k].a, Y[k].b
                        v1, v2 = func(y1).a, func(y2).a
                        if v1 < v2:
                            Y = y1
                        else:
                            Y = y2

                v = func(Y).a
                newcol = (Y, v)
                bslindex = bisect_left(KeyWrapper(L, key=lambda c: c[1]), newcol[1])
                L.insert(bslindex, newcol)

            return Y, v, nit_mon

        Y = x0.copy
        y = func(Y).a
        L = [(Y, y)]
        n, m = len(a), len(grad)

        nit_mon = 0
        gamma = float('inf')
        nit = 1
        while func(Y).wid >= tol and nit <= maxiter:
            if nit % stepwise == 0:
                print('nit: ', nit)
                print('x: ', Y)
                print('tol: ', -func(Y))
                print('+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+\n\n')

            gg = np.array([np.max((grad[k](a, Y)).mag) * Y[k].wid for k in range(m)])
            l = np.argmax(gg)
            Y1 = L[0][0].copy
            Y2 = L[0][0].copy
            Y1[l], Y2[l] = Interval(Y[l].a, Y[l].mid, sortQ=False), Interval(Y[l].mid, Y[l].b, sortQ=False)

            del L[0]

            bmm1, bmm2 = bm - model(a, Y1), bm - model(a, Y2)
            mag1, mag2 = abs(bmm1), abs(bmm2)
            v1, v2 = -min(br - mag1), -min(br - mag2)

            zeroS1 = True if (bmm1.a <= 0).all() and (0 <= bmm1.b).all() else False
            zeroS2 = True if (bmm2.a <= 0).all() and (0 <= bmm2.b).all() else False

            Y1, v1, nit_mon = insert(zeroS1, Y1, v1, mag1, bmm1, nit_mon)
            Y2, v2, nit_mon = insert(zeroS2, Y2, v2, mag2, bmm2, nit_mon)

            gamma1 = func(Y1.mid).a
            gamma2 = func(Y2.mid).a
            if gamma1 < gamma:
                gamma = gamma1
            if gamma2 < gamma:
                gamma = gamma2

            L = [l for l in L if l[1] <= gamma]

            Y = L[0][0]
            nit += 1

        print('nit: ', nit)
        print('len(L): ', len(L))
        return L[0][0], func(L[0][0]), nit

    a = asinterval(a)
    br = b.rad
    bm = b.mid

    minus_tol_interval = lambda x: -min(br - abs(bm - model(a, x)))
    xx, ff, nit = tol_globopt(minus_tol_interval, x0, grad, tol, maxiter)
    success = nit <= maxiter

    return success, xx, -ff


def Tol(model, a, b, x=None, maxQ=False, grad=None, weight=None, x0=None, tol=1e-12, maxiter=2000, stepwise=float('inf'),
        alpha=2.3, nsims=30, h0=1, nh=3, q1=0.9, q2=1.1, tolx=1e-12, tolg=1e-12, tolf=1e-12):

    if maxQ:
        if isinstance(x0, INTERVAL_CLASSES):
            return _tol_iopt(model, a, b, grad, x0, tol, maxiter, stepwise)
        else:
            return _tol_tsopt(model, grad, a, b, x0, weight=weight, tol=tol, maxiter=maxiter,
                                alpha=alpha, nsims=nsims, h0=h0, nh=nh, q1=q1, q2=q2, tolx=tolx, tolg=tolg, tolf=tolf)
    else:
        br = b.rad
        bm = b.mid

        tol_interval = lambda x: min( br - abs(bm - model(a, x)) )
        tol_exact    = lambda x: np.min( br - (bm - model(a, x)).mag )

        if isinstance(x, INTERVAL_CLASSES):
            return tol_interval(x)
        else:
            if x is None:
                raise TypeError('It is necessary to specify at which point to calculate the recognizing functional.')
            return tol_exact(x)


def _uni_usopt(model, grad, a, b, x0, weight=None, tol=1e-12, maxiter=2000, alpha=2.3, nsims=30, h0=1, nh=3, q1=0.9, q2=1.1, tolx=1e-12, tolg=1e-12, tolf=1e-12):

    xr, fr, nit, ncalls, ccode = recfunsolvty(model, grad, a, b, x0, consistency='uni', weight=weight, tol=tol, maxiter=maxiter,
                                              alpha=alpha, nsims=nsims, h0=h0, nh=nh, q1=q1, q2=q2, tolx=tolx, tolg=tolg, tolf=tolf)
    ccode = False if (ccode==4 or ccode==5) else True
    return ccode, xr, fr

def Uni(model, a, b, x=None, maxQ=False, grad=None, weight=None, x0=None, tol=1e-12, maxiter=2000, stepwise=float('inf'),
        alpha=2.3, nsims=30, h0=1, nh=3, q1=0.9, q2=1.1, tolx=1e-12, tolg=1e-12, tolf=1e-12):

    br = b.rad
    bm = b.mid

    uni_exact = lambda x: min(br - (bm - model(a, x)).mig)

    if maxQ:
        return _uni_usopt(model, grad, a, b, x0, weight=weight, tol=tol, maxiter=maxiter,
                            alpha=alpha, nsims=nsims, h0=h0, nh=nh, q1=q1, q2=q2, tolx=tolx, tolg=tolg, tolf=tolf)
    else:
        if x is None:
            raise TypeError('It is necessary to specify at which point to calculate the recognizing functional.')
        return uni_exact(x)
