import numpy as np

from intvalpy.utils import asinterval, zeros, sgn
from intvalpy.RealInterval import Interval, INTERVAL_CLASSES

from bisect import bisect_left


class KeyWrapper:
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)


def __tolsolvty(func, grad, a, b, weight=None, x0=None,
                tol_f=1e-12, tol_x=1e-12, tol_g=1e-12, maxiter=2000, stepwise=float('inf')):

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
    tt = None

    def calcfg(x):
        index = x >= 0
        infsup = bc - func(a, x)

        tt = weight * (br - np.maximum(np.abs(infsup.a), np.abs(infsup.b)))
        mc = np.argmin(tt)
        if n == 1:
            _grad = asinterval(grad[0](a[mc], x))
        else:
            _grad = asinterval([g(a[mc], x) for g in grad])

        if -infsup[mc].a <= infsup[mc].b:
            dd = weight[mc] * (_grad.a * index + _grad.b * (~index))
        else:
            dd = -weight[mc] * (_grad.b * index + _grad.a * (~index))

        return tt[mc], dd, tt


    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0

    B = np.eye(n)
    vf = np.zeros(nsims) + float('inf')
    w = 1./alpha - 1

    f, g0, _ = calcfg(x)
    ff = f;  xx = x
    cal = 1;  ncals = 1
    ccode = False

    for nit in range(1, int(maxiter)):
        if nit % stepwise == 0:
            print('nit: ', nit)
            print('x: ', xx)
            print('tol: ', ff)
            print('+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+\n\n')

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
            f, g1, tt = calcfg(x)

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

    return ccode, xx, ff, tt


def _tol_tsopt(model, a, b, grad, weight, x0, tol, maxiter, stepwise):
    return __tolsolvty(model, grad, a, b, weight=weight, x0=x0, maxiter=maxiter, stepwise=stepwise,
                       tol_f=tol, tol_x=tol, tol_g=tol)[:-1]


def _tol_iopt(model, a, b, grad, x0, tol, maxiter, stepwise):
    def tol_globopt(func, x0, grad, tol, maxiter):
        def insert(zeroS, Y, v, vmag, bmm, nit_mon):
            if zeroS:
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

            Y = L[0][0]
            nit += 1

        return L[0][0], func(L[0][0]), nit


    br = b.rad
    bm = b.mid

    minus_tol_interval = lambda x: -min(br - abs(bm - model(a, x)))
    xx, ff, nit = tol_globopt(minus_tol_interval, x0, grad, tol, maxiter)
    success = nit <= maxiter

    return success, xx, -ff


def Tol(model, a, b, x=None, maxQ=False, grad=None, weight=None, x0=None, tol=1e-12, maxiter=2000, stepwise=float('inf')):

    if maxQ:
        if isinstance(x0, INTERVAL_CLASSES):
            return _tol_iopt(model, a, b, grad, x0, tol, maxiter, stepwise)
        else:
            return _tol_tsopt(model, a, b, grad, weight, x0, tol, maxiter, stepwise)

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
