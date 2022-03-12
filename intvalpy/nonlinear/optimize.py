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


def __tolsolvty(func, grad, a, b, weight=None, x0=None,
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


def Uni(func, a, b, x=None, maxQ=False, grad=None, weight=None, x0=None, tol=1e-12, maxiter=2000):

    br = b.rad
    bm = b.mid

    def __uni(x):
        data = func(a, x)
        return np.min(br - (bm - data).mig)
    __minus_uni = lambda x: -__uni(x)

    if maxQ:
        if x0 is None:
            _, x0, _, _ = __tolsolvty(func, grad, a, b, weight=weight, x0=None, maxiter=maxiter,
                               tol_f=1e-8, tol_x=1e-8, tol_g=1e-8)

        from scipy.optimize import minimize
        maximize = minimize(__minus_uni, x0, method='Nelder-Mead', tol=tol,
                            options={'maxiter': maxiter})
        return maximize.success, maximize.x, -maximize.fun

    else:
        if x is None:
            x = np.zeros(len(a))
        return __uni(x)



def _tol_tsopt(model, a, b, grad, weight=None, x0=None, tol=1e-12, maxiter=2000):
    return __tolsolvty(model, grad, a, b, weight=weight, x0=x0, maxiter=maxiter,
                       tol_f=tol, tol_x=tol, tol_g=tol)[:-1]


def _tol_iopt(model, a, b, grad, x0, tol=1e-12, maxiter=2000):
    return None, None, None


def Tol(func, a, b, x=None, maxQ=False, grad=None, weight=None, x0=None, tol=1e-12, maxiter=2000):

    if maxQ:
        return __tolsolvty(func, grad, a, b, weight=weight, x0=x0, maxiter=maxiter,
                           tol_f=tol, tol_x=tol, tol_g=tol)[:-1]

    else:
        br = b.rad
        bm = b.mid

        def __tol(x):
            data = func(a, x)
            return np.min(br - (bm - data).mag)

        if x is None:
            x = np.zeros(len(a))

        return __tol(x)


def outliers(func, a, b, grad, functional='tol', weight=None, x0=None, tol=1e-12, maxiter=2000, method='standard deviations'):

    def interquartile(data):
        q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
        iqr = q75 - q25
        cut_off = iqr * 1.5

        lower, upper = q25 - cut_off, q75 + cut_off
        return np.argwhere((data < lower) | (data > upper)).flatten()

    def standard_deviations(data):
        # Set upper and lower limit to 3 standard deviation
        std, mean = np.std(data), np.mean(data)
        cut_off = std * 3

        lower, upper = mean - cut_off, mean + cut_off
        return np.argwhere((data < lower) | (data > upper)).flatten()

    WorkLista = asinterval(a).copy
    WorkListb = asinterval(b).copy

    if functional == 'tol':
        _, _, _, tt = __tolsolvty(func, grad, a, b, weight=weight, x0=x0, maxiter=maxiter,
                                  tol_f=tol, tol_x=tol, tol_g=tol)
    else:
        Exception('Данный функционал не предусмотрен.')

    if method == 'standard deviations':
        outliers_index = standard_deviations(tt)
    elif method == 'interquartile':
        outliers_index = interquartile(tt)
    else:
        Exception('Данный метод не предусмотрен.')

    index = np.delete(np.arange(WorkLista.shape[0]), outliers_index)
    WorkLista = WorkLista[index]
    WorkListb = WorkListb[index]

    return WorkLista, WorkListb, outliers_index, tt
