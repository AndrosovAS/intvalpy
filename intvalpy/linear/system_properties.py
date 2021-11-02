import numpy as np
from scipy.optimize import minimize

from intvalpy.RealInterval import Interval
from intvalpy.intoper import zeros, infinity


def __tolsolvty(infA, supA, infb, supb, weight=None, \
                tol_f=1e-12, tol_x=1e-12, tol_g=1e-12, maxiter=2000):

    nsims = 30
    alpha = 2.3
    hs = 1
    nh = 3
    q1 = 0.9
    q2 = 1.1

    m, n = infA.shape
    if weight is None:
        weight = np.ones(m)

    Ac = 0.5 * (infA + supA)
    Ar = 0.5 * (supA - infA)
    bc = 0.5 * (infb + supb)
    br = 0.5 * (supb - infb)

    sv = np.linalg.svd(np.array(Ac, dtype=np.float64), compute_uv=False)
    minsv, maxsv = min(sv), max(sv)

    def calcfg(x):
        index = x >= 0

        Ac_x = Ac @ x
        Ar_absx = Ar @ np.abs(x)
        infs = bc - (Ac_x + Ar_absx)
        sups = bc - (Ac_x - Ar_absx)

        tt = weight * (br - np.maximum(np.abs(infs), np.abs(sups)))
        mc = np.argmin(tt)

        if -infs[mc] <= sups[mc]:
            dd = weight[mc] * (infA[mc] * index + supA[mc] * (~index))
        else:
            dd = -weight[mc] * (supA[mc] * index + infA[mc] * (~index))

        return tt[mc], dd

    if (minsv != 0 and maxsv/minsv < 1e15):
        x = np.linalg.lstsq(np.array(Ac, dtype=np.float64), np.array(bc, dtype=np.float64), rcond=-1)[0]
    else:
        x = np.zeros(n)

    B = np.eye(n)
    vf = np.zeros(nsims) + infinity
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

    midA = A.mid
    radA = A.rad
    br = b.rad
    bm = b.mid
    def __dot(A, x):
        tmp1 = np.dot(midA, x)
        tmp2 = np.dot(radA, abs(x))
        return Interval(tmp1 - tmp2, tmp1 + tmp2, sortQ=False)
    __uni = lambda x: min(br - (bm - __dot(A, x)).mig)
    __minus_uni = lambda x: -__uni(x)

    if maxQ==False:
        if x is None:
            x = np.zeros(A.shape[1])
        return __uni(x)
    else:
        from scipy.optimize import minimize

        if x0 is None:
            # x0 = np.ones(A.shape[1])
            _, x0, _ = __tolsolvty(A.a, A.b, b.a, b.b, \
                                   tol_f=1e-8, tol_x=1e-8, tol_g=1e-8)
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

    br = b.rad
    bm = b.mid
    __tol = lambda x: np.min(br - abs(bm - A @ x))
    __minus_tol = lambda x: -__tol(x)

    if maxQ==False:
        if x is None:
            x = np.zeros(A.shape[1])
        return __tol(x)
    else:
        from scipy.optimize import minimize

        ccode, x, fun = __tolsolvty(A.a, A.b, b.a, b.b, weight=x0, maxiter=maxiter, \
                                    tol_f=tol, tol_x=tol, tol_g=tol)
        return ccode, x, fun

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
                angle_A[k, l] = np.random.choice([_inf[k,l], _sup[k,l]])
        tmp = np.linalg.cond(angle_A)
        cond = tmp if tmp<cond else cond

    return np.sqrt(A.shape[1]) * _max * cond * \
           (np.linalg.norm(_arg_max, ord=2)/np.sqrt(sum(abs(b)**2)))
