import numpy as np

from intvalpy.RealInterval import Interval, INTERVAL_CLASSES
from intvalpy.ralgb5 import ralgb5


################################################################################
################################################################################
# define calcfg
# tol
def calcfg_tol(x, infA, supA, Ac, Ar, bc, functional, weight):
    index = x>=0
    Ac_x = Ac @ x
    Ar_absx = Ar @ np.abs(x)
    infs = bc - (Ac_x + Ar_absx)
    sups = bc - (Ac_x - Ar_absx)
    tt = functional(infs, sups)
    mc = np.argmin(tt)
    if -infs[mc] <= sups[mc]:
        dd = weight[mc] * (infA[mc] * index + supA[mc] * (~index))
    else:
        dd = -weight[mc] * (supA[mc] * index + infA[mc] * (~index))
    return -tt[mc], -dd

def calcfg_tol_constraint(x, infA, supA, Ac, Ar, bc, functional, weight, linear_constraint):
    n, m = linear_constraint.C.shape
    alphai, dalphai = np.zeros(n), np.zeros((n, m))
    for i in range(n):
        Cix, condQ = linear_constraint.largeCondQ(x, i)
        if condQ:
            alphai[i] = Cix - linear_constraint.b[i]
            dalphai[i] = linear_constraint.C[i]

    alpha = linear_constraint.mu * sum(alphai)
    grad_alpha = linear_constraint.mu * np.array([ sum(dalphai[:, k]) for k in range(m)])

    index = x>=0
    Ac_x = Ac @ x
    Ar_absx = Ar @ np.abs(x)
    infs = bc - (Ac_x + Ar_absx)
    sups = bc - (Ac_x - Ar_absx)
    tt = functional(infs, sups) - alpha
    mc = np.argmin(tt)
    if -infs[mc] <= sups[mc]:
        dd = weight[mc] * (infA[mc] * index + supA[mc] * (~index)) - grad_alpha
    else:
        dd = -weight[mc] * (supA[mc] * index + infA[mc] * (~index)) - grad_alpha
    return -tt[mc], -dd

################################################################################
# uni
def calcfg_uni(x, infA, supA, Ac, Ar, bc, functional, weight):
    index = x>=0
    Ac_x = Ac @ x
    Ar_absx = Ar @ np.abs(x)
    infs = bc - (Ac_x + Ar_absx)
    sups = bc - (Ac_x - Ar_absx)
    tt = functional(infs, sups)
    mc = np.argmin(tt)
    if -infs[mc] <= sups[mc]:
        dd = weight[mc] * (supA[mc] * index + infA[mc] * (~index))
    else:
        dd = -weight[mc] * (infA[mc] * index + supA[mc] * (~index))
    return -tt[mc], -dd

def calcfg_uni_constraint(x, infA, supA, Ac, Ar, bc, functional, weight, linear_constraint):
    n, m = linear_constraint.C.shape
    alphai, dalphai = np.zeros(n), np.zeros((n, m))
    for i in range(n):
        Cix, condQ = linear_constraint.largeCondQ(x, i)
        if condQ:
            alphai[i] = Cix - linear_constraint.b[i]
            dalphai[i] = linear_constraint.C[i]

    alpha = linear_constraint.mu * sum(alphai)
    grad_alpha = linear_constraint.mu * np.array([ sum(dalphai[:, k]) for k in range(m)])

    index = x>=0
    Ac_x = Ac @ x
    Ar_absx = Ar @ np.abs(x)
    infs = bc - (Ac_x + Ar_absx)
    sups = bc - (Ac_x - Ar_absx)
    tt = functional(infs, sups) - alpha
    mc = np.argmin(tt)
    if -infs[mc] <= sups[mc]:
        dd = weight[mc] * (supA[mc] * index + infA[mc] * (~index)) - grad_alpha
    else:
        dd = -weight[mc] * (infA[mc] * index + supA[mc] * (~index)) - grad_alpha
    return -tt[mc], -dd

################################################################################
################################################################################


def recfunsolvty(A, b, consistency='uni', x0=None, weight=None, tolx=1e-12, tolg=1e-12, tolf=1e-12, maxiter=2000, linear_constraint=None, alpha=2.3, nsims=30, h0=1, nh=3, q1=0.9, q2=1.1):

    n, m = A.shape
    A = A.to_float()
    b = b.to_float()

    infA, supA = A.a, A.b
    infb, supb = b.a, b.b

    Ac = 0.5 * (infA + supA)
    Ar = 0.5 * (supA - infA)
    bc = 0.5 * (infb + supb)
    br = 0.5 * (supb - infb)

    if weight is None:
        weight = np.ones(n)

    # для штрафной функции alpha = sum G x - c, где G-матрица ограничений, c-вектор ограничений
    # находим значение весового коэффициента mu, чтобы гарантировано не выходить за пределы ограничений
    if (not linear_constraint is None) and (linear_constraint.mu is None):
        linear_constraint.mu = linear_constraint.find_mu(np.max(A.mag))


    if x0 is None:
        sv = np.linalg.svd(Ac, compute_uv=False)
        minsv, maxsv = np.min(sv), np.max(sv)

        if (minsv != 0 and maxsv/minsv < 1e15):
            x0 = np.linalg.lstsq(Ac, bc, rcond=-1)[0]
        else:
            x0 = np.zeros(m)
    else:
        x0 = np.copy(x0)


    def mig(inf, sup):
        if inf*sup <= 0:
            return 0.0
        else:
            return min(abs(inf), abs(sup))

    if consistency=='uni':
        functional = lambda infs, sups: weight * (br - np.vectorize(mig)(infs, sups))
        if linear_constraint is None:
            def calcfg(x):
                return calcfg_uni(x, infA, supA, Ac, Ar, bc, functional, weight)
        else:
            def calcfg(x):
                return calcfg_uni_constraint(x, infA, supA, Ac, Ar, bc, functional, weight, linear_constraint)

    else:
        functional = lambda infs, sups: weight * (br - np.maximum(np.abs(infs), np.abs(sups)))
        if linear_constraint is None:
            def calcfg(x):
                return calcfg_tol(x, infA, supA, Ac, Ar, bc, functional, weight)
        else:
            def calcfg(x):
                return calcfg_tol_constraint(x, infA, supA, Ac, Ar, bc, functional, weight, linear_constraint)


    xr, fr, nit, ncalls, ccode = ralgb5(calcfg, x0, tolx=tolx, tolg=tolg, tolf=tolf, maxiter=maxiter, alpha=alpha, nsims=nsims, h0=h0, nh=nh, q1=q1, q2=q2)
    return xr, -fr, nit, ncalls, ccode


def dot(midA, radA, x):
    midAx = midA @ x
    radAabsx = radA @ abs(x)
    return Interval(midAx - radAabsx, midAx + radAabsx, sortQ=False)


def Tol(A, b, x=None, maxQ=False, x0=None, weight=None, tolx=1e-12, tolg=1e-12, tolf=1e-12, maxiter=2000, linear_constraint=None, alpha=2.3, nsims=30, h0=1, nh=3, q1=0.9, q2=1.1):
    """
    When it is necessary to check the interval system of linear equations for its strong
    solvability you should use the Tol functionality. If maxQ=True, then the maximum
    of the functional is found, otherwise, the value at point x is calculated.
    To optimize it, a proven the tolsolvty program, which is suitable for solving practical problems.


    Parameters:

        A: Interval
            The input interval matrix of ISLAE, which can be either square or rectangular.

        b: Interval
            The interval vector of the right part of the ISLAE.

        x: float, array_like, optional
            The point at which the recognizing functional is calculated. By default, x is equal to an array of zeros.

        maxQ: bool, optional
            If the parameter value is True, then the functional is maximized.

        x0: float, array_like, optional
            The initial guess for finding the global maximum.

        tolx: float, optional
            Absolute error in xopt between iterations that is acceptable for convergence.

        tolg: float, optional
            Absolute error in tolgrad between iterations that is acceptable for convergence.

        tolf: float, optional
            Absolute error in tolmax between iterations that is acceptable for convergence.

        maxiter: int, optional
            The maximum number of iterations.

        linear_constraint: LinearConstraint, optional
            System (lb <= C <= ub) describing linear dependence between parameters.

    Returns:

        out: float, tuple
            The value of the recognizing functional at point x is returned.
            If maxQ=True, then a tuple is returned, where the first element is the correctness of the optimization completion,
            the second element is the optimum point, and the third element is the value of the function at this point.
    """

    if not maxQ:
        if isinstance(x, INTERVAL_CLASSES):
            return min(b.rad - abs(b.mid - (A @ x)))
        else:
            x = np.zeros(A.shape[1]) if x is None else x
            return min(b.rad - (b.mid - (A @ x)).mag)
    else:
        xr, fr, nit, ncalls, ccode = recfunsolvty(A, b, consistency='tol', x0=x0, weight=weight, tolx=tolx, tolg=tolg, tolf=tolf, maxiter=maxiter, linear_constraint=linear_constraint,
                                                  alpha=alpha, nsims=nsims, h0=h0, nh=nh, q1=q1, q2=q2)
        ccode = False if (ccode==4 or ccode==5) else True
        return ccode, xr, fr


def Uni(A, b, x=None, maxQ=False, x0=None, weight=None, tolx=1e-12, tolg=1e-12, tolf=1e-12, maxiter=2000, linear_constraint=None, alpha=2.3, nsims=30, h0=1, nh=3, q1=0.9, q2=1.1):
    """
    When it is necessary to check an interval system of linear equations for its weak solvability
    you should use the Uni functionality. If maxQ=True, then the maximum of the functional is found,
    otherwise, the value at point x is calculated.

    To optimize it, the well-known Nelder-Mead method is used, which does not use gradients,
    since there is an absolute value in the function.


    Parameters:

        A: Interval
            The input interval matrix of ISLAE, which can be either square or rectangular.

        b: Interval
            The interval vector of the right part of the ISLAE.

        x: float, array_like, optional
            The point at which the recognizing functional is calculated. By default, x is equal to an array of zeros.

        maxQ: bool, optional
            If the parameter value is True, then the functional is maximized.

        x0: float, array_like, optional
            The initial guess for finding the global maximum.

        tolx: float, optional
            Absolute error in xopt between iterations that is acceptable for convergence.

        tolg: float, optional
            Absolute error in unigrad between iterations that is acceptable for convergence.

        tolf: float, optional
            Absolute error in unimax between iterations that is acceptable for convergence.

        maxiter: int, optional
            The maximum number of iterations.

        linear_constraint: LinearConstraint, optional
            System (lb <= C <= ub) describing linear dependence between parameters.

    Returns:

        out: float, tuple
            The value of the recognizing functional at point x is returned.
            If maxQ=True, then a tuple is returned, where the first element is the correctness of the optimization completion,
            the second element is the optimum point, and the third element is the value of the function at this point.
    """

    def ext_mig(x):
        if 0 in x:
            return Interval(0, 0, sortQ=False)
        else:
            return Interval(0, x.mig, sortQ=False)

    if not maxQ:
        if isinstance(x, INTERVAL_CLASSES):
            return min(b.rad - np.vectorize(ext_mig)( (b.mid - (A @ x)).data ))
        else:
            x = np.zeros(A.shape[1]) if x is None else x
            return min(b.rad - (b.mid - (A @ x)).mig)
    else:
        xr, fr, nit, ncalls, ccode = recfunsolvty(A, b, consistency='uni', x0=x0, weight=weight, tolx=tolx, tolg=tolg, tolf=tolf, maxiter=maxiter, linear_constraint=linear_constraint,
                                                  alpha=alpha, nsims=nsims, h0=h0, nh=nh, q1=q1, q2=q2)

        ccode = False if (ccode==4 or ccode==5) else True
        return ccode, xr, fr
