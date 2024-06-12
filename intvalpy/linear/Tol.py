import numpy as np

from ..kernel.abstract import BaseRecFun


class Tol(BaseRecFun):
    """
    To check the interval system of linear equations for its strong compatibility,
    the recognizing functional Tol should be used.
    """

    @staticmethod
    def _constituent(A, b, x, weight=None):
        if weight is None:
            weight = np.ones(len(b))
        return weight * (b.rad - (b.mid - A @ x).mag)


    @staticmethod
    def _value(A, b, x, weight=None):
        return np.min(Tol.constituent(A, b, x, weight=weight))


    @staticmethod
    def calcfg(x, infA, supA, Am, Ar, bm, br, weight):
        index = x>=0
        Am_x = Am @ x
        Ar_absx = Ar @ np.abs(x)
        infs = bm - (Am_x + Ar_absx)
        sups = bm - (Am_x - Ar_absx)
        tol = weight * (br - np.maximum(abs(infs), abs(sups)))
        mc = np.argmin(tol)
        if -infs[mc] <= sups[mc]:
            dtol = weight[mc] * (infA[mc] * index + supA[mc] * (~index))
        else:
            dtol = -weight[mc] * (supA[mc] * index + infA[mc] * (~index))
        return -tol[mc], -dtol


    @staticmethod
    def calcfg_constr(x, infA, supA, Am, Ar, bm, br, weight, linear_constraint):
        p, dp = BaseRecFun.linear_penalty(x, linear_constraint)
        tol, dtol = Tol.calcfg(x, infA, supA, Am, Ar, bm, br, weight)

        return tol + p, dtol + dp


    @staticmethod
    def _maximize(A, b, x0=None, weight=None, linear_constraint=None, **kwargs):
        xr, fr, nit, ncalls, ccode = BaseRecFun.optimize(
            A, b,
            Tol,
            x0=x0,
            weight=weight,
            linear_constraint=linear_constraint,
            **kwargs
        )
        return xr, -fr, nit, ncalls, ccode