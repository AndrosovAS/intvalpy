import numpy as np

from ..kernel.abstract import BaseRecFun


class Uni(BaseRecFun):
    """
    To check the interval system of linear equations for its weak compatibility,
    the recognizing functional Uni should be used.
    """

    @staticmethod
    def _constituent(A, b, x, weight=None):
        if weight is None:
            weight = np.ones(len(b))
        return weight * (b.rad - (b.mid - A @ x).mig)


    @staticmethod
    def _value(A, b, x, weight=None):
        return np.min(Uni.constituent(A, b, x, weight=weight))


    @staticmethod
    def calcfg(x, infA, supA, Am, Ar, bm, br, weight):
        index = x>=0
        Am_x = Am @ x
        Ar_absx = Ar @ np.abs(x)
        infs = bm - (Am_x + Ar_absx)
        sups = bm - (Am_x - Ar_absx)
        mig = np.array([
            0.0 if inf*sup <= 0 else min(abs(inf), abs(sup))
            for inf, sup in zip(infs, sups)
        ])
        uni = weight * (br - mig)
        mc = np.argmin(uni)
        if -infs[mc] <= sups[mc]:
            duni = weight[mc] * (supA[mc] * index + infA[mc] * (~index))
        else:
            duni = -weight[mc] * (infA[mc] * index + supA[mc] * (~index))
        return -uni[mc], -duni


    @staticmethod
    def calcfg_constr(x, infA, supA, Am, Ar, bm, br, weight, linear_constraint):
        p, dp = BaseRecFun.linear_penalty(x, linear_constraint)
        uni, duni = Uni.calcfg(x, infA, supA, Am, Ar, bm, br, weight)

        return uni + p, duni + dp


    @staticmethod
    def _maximize(A, b, x0=None, weight=None, linear_constraint=None, **kwargs):
        xr, fr, nit, ncalls, ccode = BaseRecFun.optimize(
            A, b,
            Uni,
            x0=x0,
            weight=weight,
            linear_constraint=linear_constraint,
            **kwargs
        )
        return xr, -fr, nit, ncalls, ccode