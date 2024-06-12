import numpy as np

from ..kernel.abstract import BaseRecFun


class Uss(BaseRecFun):
    """
    To check the interval system of linear equations for its weak compatibility,
    the recognizing functional Uss should be used.
    """

    @staticmethod
    def _constituent(A, b, x, weight=None):
        if weight is None:
            weight = np.ones(len(b))
        return weight * (b.rad + A.rad @ np.abs(x) - np.abs(b.mid - A.mid @ x))


    @staticmethod
    def _value(A, b, x, weight=None):
        return np.min(Uss.constituent(A, b, x, weight=weight))


    @staticmethod
    def calcfg(x, infA, supA, Am, Ar, bm, br, weight):
        signx = np.sign(x)
        bm_Amx = bm - Am @ x
        uss = weight * (br + Ar @ np.abs(x) - np.abs(bm_Amx))
        mc = np.argmin(uss)
        duss = weight[mc] * (signx*Ar[mc] + np.sign(bm_Amx[mc]) * Am[mc])
        return -uss[mc], -duss


    @staticmethod
    def calcfg_constr(x, infA, supA, Am, Ar, bm, br, weight, linear_constraint):
        p, dp = BaseRecFun.linear_penalty(x, linear_constraint)
        uni, duni = Uss.calcfg(x, infA, supA, Am, Ar, bm, br, weight)

        return uni + p, duni + dp


    @staticmethod
    def _maximize(A, b, x0=None, weight=None, linear_constraint=None, **kwargs):
        xr, fr, nit, ncalls, ccode = BaseRecFun.optimize(
            A, b,
            Uss,
            x0=x0,
            weight=weight,
            linear_constraint=linear_constraint,
            **kwargs
        )
        return xr, -fr, nit, ncalls, ccode