import numpy as np

import cvxopt
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}

from ..kernel.abstract import BaseRecFun


class ChebNorm:
    """
    To calculate the Chebyshev norm of linear equation system residual.
    """
    @staticmethod
    def constituent(A, b, x, weight=None):
        if weight is None:
            weight = np.ones(len(b))

        A_opt = ChebNorm.__calc_opt_A(x, A.a, A.b, b.mid)
        return weight * (b.rad.T + abs(b.mid.T - A_opt @ x))


    @staticmethod
    def value(A, b, x, weight=None):
        return np.max(ChebNorm.constituent(A, b, x, weight=weight))

    @staticmethod
    def __calc_lp_on_box(c_, x_min, x_max):
        c = cvxopt.matrix(c_)
        G = cvxopt.matrix(np.vstack([-np.eye(len(x_min)), np.eye(len(x_min))]))
        h = cvxopt.matrix(np.hstack([-x_min, x_max]).astype(np.double))
        sol = cvxopt.solvers.lp(c, G, h, verbose=True)
        return np.array(sol['x']).T

    @staticmethod
    def __calc_opt_A(x, infA, supA, bm):
        A_opt = np.zeros_like(infA)
        for j in range(len(bm)):
            x_min = ChebNorm.__calc_lp_on_box(x, infA[j], supA[j])
            f_min = x_min @ x
            x_max = ChebNorm.__calc_lp_on_box(-x, infA[j], supA[j])
            f_max = x_max @ (-x)
            if bm[j] - f_min >= f_max - bm[j]:
                A_opt[j] = x_min
            else:
                A_opt[j] = x_max
        return A_opt

    @staticmethod
    def calcfg(x, infA, supA, Am, Ar, bm, br, weight):
        if weight is None:
            weight = np.ones(len(bm))

        A_opt = ChebNorm.__calc_opt_A(x, infA, supA, bm)
        index = (weight * (br.T + abs(bm.T - A_opt @ x))).argmax()
        val =(weight * (br.T + abs(bm.T - A_opt @ x))).max()
        grad = A_opt[index,:] * np.sign(A_opt[index] @ x - bm[index])

        return val, grad


    @staticmethod
    def calcfg_constr(x, infA, supA, Am, Ar, bm, br, weight, linear_constraint):
        p, dp = BaseRecFun.linear_penalty(x, linear_constraint)
        tol, dtol = ChebNorm.calcfg(x, infA, supA, Am, Ar, bm, br, weight)

        return tol + p, dtol + dp


    @staticmethod
    def minimize(A, b, x0=None, weight=None, linear_constraint=None, **kwargs):
        xr, fr, nit, ncalls, ccode = BaseRecFun.optimize(
            A, b,
            ChebNorm,
            x0=x0,
            weight=weight,
            linear_constraint=linear_constraint,
            **kwargs
        )
        return xr, fr, nit, ncalls, ccode