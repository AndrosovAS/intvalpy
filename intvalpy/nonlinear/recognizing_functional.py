import numpy as np

from intvalpy.RealInterval import Interval
from intvalpy.ralgb5 import ralgb5


class BaseRecFun(object):

    @staticmethod
    def linear_penalty(x, linear_constraint):
        C, b = linear_constraint.C, linear_constraint.b
        mu = linear_constraint.mu

        n, m = C.shape
        # the arrays to store the values of the penalty function
        # and its Jacobian are initialized.
        arr_p, arr_dp = np.zeros(n), np.zeros((n, m))
        for i in range(n):
            #the condition that the vector x is within the specified bounds is tested.
            Cix, beyondQ = linear_constraint.largeCondQ(x, i)
            if beyondQ:
                arr_p[i] = Cix - b[i]
                arr_dp[i] = C[i]

        #the final value of the penalty function and its gradient vector are obtained.
        p = mu * np.sum(arr_p)
        dp = mu * np.sum(arr_dp, axis=0)
        return p, dp


    @staticmethod
    def optimize(model, grad, a, b, recfunc, x0, weight=None, linear_constraint=None, **kwargs):

        n, = a.shape
        assert n == len(b), 'Inconsistent dimensions of matrix and right-hand side vector'

        am, ar = a.mid, a.rad
        bm, br = b.mid, b.rad

        if weight is None:
            weight = np.ones(n)

        # для штрафной функции alpha = sum G x - c, где G-матрица ограничений, c-вектор ограничений
        # находим значение весового коэффициента mu, чтобы гарантировано не выходить за пределы ограничений
        if linear_constraint is None:
            calcfg = lambda x: recfunc.calcfg(x, model, grad, a, bm, br, weight)
        else:
            calcfg = lambda x: recfunc.calcfg_constr(x, model, grad, a, bm, br, weight, linear_constraint)

        x0 = np.copy(x0)

        return ralgb5(calcfg, x0, **kwargs)


    @classmethod
    def constituent(cls, model, a, b, x, weight=None):
        """
        The function computes all the formings of the recognizing functional and returns them.

        Parameters:

            ...

            x: np.array, optional
                The point at which the recognizing functional is calculated.

            weight: float, np.array, optional
                The vector of weight coefficients for each forming of the recognizing functional.
                By default, it is a vector consisting of ones.


        Returns:

            out: float
                The values of each forming of the recognizing functional at the point x are returned.
        """
        return cls._constituent(model, a, b, x, weight=weight)


    @classmethod
    def value(cls, model, a, b, x, weight=None):
        """
        The function computes the value of the recognizing functional at the point x.

        Parameters:

            ...

            x: np.array, optional
                The point at which the recognizing functional is calculated.

            weight: float, np.array, optional
                The vector of weight coefficients for each forming of the recognizing functional.
                By default, it is a vector consisting of ones.


        Returns:

            out: float
                The value of the recognizing functional at the point x.
        """
        return cls._value(model, a, b, x, weight=weight)


    @classmethod
    def maximize(cls, model, grad, a, b, x0, weight=None, linear_constraint=None, **kwargs):
        """
        The function is intended for finding the global maximum of the recognizing functional.
        The ralgb5 subgradient method is used for optimization.

        Parameters:

            ...

            x0: np.array, optional
                The initial assumption is at what point the maximum is reached. By default, x0
                is equal to the vector which is the solution (pseudo-solution) of the system
                mid(A) x = mid(b).

            weight: float, np.array, optional
                The vector of weight coefficients for each forming of the recognizing functional.
                By default, it is a vector consisting of ones.

            linear_constraint: LinearConstraint, optional
                System (lb <= C <= ub) describing linear dependence between parameters.
                By default, the problem of unconditional maximization is being solved.

            kwargs: optional params
                The ralgb5 function uses additional parameters to adjust its performance.
                These parameters include the step size, the stopping criteria, the maximum number
                of iterations and others. Specified in the function description ralgb5.


        Returns:

            out: tuple
                The function returns the following values in the specified order:
                1. the vector solution at which the recognition functional reaches its maximum,
                2. the value of the recognition functional,
                3. the number of iterations taken by the algorithm,
                4. the number of calls to the calcfg function,
                5. the exit code of the algorithm (1 = tolf, 2 = tolg, 3 = tolx, 4 = maxiter, 5 = error).
        """
        return cls._maximize(model, grad, a, b, x0, weight=weight, linear_constraint=linear_constraint, **kwargs)



class Tol(BaseRecFun):
    """
    To check the interval system of linear equations for its strong compatibility,
    the recognizing functional Tol should be used.
    """

    @staticmethod
    def _constituent(model, a, b, x, weight=None):
        if weight is None:
            weight = np.ones(len(b))
        return weight * (b.rad - (b.mid - model(a, x)).mag)


    @staticmethod
    def _value(model, a, b, x, weight=None):
        return np.min(Tol.constituent(model, a, b, x, weight=weight))


    @staticmethod
    def calcfg(x, model, grad, a, bm, br, weight):
        infsup = bm - model(a, x)
        tol = weight * (br - infsup.mag)
        mc = np.argmin(tol)
        gradx = np.array([g(a[mc], x) for g in grad])
        if -infsup[mc].a <= infsup[mc].b:
            dtol = weight[mc] * np.array([min(gx.a, gx.b) for gx in gradx])
        else:
            dtol = -weight[mc] * np.array([max(gx.a, gx.b) for gx in gradx])
        return -tol[mc], -dtol


    @staticmethod
    def calcfg_constr(x, model, grad, a, bm, br, weight, linear_constraint):
        p, dp = BaseRecFun.linear_penalty(x, linear_constraint)
        tol, dtol = Tol.calcfg(x, model, grad, a, bm, br, weight)

        return tol + p, dtol + dp


    @staticmethod
    def _maximize(model, grad, a, b, x0, weight=None, linear_constraint=None, **kwargs):
        xr, fr, nit, ncalls, ccode = BaseRecFun.optimize(
            model, grad,
            a, b,
            Tol,
            x0,
            weight=weight,
            linear_constraint=linear_constraint,
            **kwargs
        )
        return xr, -fr, nit, ncalls, ccode


class Uni(BaseRecFun):
    """
    To check the interval system of linear equations for its weak compatibility,
    the recognizing functional Uni should be used.
    """

    @staticmethod
    def _constituent(model, a, b, x, weight=None):
        if weight is None:
            weight = np.ones(len(b))
        return weight * (b.rad - (b.mid - model(a, x)).mig)


    @staticmethod
    def _value(model, a, b, x, weight=None):
        return np.min(Uni.constituent(model, a, b, x, weight=weight))


    @staticmethod
    def calcfg(x, model, grad, a, bm, br, weight):
        infsup = bm - model(a, x)
        uni = weight * (br - infsup.mig)
        mc = np.argmin(uni)
        gradx = np.array([g(a[mc], x) for g in grad])
        if -infsup[mc].a <= infsup[mc].b:
            duni = weight[mc] * np.array([max(gx.a, gx.b) for gx in gradx])
        else:
            duni = -weight[mc] * np.array([min(gx.a, gx.b) for gx in gradx])
        return -uni[mc], -duni


    @staticmethod
    def calcfg_constr(x, model, grad, a, bm, br, weight, linear_constraint):
        p, dp = BaseRecFun.linear_penalty(x, linear_constraint)
        uni, duni = Uni.calcfg(x, model, grad, a, bm, br, weight)

        return uni + p, duni + dp


    @staticmethod
    def _maximize(model, grad, a, b, x0, weight=None, linear_constraint=None, **kwargs):
        xr, fr, nit, ncalls, ccode = BaseRecFun.optimize(
            model, grad,
            a, b,
            Uni,
            x0,
            weight=weight,
            linear_constraint=linear_constraint,
            **kwargs
        )
        return xr, -fr, nit, ncalls, ccode
