import numpy as np

from intvalpy.RealInterval import Interval
from intvalpy.ralgb5 import ralgb5
import cvxopt


class BaseRecFun:

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
    def optimize(A, b, recfunc, x0=None, weight=None, linear_constraint=None, **kwargs):

        n, m = A.shape
        assert n == len(b), 'Inconsistent dimensions of matrix and right-hand side vector'


        infA, supA = A.a, A.b
        Am, Ar = A.mid, A.rad
        bm, br = b.mid, b.rad

        if weight is None:
            weight = np.ones(n)

        # для штрафной функции alpha = sum G x - c, где G-матрица ограничений, c-вектор ограничений
        # находим значение весового коэффициента mu, чтобы гарантировано не выходить за пределы ограничений
        if linear_constraint is None:
            calcfg = lambda x: recfunc.calcfg(x, infA, supA, Am, Ar, bm, br, weight)
        else:
            if linear_constraint.mu is None:
                linear_constraint.mu = linear_constraint.find_mu(np.max(A.mag))

            calcfg = lambda x: recfunc.calcfg_constr(x, infA, supA, Am, Ar, bm, br, weight, linear_constraint)


        if x0 is None:
            Ac = np.array(Am, dtype=np.float64)
            bc = np.array(bm, dtype=np.float64)

            sv = np.linalg.svd(Ac, compute_uv=False)
            minsv, maxsv = np.min(sv), np.max(sv)

            if (minsv != 0 and maxsv/minsv < 1e15):
                x0 = np.linalg.lstsq(Ac, bc, rcond=-1)[0]
            else:
                x0 = np.zeros(m)
        else:
            x0 = np.copy(x0)

        return ralgb5(calcfg, x0, **kwargs)


class Tol:
    """
    To check the interval system of linear equations for its strong compatibility,
    the recognizing functional Tol should be used.
    """

    @staticmethod
    def constituent(A, b, x, weight=None):
        """
        The function computes all the formings of the recognizing functional and returns them.

        Parameters:

            A: Interval
                The input interval matrix of ISLAE, which can be either square or rectangular.

            b: Interval
                The interval vector of the right part of the ISLAE.

            x: np.array, optional
                The point at which the recognizing functional is calculated.

            weight: float, np.array, optional
                The vector of weight coefficients for each forming of the recognizing functional.
                By default, it is a vector consisting of ones.


        Returns:

            out: float
                The values of each forming of the recognizing functional at the point x are returned.
        """
        if weight is None:
            weight = np.ones(len(b))
        return weight * (b.rad - (b.mid - A @ x).mag)


    @staticmethod
    def value(A, b, x, weight=None):
        """
        The function computes the value of the recognizing functional at the point x.

        Parameters:

            A: Interval
                The input interval matrix of ISLAE, which can be either square or rectangular.

            b: Interval
                The interval vector of the right part of the ISLAE.

            x: np.array, optional
                The point at which the recognizing functional is calculated.

            weight: float, np.array, optional
                The vector of weight coefficients for each forming of the recognizing functional.
                By default, it is a vector consisting of ones.


        Returns:

            out: float
                The value of the recognizing functional at the point x.
        """

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
    def maximize(A, b, x0=None, weight=None, linear_constraint=None, **kwargs):
        """
        The function is intended for finding the global maximum of the recognizing functional.
        The ralgb5 subgradient method is used for optimization.

        Parameters:

            A: Interval
                The input interval matrix of ISLAE, which can be either square or rectangular.

            b: Interval
                The interval vector of the right part of the ISLAE.

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

        xr, fr, nit, ncalls, ccode = BaseRecFun.optimize(
            A, b,
            Tol,
            x0=x0,
            weight=weight,
            linear_constraint=linear_constraint,
            **kwargs
        )
        return xr, -fr, nit, ncalls, ccode
    
class ChebNorm:

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


class Uni:
    """
    To check the interval system of linear equations for its weak compatibility,
    the recognizing functional Uni should be used.
    """

    @staticmethod
    def constituent(A, b, x, weight=None):
        """
        The function computes all the formings of the recognizing functional and returns them.

        Parameters:

            A: Interval
                The input interval matrix of ISLAE, which can be either square or rectangular.

            b: Interval
                The interval vector of the right part of the ISLAE.

            x: np.array, optional
                The point at which the recognizing functional is calculated.

            weight: float, np.array, optional
                The vector of weight coefficients for each forming of the recognizing functional.
                By default, it is a vector consisting of ones.


        Returns:

            out: float
                The values of each forming of the recognizing functional at the point x are returned.
        """
        if weight is None:
            weight = np.ones(len(b))
        return weight * (b.rad - (b.mid - A @ x).mig)


    @staticmethod
    def value(A, b, x, weight=None):
        """
        The function computes the value of the recognizing functional at the point x.

        Parameters:

            A: Interval
                The input interval matrix of ISLAE, which can be either square or rectangular.

            b: Interval
                The interval vector of the right part of the ISLAE.

            x: np.array, optional
                The point at which the recognizing functional is calculated.

            weight: float, np.array, optional
                The vector of weight coefficients for each forming of the recognizing functional.
                By default, it is a vector consisting of ones.


        Returns:

            out: float
                The value of the recognizing functional at the point x.
        """
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
    def maximize(A, b, x0=None, weight=None, linear_constraint=None, **kwargs):
        """
        The function is intended for finding the global maximum of the recognizing functional.
        The ralgb5 subgradient method is used for optimization.
        It is important to note that Uni is not a convex function, so there is a possibility
        of obtaining an incorrect point at which the maximum is reached. However, the function
        is convex in each of the orthants, so constrained optimization can be used.

        Parameters:

            A: Interval
                The input interval matrix of ISLAE, which can be either square or rectangular.

            b: Interval
                The interval vector of the right part of the ISLAE.

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

        xr, fr, nit, ncalls, ccode = BaseRecFun.optimize(
            A, b,
            Uni,
            x0=x0,
            weight=weight,
            linear_constraint=linear_constraint,
            **kwargs
        )
        return xr, -fr, nit, ncalls, ccode


class Uss:
    """
    To check the interval system of linear equations for its weak compatibility,
    the recognizing functional Uss should be used.
    """

    @staticmethod
    def constituent(A, b, x, weight=None):
        """
        The function computes all the formings of the recognizing functional and returns them.

        Parameters:

            A: Interval
                The input interval matrix of ISLAE, which can be either square or rectangular.

            b: Interval
                The interval vector of the right part of the ISLAE.

            x: np.array, optional
                The point at which the recognizing functional is calculated.

            weight: float, np.array, optional
                The vector of weight coefficients for each forming of the recognizing functional.
                By default, it is a vector consisting of ones.


        Returns:

            out: float
                The values of each forming of the recognizing functional at the point x are returned.
        """
        if weight is None:
            weight = np.ones(len(b))
        return weight * (b.rad + A.rad @ np.abs(x) - np.abs(b.mid - A.mid @ x))


    @staticmethod
    def value(A, b, x, weight=None):
        """
        The function computes the value of the recognizing functional at the point x.

        Parameters:

            A: Interval
                The input interval matrix of ISLAE, which can be either square or rectangular.

            b: Interval
                The interval vector of the right part of the ISLAE.

            x: np.array, optional
                The point at which the recognizing functional is calculated.

            weight: float, np.array, optional
                The vector of weight coefficients for each forming of the recognizing functional.
                By default, it is a vector consisting of ones.


        Returns:

            out: float
                The value of the recognizing functional at the point x.
        """
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
    def maximize(A, b, x0=None, weight=None, linear_constraint=None, **kwargs):
        """
        The function is intended for finding the global maximum of the recognizing functional.
        The ralgb5 subgradient method is used for optimization.
        It is important to note that Uss is not a convex function, so there is a possibility
        of obtaining an incorrect point at which the maximum is reached. However, the function
        is convex in each of the orthants, so constrained optimization can be used.

        Parameters:

            A: Interval
                The input interval matrix of ISLAE, which can be either square or rectangular.

            b: Interval
                The interval vector of the right part of the ISLAE.

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

        xr, fr, nit, ncalls, ccode = BaseRecFun.optimize(
            A, b,
            Uss,
            x0=x0,
            weight=weight,
            linear_constraint=linear_constraint,
            **kwargs
        )
        return xr, -fr, nit, ncalls, ccode
