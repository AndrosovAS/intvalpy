import numpy as np
import itertools

from .ralgb5 import ralgb5
from .utils import rad, mid, inf, sup, mag


#############################################################################################################
#############################################################################################################
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
    def optimize(A, b, recfunc, x0=None, weight=None, linear_constraint=None, **kwargs):

        n, m = A.shape
        assert n == len(b), 'Inconsistent dimensions of matrix and right-hand side vector'


        infA, supA = inf(A), sup(A)
        Am, Ar = mid(A), rad(A)
        bm, br = mid(b), rad(b)

        if weight is None:
            weight = np.ones(n)

        # для штрафной функции alpha = sum G x - c, где G-матрица ограничений, c-вектор ограничений
        # находим значение весового коэффициента mu, чтобы гарантировано не выходить за пределы ограничений
        if linear_constraint is None:
            calcfg = lambda x: recfunc.calcfg(x, infA, supA, Am, Ar, bm, br, weight)
        else:
            if linear_constraint.mu is None:
                mag_value = mag(A)
                linear_constraint.mu = linear_constraint.find_mu(np.max(mag_value))

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


    @classmethod
    def constituent(cls, A, b, x, weight=None):
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
        return cls._constituent(A, b, x, weight=weight)


    @classmethod
    def value(cls, A, b, x, weight=None):
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
        return cls._value(A, b, x, weight=weight)


    @classmethod
    def maximize(cls, A, b, x0=None, weight=None, linear_constraint=None, **kwargs):
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
        return cls._maximize(A, b, x0=x0, weight=weight, linear_constraint=linear_constraint, **kwargs)


#############################################################################################################
#############################################################################################################
class LinearConstraint:
    """
    Linear constraint on the variables.

    The constraint has the general inequality form:
        lb <= C <= ub


    Parameters:
        C: array_like, shape (n, m)
            Matrix defining the constraint.

        lb: array_like, shape (n, ), optional
            Lower limits on the constraint. Defaults to lb = -np.inf (no limits).

        ub: array_like, shape (n, ), optional
            Upper limits on the constraint. Defaults to ub = np.inf (no limits).

        mu: float
            The weighting factor by which the penalty is multiplied. Default value is None.
    """


    def __init__(self, C, lb=None, ub=None, mu=None):
        # TODO
        # идёт преобразование C x <= b
        # надо удалить все строки, где значение inf, а также, где нет зависимости от x

        assert C.shape[0] >= 1, 'Inconsistent dimension of matrix'
        if (not lb is None) or (not ub is None):
            self.C, self.b = [], []
            if (not lb is None):
                assert C.shape[0] == len(lb), 'Inconsistent dimensions of matrix and left-hand side vector'
                for k in range(C.shape[0]):
                    if abs(lb[k]) != np.inf and C[k].any():
                        self.C.append(-C[k])
                        self.b.append(-lb[k])

            if (not ub is None):
                assert C.shape[0] == len(ub), 'Inconsistent dimensions of matrix and left-hand side vector'
                for k in range(C.shape[0]):
                    if abs(ub[k]) != np.inf and C[k].any():
                        self.C.append(C[k])
                        self.b.append(ub[k])

            n = len(self.C)
            if n == 0:
                self.C.append(C[0])
                self.b.append(ub[0])
            else:
                w = np.random.uniform(1, 1, n) # TODO
                # TODO
                # переписать более оптимальным способом
                W = np.zeros( (n, C.shape[1]), dtype=np.float64)
                for k in range(W.shape[1]):
                    W[:, k] = w

            self.C, self.b = np.array(self.C)*W, np.array(self.b)*w


        else:
            self.C, self.b = C[0], np.array([np.inf])

        self.mu = mu

    def largeCondQ(self, x, i):
        Cix = self.C[i] @ x
        return Cix, Cix > self.b[i]

    def find_mu(self, numerator):
        # solve |sum(C[:, k] * x)| -> min, for x
        # sum(x) >= 1, x_i \in {0, 1} \forall i = 1,..., len(C)
        denominator = []
        for c in self.C.T:
            c = c[ c!=0 ]
            n = c.shape[0]
            if n == 0: continue

            dot = np.zeros(2**n)
            k = 0
            for x in itertools.product([0, 1], repeat=n):
                dot[k] = c @ x
                k += 1
            denominator.append(min(abs(dot[1:])))

        self.mu = numerator / min(denominator)
        return self.mu