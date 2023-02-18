import numpy as np
import itertools
from collections.abc import Sequence

from .RealInterval import ClassicalArithmetic, KaucherArithmetic, ArrayInterval, Interval, INTERVAL_CLASSES, single_type, ARITHMETICS

infinity = float('inf')
nan = np.nan

################################################################################
################################################################################
import itertools

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
                w = np.random.uniform(1, 2, n)
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

################################################################################
################################################################################


def get_shape(lst, shape=()):
    """
    Возвращает форму вложенных списков аналогично форме numpy.

    Parameters:
                lst: list
                    Вложенный список.

                shape: tuple
                    Форма до текущей глубины рекурсии.

    Returns:
                out: tuple
                    Форма текущей глубины. В конце это будет полная глубина.
    """

    if isinstance(lst, single_type) or (len(lst) == 1 and isinstance(lst, INTERVAL_CLASSES)):
        return shape
    elif isinstance(lst, np.ndarray):
        return lst.shape

    if isinstance(lst[0], Sequence):
        l = len(lst[0])
        if not all(len(item) == l for item in lst):
            msg = 'Не все списки имеют одинаковую длину!'
            raise ValueError(msg)

    shape += (len(lst), )

    # рекусрсия
    shape = get_shape(lst[0], shape)

    return shape


def asinterval(a):
    """

    To convert the input data to the interval type, use the asinterval function:

    Parameters:

        a: int, float, array_like
            Input data in any form that can be converted to an interval data type.
            These include int, float, list and ndarrays.


    Returns:

        out: Interval
            The conversion is not performed if the input is already of type Interval.
            Otherwise an object of interval type is returned.
    """

    if isinstance(a, INTERVAL_CLASSES):
        return a

    elif isinstance(a, single_type):
        return Interval(a, a, sortQ=False)

    elif isinstance(a, (list, np.ndarray)):
        a = np.asarray(a)
        shape = get_shape(a)

        result = zeros(shape)
        for index in itertools.product(*result.ranges):
            if isinstance(a[index], INTERVAL_CLASSES):
                result[index] = a[index]
            else:
                result[index] = Interval(a[index], a[index], sortQ=False)
        return result

    else:
        msg = 'Invalid input data.'
        raise TypeError(msg)


def intersection(x, y):
    """
    Покомпонентное пересечение двух интервальных массивов.

    Parameters:
                A, B: Interval
                    В случае, если операнды не являются интервальным типом, то
                    они преобразуются функцией asinterval.

    Returns:
                out: Interval
                    Возвращается массив пересечённых интервалов.
                    Если некоторые интервалы не пересекаются, то на их месте
                    выводится интервал Interval(None, None).
    """

    def intersect(x, y):
        sup = y.a if x.a < y.a else x.a
        inf = x.b if x.b < y.b else y.b
        if sup - inf <= 1e-15:
            return Interval(sup, inf, sortQ=True)
        else:
            return Interval(nan, nan)

    if isinstance(x, ArrayInterval) and isinstance(y, ArrayInterval):
        assert x.shape == y.shape, 'Не совпадают размерности входных массивов!'
        return ArrayInterval(np.vectorize(intersect)(x.data, y.data))
    elif isinstance(x, ARITHMETICS) and isinstance(y, ArrayInterval):
        return ArrayInterval(np.vectorize(intersect)(x, y.data))
    elif isinstance(x, ArrayInterval) and isinstance(y, ARITHMETICS):
        return ArrayInterval(np.vectorize(intersect)(x.data, y))
    else:
        return intersect(x, y)

def dist(x, y, order=infinity):
    """
    Метрика в интервальных пространствах.

    Parameters:
                a, b: Interval
                    Интервалы между которыми необходимо рассчитать dist.
                    В случае многомерности операндов вычисляется мультиметрика.

                order: int
                    Задаются различные метрики. По умолчанию используется
                    Чебышёвское расстояние.

    Returns:
                out: float
                    Возвращается расстояние между входными операндами.
    """

    def cheb(x, y):
        return np.maximum(abs(x.a - y.a), abs(x.b - y.b))

    if order == infinity:
        return np.amax(cheb(x, y))
    elif isinstance(order, int):
        return pow(np.sum(cheb(x, y) ** order), 1/order)
    else:
        raise Exception('Не верно задан порядок нормы order.')


def zeros(shape):
    """
    To create an interval array where each element is point and equal to zero.

    Parameters:

        shape: int, tuple
            Shape of the new interval array, e.g., (2, 3) or 4.

    Returns:

        out: Interval
            An interval array of zeros with a given shape
    """
    return Interval(np.zeros(shape, dtype='float64'),
                    np.zeros(shape, dtype='float64'), sortQ=False)


def full(shape, inf, sup):
    return Interval(np.full(shape, inf, dtype='float64'),
                    np.full(shape, sup, dtype='float64'), sortQ=False)


def eye(N, M=None, k=0):
    """
    Return a 2-D interval array with ones on the diagonal and zeros elsewhere.

    Parameters:

        N: int
            Shape of the new interval array, e.g., (2, 3) or 4.

        M: int, optional
            Number of columns in the output. By default, M = N.

        k: int, optional
            Index of the diagonal: 0 refers to the main diagonal, a positive value refers
            to an upper diagonal, and a negative value to a lower diagonal. By default, k = 0.


    Returns:

        out: Interval of shape (N, M)
            An interval array where all elements are equal to zero, except for the k-th diagonal,
            whose values are equal to one.
    """


    if M is None:
        M = N
    return Interval(np.eye(N, M=M, k=k, dtype=np.int64),
                    np.eye(N, M=M, k=k, dtype=np.int64), sortQ=False)


def diag(v, k=0):
    """
    Extract a diagonal or construct a diagonal interval array.

    Parameters:

        v: Interval
            If v is a 2-D interval array, return a copy of its k-th diagonal.
            If v is a 1-D interval array, return a 2-D interval array with v on the k-th diagonal.

        k: int, optional
            Diagonal in question. Use k>0 for diagonals above the main diagonal, and k<0 for diagonals
            below the main diagonal. By default, k=0.

    Returns:

        out: Interval
            The extracted diagonal or constructed diagonal interval array.
    """

    return asinterval(np.diag(v.data, k=k))


def compmat(A):
    """
    Компарант интервальной матрицы
    """
    Amag = A.mag
    Amig = A.mig
    return np.array([[Amig[k, l] if k==l else -Amag[k, l] for l in range(A.shape[1])] for k in range(A.shape[0])])


def Neumaier(n, theta, infb=None, supb=None):
    """
    This system is a parametric interval linear system, first proposed by K. Reichmann [2],
    and then slightly modified by A. Neumaier. The matrix of the system can be both regular
    and not strongly regular for some values of the diagonal parameter. It is shown that
    n × n matrices are non-singular for theta > n provided that n is even, and, for odd order n,
    the matrices are non-singular for theta > sqrt(n^2 - 1).

    Parameters:

        n: int
            Dimension of the interval system. It may be greater than or equal to two.

        theta: float, optional
            Nonnegative real parameter, which is the number that stands on the main
            diagonal of the matrix А.

        infb: float, optional
            A real parameter that specifies the lower endpoints of the components
            of the right-hand side vector. By default, infb = -1.

        supb: float, optional
            A real parameter that specifies the upper endpoints of the components
            of the right-hand side vector. By default, supb = 1.


    Returns:

        out: Interval, tuple
            The interval matrix and interval vector of the right side are returned, respectively.
    """


    interval = [None, None]
    if infb is None:
        interval[0] = -1
    else:
        interval[0] = infb
    if supb is None:
        interval[1] = 1
    else:
        interval[1] = supb
    b = Interval([interval]*n)

    A = zeros((n,n)) + Interval(0, 2, sortQ=False)
    for k in range(n):
        A[k, k] = Interval(theta, theta, sortQ=False)

    return A, b


def Shary(n, N=None, alpha=0.23, beta=0.35):
    """
    One of the popular test systems is the Shary system. Due to its symmetry, it is quite simple
    to determine the structure of its united solution set as well as other solution sets.
    Changing the values of the system parameters, you can get an extensive family of interval linear
    systems for testing the numerical algorithms. As the parameter beta decreases, the matrix
    of the system becomes more and more singular, and the united solution set enlarges indefinitely.

    Parameters:

        n: int
            Dimension of the interval system. It may be greater than or equal to two.

        N: float, optional
            A real number not less than (n − 1). By default, N = n.

        alpha: float, optional
            A parameter used for specifying the lower endpoints of the elements in the interval matrix.
            The parameter is limited to 0 < alpha <= beta <= 1. By default, alpha = 0.23.

        beta: float, optional
            A parameter used for specifying the upper endpoints of the elements in the interval matrix.
            The parameter is limited to 0 < alpha <= beta <= 1. By default, beta = 0.35.

    Returns:

        out: Interval, tuple
            The interval matrix and interval vector of the right side are returned, respectively.
    """


    if N is None:
        N = n
    else:
        assert N>=n-1, "Parameter N must be a real number not less than (n − 1)."

    if alpha < 0 or beta < alpha or beta > 1:
        raise Exception('The parameter is limited to 0 < alpha <= beta <= 1.')

    A = zeros((n,n)) + Interval(alpha-1, 1-beta, sortQ=False)
    b = zeros(n) + Interval(1-n, n-1, sortQ=False)

    for k in range(n):
        A[k, k] = Interval(n-1, N, sortQ=False)

    return A, b


def Toft(n, r=0, R=0):
    assert r >= 0, 'The parameter r must be a positive real number.'
    assert R >= 0, 'The parameter R must be a positive real number.'

    A = zeros((n,n))
    b = zeros(n) + Interval(1-R, 1+R, sortQ=False)

    a = zeros(n)
    for k in range(n):
        A[k, k] = Interval(1-r, 1+r, sortQ=False)
        a[k] = Interval(k+1-r, k+1+r)
    A[:, -1], A[-1, :] = a.copy, a.copy

    return A, b


def isnan(x):
    def _isnan(x):
        isnanQ = np.isnan(float(x.a)) or np.isnan(float(x.b))
        return isnanQ
    if isinstance(x, ARITHMETICS):
        return _isnan(x)
    else:
        return np.vectorize(_isnan)(x.data)


subset = lambda a, b: np.array(((a.a >= b.a) & (a.b <= b.b)), dtype=np.bool).all()
superset = lambda a, b: subset(b, a)

proper_subset = lambda a, b: np.array(((a.a > b.a) & (a.b < b.b)), dtype=np.bool).all()
proper_superset = lambda a, b: proper_subset(b, a)

contain = lambda a, b: np.array(((a.a >= b.a) & (a.b <= b.b)), dtype=np.bool)
supercontain = lambda a, b: subset(b, a)

def sqrt(x):
    """
    Interval enclosure of the square root intrinsic over an interval.

    Parameters:

        x: Interval
            The values whose square-roots are required.


    Returns:

        out: Interval
            An array of the same shape as x, containing the interval enclosure of the square root
            of each element in x.
    """
    return np.sqrt(x)

def exp(x):
    """
    Interval enclosure of the exponential intrinsic over an interval.

    Parameters:

        x: Interval
            The values to take the exponent from.


    Returns:

        out: Interval
            An array of the same shape as x, containing the interval enclosure of the exponential
            of each element in x.
    """
    return np.exp(x)

def log(x):
    """
    Interval enclosure of the natural logarithm intrinsic over an interval.

    Parameters:

        x: Interval
            The values to take the natural logarithm from.


    Returns:

        out: Interval
            An array of the same shape as x, containing the interval enclosure of the natural logarithm
            of each element in x.
    """
    return np.log(x)

def sin(x):
    """
    Interval enclosure of the sin intrinsic over an interval.

    Parameters:

        x: Interval
            The values to take the sin from.


    Returns:

        out: Interval
            An array of the same shape as x, containing the interval enclosure of the sin
            of each element in x.
    """
    return np.sin(x)

def cos(x):
    """
    Interval enclosure of the cos intrinsic over an interval.

    Parameters:

        x: Interval
            The values to take the cos from.


    Returns:

        out: Interval
            An array of the same shape as x, containing the interval enclosure of the cos
            of each element in x.
    """
    return np.cos(x)


def sgn(x):
    def _sgn(x):
        if x.b < 0:
            return Interval(-1, -1, sortQ=False)
        elif x.a < 0 and x.b == 0:
            return Interval(-1, 0, sortQ=False)
        elif x.a < 0 and 0 < x.b:
            return Interval(-1, 1, sortQ=False)
        elif x.a == 0 and x.b == 0:
            return Interval(0, 0, sortQ=False)
        elif x.a == 0 and 0 < x.b:
            return Interval(0, 1, sortQ=False)
        else:
            return Interval(1, 1, sortQ=False)

    if isinstance(x, ARITHMETICS):
        return _sgn(x)
    else:
        return asinterval(np.vectorize(_sgn)(x.data))
