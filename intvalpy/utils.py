import numpy as np
import itertools
from collections.abc import Sequence

from .RealInterval import ClassicalArithmetic, KaucherArithmetic, ArrayInterval, Interval, INTERVAL_CLASSES, single_type, ARITHMETICS

infinity = float('inf')
nan = float('nan')

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

            dot = np.zeros(2**n)
            k = 0
            for x in itertools.product([0, 1], repeat=n):
                dot[k] = c @ x
                k += 1
            denominator.append(min(abs(dot[1:])))

        return numerator / min(denominator)

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
    Преобразование входных данных в массив типа Interval.

    Parameters:
                a: array_like
                    Входные данные, в любой форме, которые могут быть преобразованы
                    в массив интервалов.
                    Это включает в себя int, float, list и ndarrays.

    Returns:
                out: Interval
                    Преобразование не выполняется, если входные данные уже являются
                    типом Interval.
                    Если a - int, float, list или ndarrays, то возвращается
                    базовый класс Interval.
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
        msg = 'Входные данные неприводимы к интервальному массиву!'
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
    """Функция создаёт массив размерности shape."""
    return Interval(np.zeros(shape, dtype='float64'),
                    np.zeros(shape, dtype='float64'), sortQ=False)


def diag(mat):
    """Функция возвращает диагональные элементы матрицы."""
    mat = asinterval(mat)

    if mat.ndim != 2:
        raise Exception('Входной массив не является двумерным!')
    n = min(mat.shape)
    result = zeros(n)
    for k in range(n):
        result[k] = mat[k, k]
    return result


def compmat(A):
    """
    Компарант интервальной матрицы
    """
    Amag = A.mag
    Amig = A.mig
    return np.array([[Amig[k, l] if k==l else -Amag[k, l] for l in range(A.shape[1])] for k in range(A.shape[0])])


def eye(n):
    A = zeros((n, n))
    for k in range(n):
        A[k, k] = Interval(1, 1, sortQ=False)
    return A


def Neumeier(n, theta, infb=None, supb=None):
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
    if N is None:
        N = n
    else:
        assert N>=n-1, "Параметры заданы неверно!"

    if alpha < 0 or beta < alpha or beta > 1:
        raise Exception('Параметры заданы неверно!')

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


def create_data(x=None, N=3, model=None):

    def T(n, x):
        if n == 0:
            return 1
        elif n == 1:
            return x
        else:
            return 2 * x * T(n-1, x) - T(n-2, x)

    if x is None:
        x = uniform(-1, 1, 10)

    if model is None:
        model = lambda x: x ** 2

    b = model(x)
    A = []
    for k in range(len(x)):
        A.append([T(l, x[k]) for l in range(N)])
    A = asinterval(A)

    return A, b


def randint(inf, sup, shape=1):
    return Interval(np.random.randint(inf, sup, shape),
                    np.random.randint(inf, sup, shape))


def uniform(inf, sup, shape=1):
    return Interval(np.random.uniform(inf, sup, shape),
                    np.random.uniform(inf, sup, shape))


def normal(mu, sigma, shape=1):
    return Interval(np.random.normal(mu, sigma, shape),
                    np.random.normal(mu, sigma, shape))

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
    """Interval enclosure of the square root intrinsic over an interval."""
    return np.sqrt(x)

def exp(x):
    """Interval enclosure of the exponential intrinsic over an interval."""
    return np.exp(x)

def log(x):
    """Interval enclosure of the natural logarithm intrinsic over an interval."""
    return np.log(x)

def sin(x):
    """Interval enclosure of the sin intrinsic over an interval."""
    return np.sin(x)

def cos(x):
    """Interval enclosure of the cos intrinsic over an interval."""
    return np.sin(x)


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
