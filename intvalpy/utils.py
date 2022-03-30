import numpy as np
import itertools
from collections.abc import Sequence

from .RealInterval import ClassicalArithmetic, KaucherArithmetic, ArrayInterval, Interval, INTERVAL_CLASSES, single_type, ARITHMETICS

infinity = float('inf')
nan = float('nan')


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
