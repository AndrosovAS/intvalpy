import numpy as np
import itertools
from collections.abc import Sequence

from .RealInterval import Interval


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

    if isinstance(lst, (int, float)) or (len(lst) == 1 and isinstance(lst, Interval)):
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

    if isinstance(a, Interval):
        return a

    elif isinstance(a, (int, float)):
        return Interval(a, a, sortQ=False)

    elif isinstance(a, (list, np.ndarray)):
        a = np.asarray(a)
        shape = get_shape(a)

        result = zeros(shape)
        for index in itertools.product(*result.ranges):
            if isinstance(a[index], Interval):
                result[index] = a[index]
            else:
                result[index] = Interval(a[index], a[index], sortQ=False)
        return result

    else:
        msg = 'Входные данные неприводимы к интервальному массиву!'
        raise TypeError(msg)


def intersection(A, B):
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
                    выводится интервал Interval(float('-inf'), float('-inf')).
    """

    wA = asinterval(A)
    wB = asinterval(B)

    if wA.shape == wB.shape:
        result = zeros(wA.shape)
        for index in itertools.product(*wA.ranges):
            _max = wB[index].a if wA[index].a < wB[index].a else wA[index].a
            _min = wA[index].b if wA[index].b < wB[index].b else wB[index].b
            if _max <= _min:
                result[index] = Interval(_max, _min, sortQ=False)
            else:
                result[index] = Interval(float('-inf'), float('-inf'), sortQ=False)

    elif wA.shape == () or wA.shape == (1, ):
        result = zeros(wB.shape)
        for index in itertools.product(*wB.ranges):
            _max = wB[index].a if wA.a < wB[index].a else wA.a
            _min = wA.b if wA.b < wB[index].b else wB[index].b
            if _max <= _min:
                result[index] = Interval(_max, _min, sortQ=False)
            else:
                result[index] = Interval(float('-inf'), float('-inf'), sortQ=False)

    elif wB.shape == () or wB.shape == (1, ):
        result = zeros(wA.shape)
        for index in itertools.product(*wA.ranges):
            _max = wB.a if wA[index].a < wB.a else wA[index].a
            _min = wA[index].b if wA[index].b < wB.b else wB.b
            if _max <= _min:
                result[index] = Interval(_max, _min, sortQ=False)
            else:
                result[index] = Interval(float('-inf'), float('-inf'), sortQ=False)

    else:
        raise Exception('Не совпадают размерности входных массивов!')

    return result


def dist(a, b, order=float('inf')):
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

    def cheb(a, b):
        return max(abs(a.a-b.a), abs(a.b-b.b))

    if a.shape != b.shape:
        raise Exception('Размерности входных значений не совпадают!')

    if a.ndim > 2:
        raise Exception('Глубина входных значений не может быть больше двух!')
    elif a.ndim == 0:
        return cheb(a, b)

    if order == float('inf'):
        result = np.zeros(a.shape)
        for index in itertools.product(*a.ranges):
            result[index] = cheb(a[index], b[index])
        return np.amax(result)

    elif isinstance(order, int):
        result = 0
        for index in itertools.product(*a.ranges):
            result += cheb(a[index], b[index])**order
        return pow(result, 1/order)

    else:
        raise Exception('Не верно задан порядок нормы order.')


def zeros(shape):
    """Функция создаёт массив размерности shape."""
    return Interval(np.zeros(shape, dtype='float64'), \
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
    b = zeros(n)
    if infb is None:
        b._a = -np.ones(n)
    else:
        b._a += np.array([infb]*n)
    if supb is None:
        b._b = np.ones(n)
    else:
        b._b += np.array([supb]*n)

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
    return Interval(np.random.randint(inf, sup, shape), \
                    np.random.randint(inf, sup, shape))


def uniform(inf, sup, shape=1):
    return Interval(np.random.uniform(inf, sup, shape), \
                    np.random.uniform(inf, sup, shape))


def normal(mu, sigma, shape=1):
    return Interval(np.random.normal(mu, sigma, shape), \
                    np.random.normal(mu, sigma, shape))


def dot(a, b, aspotQ=False, bspotQ=False):
    if aspotQ:
        midb = b.mid
        radb = b.rad

        tmp1 = np.dot(a, midb)
        tmp2 = np.dot(abs(a), radb)

        return Interval(tmp1 - tmp2, tmp1 + tmp2, sortQ=False)

    elif bspotQ:
        mida = a.mid
        rada = a.rad

        tmp1 = np.dot(mida, b)
        tmp2 = np.dot(rada, abs(b))

        return Interval(tmp1 - tmp2, tmp1 + tmp2, sortQ=False)

    else:
        return a @ b
