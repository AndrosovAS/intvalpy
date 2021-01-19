import numpy as np
import itertools
from collections.abc import Sequence

from .MyClass import Interval


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

    elif wA.shape == ():
        result = zeros(wB.shape)
        for index in itertools.product(*wB.ranges):
            _max = wB[index].a if wA.a < wB[index].a else wA.a
            _min = wA.b if wA.b < wB[index].b else wB[index].b
            if _max <= _min:
                result[index] = Interval(_max, _min, sortQ=False)
            else:
                result[index] = Interval(float('-inf'), float('-inf'), sortQ=False)

    elif wB.shape == ():
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


# @njit(fastmath=True)
def create_data(shape, distribution='normal'):
    if distribution is 'normal':
        rand = lambda shape: np.random.normal(0, 1, shape)
    elif distribution is 'randint':
        rand = lambda shape: np.random.randint(-8, 8, shape)
    else:
        raise Exception('Неверно задано распределение.')

    return Interval(rand(shape), rand(shape))