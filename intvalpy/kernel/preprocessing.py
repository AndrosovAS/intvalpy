import numpy as np

import itertools
from collections.abc import Sequence

from .real_intervals import Interval, ArrayInterval, single_type, INTERVAL_CLASSES, ARITHMETICS
from .new_objects import zeros

infinity = float('inf')
nan = np.nan

#############################################################################################################
#############################################################################################################


def unique(a, decimals=12):
    a = np.ascontiguousarray(a)
    a = np.around(a, decimals=int(decimals))
    _, index = np.unique(a.view([('', a.dtype)]*a.shape[1]), return_index=True)
    index = sorted(index)
    return a[index]


def non_repeat(a, b):
    a = np.copy(np.ascontiguousarray(a))
    a = np.around(a, decimals=12)
    b = np.around(np.copy(b), decimals=12)
    a1 = (a.T - b).T
    _, index = np.unique(a1.view([('', a1.dtype)]*a1.shape[1]), return_index=True)
    index = sorted(index)
    return a[index], b[index]


def clear_zero_rows(a, b, ndim=2):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    a, b = np.around(a, decimals=12), np.around(b, decimals=12)

    cnmty = True
    if np.sum((np.sum(abs(a) <= 1e-12, axis=1) == ndim) & (b > 0)) > 0:
        cnmty = False

    index = np.where(np.sum(abs(a) <= 1e-12, axis=1) != ndim)
    return a[index], b[index], cnmty



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

    try:
        a = np.asarray(a)
        shape = get_shape(a)

        result = zeros(shape)
        for index in itertools.product(*result.ranges):
            if isinstance(a[index], INTERVAL_CLASSES):
                result[index] = a[index]
            else:
                result[index] = Interval(a[index], a[index], sortQ=False)
        return result

    except:
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