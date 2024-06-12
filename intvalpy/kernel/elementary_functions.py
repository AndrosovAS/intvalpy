import numpy as np

from .preprocessing import asinterval
from.real_intervals import Interval, ARITHMETICS


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
    x = asinterval(x)
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
    x = asinterval(x)
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
    x = asinterval(x)
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
    x = asinterval(x)
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
    x = asinterval(x)
    return np.cos(x)


def sign(x):
    def _sign(x):
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

    x = asinterval(x)
    if isinstance(x, ARITHMETICS):
        return _sign(x)
    else:
        return asinterval(np.vectorize(_sign)(x.data))