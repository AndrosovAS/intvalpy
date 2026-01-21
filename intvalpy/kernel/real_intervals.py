import numpy as np

import math
from mpmath import mp, mpf
from numbers import Number

import itertools

from .interval_arithmetics import ARITHMETICS, ClassicalArithmetic, KaucherArithmetic


class ArrayInterval:

    def __init__(self, intervals):
        self._data = np.array(intervals, dtype=object)

        self._shape = self._data.shape
        self._ndim = self._data.ndim
        self._ranges = [range(0, self._shape[k]) for k in range(self._ndim)]

    def __repr__(self):
        return 'Interval' + self._data.__repr__()[5:-15] + ')'

    def __iter__(self):
        return self._data.__iter__()

    @property
    def a(self):
        return np.vectorize(lambda el: el.a)(self._data)

    @property
    def b(self):
        return np.vectorize(lambda el: el.b)(self._data)

    @property
    def inf(self):
        return np.vectorize(lambda el: el.a)(self._data)

    @property
    def sup(self):
        return np.vectorize(lambda el: el.b)(self._data)

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    @property
    def ranges(self):
        return self._ranges

    def to_float(self):
        return ArrayInterval(np.vectorize(lambda el: el.to_float())(self._data))

    @property
    def rad(self):
        return np.vectorize(lambda el: el.rad)(self._data)

    @property
    def mid(self):
        return np.vectorize(lambda el: el.mid)(self._data)

    @property
    def wid(self):
        return np.vectorize(lambda el: el.wid)(self._data)

    @property
    def mig(self):
        return np.vectorize(lambda el: el.mig)(self._data)

    @property
    def mag(self):
        return np.vectorize(lambda el: el.mag)(self._data)

    @property
    def dual(self):
        return ArrayInterval(np.vectorize(lambda el: el.dual)(self._data))

    @property
    def pro(self):
        return ArrayInterval(np.vectorize(lambda el: el.pro)(self._data))

    @property
    def opp(self):
        return ArrayInterval(np.vectorize(lambda el: el.opp)(self._data))

    @property
    def inv(self):
        return ArrayInterval(np.vectorize(lambda el: el.inv)(self._data))

    @property
    def khi(self):
        return np.vectorize(lambda el: el.khi)(self._data)

    @property
    def vertex(self):
        inf, sup = self.a, self.b
        if self._ndim == 1:
            n = self._shape[0]
            result = np.zeros((2**n, n))
            k = 0
            for ends in itertools.product([0, 1], repeat=n):
                ends = np.array(ends)
                result[k][ends == 0] = inf[ends == 0]
                result[k][ends == 1] = sup[ends == 1]
                k += 1

        elif self._ndim == 2:
            n, m = self._shape
            result = np.zeros((2**(n*m), n, m))
            k = 0
            for ends in itertools.product(itertools.product([0, 1], repeat=m), repeat=n):
                ends = np.array(ends)
                result[k][ends == 0] = inf[ends == 0]
                result[k][ends == 1] = sup[ends == 1]
                k += 1
        else:
            raise Exception('The function is provided for no more than two-dimensional arrays.')

        return result

    def copy(self):
        return ArrayInterval(np.copy(self._data))

    def __deepcopy__(self, memo):
        return ArrayInterval(np.copy(self._data))

    def __array__(self):
        return self._data

    @property
    def T(self):
        return ArrayInterval(self._data.T)

    def reshape(self, shape):
        return ArrayInterval(self._data.reshape(shape))

    def __neg__(self):
        return ArrayInterval(-self._data)

    def __reversed__(self):
        return iter(self.storage[::-1])

    def __abs__(self):
        """Range of absolute value."""
        return ArrayInterval(np.vectorize(abs)(self._data))

    def __contains__(self, other):
        if isinstance(other, single_type) or isinstance(other, (ClassicalArithmetic, KaucherArithmetic)):
            for index in itertools.product(*self._ranges):
                contain = other in self._data[index]
                if contain:
                    return contain
        else:
            for index in itertools.product(*self._ranges):
                contain = other[index] in self._data[index]
                if contain:
                    return contain
        return contain

    def __it__(self, other):
        if isinstance(other, ArrayInterval):
            return self._data < other._data
        else:
            return self._data < other

    def __le__(self, other):
        if isinstance(other, ArrayInterval):
            return self._data <= other._data
        else:
            return self._data <= other

    def __eq__(self, other):
        if isinstance(other, ArrayInterval):
            return self._data == other._data
        else:
            return self._data == other

    def __ne__(self, other):
        if isinstance(other, ArrayInterval):
            return self._data != other._data
        else:
            return self._data != other

    def __gt__(self, other):
        if isinstance(other, ArrayInterval):
            return self._data > other._data
        else:
            return self._data > other

    def __lt__(self, other):
        if isinstance(other, ArrayInterval):
            return self._data < other._data
        else:
            return self._data < other

    def __ge__(self, other):
        if isinstance(other, ArrayInterval):
            return self._data >= other._data
        else:
            return self._data >= other

    def __len__(self):
        return self._data.__len__()

    def __getitem__(self, key):
        result = self._data[key]
        if isinstance(result, INTERVAL_CLASSES):
            return result
        else:
            return ArrayInterval(result)

    def __setitem__(self, key, value):
        if isinstance(value, ARITHMETICS):
            self._data[key] = value
        elif isinstance(value, ArrayInterval):
            self._data[key] = value._data
        else:
            if isinstance(value, single_type):
                self._data[key] = SingleInterval(value, value)
            else:
                self._data[key] = ArrayInterval(np.vectorize(lambda v: SingleInterval(v, v))(value))._data

    def __delitem__(self, key):
        if isinstance(key, int):
            self._data = np.delete(self._data, key, axis=0)
            self._shape = self._data.shape
            self._ndim = self._data.ndim
            self._ranges = [range(0, self._shape[k]) for k in range(self._ndim)]
        elif isinstance(key, (slice, tuple, list, np.ndarray)):
            self._data = np.delete(self._data, key[-1], axis=len(key)-1)
            self._shape = self._data.shape
            self._ndim = self._data.ndim
            self._ranges = [range(0, self._shape[k]) for k in range(self._ndim)]
        else:
            msg = 'Indices must be integers / slice / tuple / list / np.ndarray'
            raise TypeError(msg)

    def __add__(self, other):
        if isinstance(other, ARITHMETICS):
            return self._data + other
        elif isinstance(other, ArrayInterval):
            return ArrayInterval(self._data + other._data)
        else:
            return ArrayInterval(self._data + other)

    def __sub__(self, other):
        if isinstance(other, ARITHMETICS):
            return self._data - other
        elif isinstance(other, ArrayInterval):
            return ArrayInterval(self._data - other._data)
        else:
            return ArrayInterval(self._data - other)

    def __mul__(self, other):
        if isinstance(other, ARITHMETICS):
            return self._data * other
        elif isinstance(other, ArrayInterval):
            return ArrayInterval(self._data * other._data)
        else:
            return ArrayInterval(self._data * other)

    def __truediv__(self, other):
        if isinstance(other, ARITHMETICS):
            return self._data / other
        elif isinstance(other, ArrayInterval):
            return ArrayInterval(self._data / other._data)
        else:
            return ArrayInterval(self._data / other)

    def __pow__(self, other):
        return ArrayInterval(np.vectorize(pow)(self._data, other))

    def __matmul__(self, other):
        matmul = self._data @ other
        if isinstance(matmul, INTERVAL_CLASSES):
            return matmul
        else:
            return ArrayInterval(matmul)

    def __radd__(self, other):
        return ArrayInterval(self._data + other)

    def __rsub__(self, other):
        return ArrayInterval(other - self._data)

    def __rmul__(self, other):
        return ArrayInterval(self._data * other)

    def __rtruediv__(self, other):
        return ArrayInterval(1 / self._data).__rmul__(other)

    def __rmatmul__(self, other):
        matmul = np.array(other) @ self._data
        if isinstance(matmul, INTERVAL_CLASSES):
            return matmul
        else:
            return ArrayInterval(matmul)

    def __iadd__(self, other):
        self._data = np.array(self._data + other)
        return self

    def __isub__(self, other):
        self._data = np.array(self._data - other)
        return self

    def __imul__(self, other):
        self._data = np.array(self._data * other)
        return self

    def __itruediv__(self, other):
        self._data = np.array(self._data / other)
        return self

    def __array_ufunc__(self, *args):
        if args[0].__name__ in ['add']:
            return ArrayInterval(self._data + args[2])

        elif args[0].__name__ in ['subtract']:
            return ArrayInterval(args[2] - self._data)

        elif args[0].__name__ in ['multiply']:
            return ArrayInterval(args[2] * self._data)

        elif args[0].__name__ in ['true_divide', 'divide']:
            return ArrayInterval(args[2] / self._data)

        elif args[0].__name__ in ['matmul']:
            matmul = args[2] @ self._data
            if isinstance(matmul, INTERVAL_CLASSES):
                return matmul
            else:
                return ArrayInterval(matmul)

        elif args[0].__name__ in ['sqrt']:
            return ArrayInterval(np.vectorize(np.sqrt)(self._data))

        elif args[0].__name__ in ['exp']:
            return ArrayInterval(np.vectorize(np.exp)(self._data))

        elif args[0].__name__ in ['log']:
            return ArrayInterval(np.vectorize(np.log)(self._data))

        elif args[0].__name__ in ['sin']:
            return ArrayInterval(np.vectorize(np.sin)(self._data))

        elif args[0].__name__ in ['cos']:
            return ArrayInterval(np.vectorize(np.cos)(self._data))

        else:
            raise Exception("Calculation of the {} function is not provided!".format(args[0].__name__))


class precision:
    extendedPrecisionQ = False
    mp.dps = 36

    def dps(_dps):
        mp.dps = _dps


def SingleInterval(left, right, sortQ=True, midRadQ=False):
    assert isinstance(left, Number), 'The left end of the interval should be a number.'
    assert isinstance(right, Number), 'The right end of the interval should be a number.'
    
    # if precision.extendedPrecisionQ:
    #     left, right = mpf(str(left)), mpf(str(right))
    # else:
    #     left, right = np.float64(left), np.float64(right)

    if midRadQ:
        assert right >= 0, "The radius of the interval cannot be negative."
        left, right = left - right, left + right

    ##############################
    # if left == right:
        # return left

    if sortQ:
        if left > right:
            return ClassicalArithmetic(right, left)
        else:
            return ClassicalArithmetic(left, right)
    else:
        if left <= right:
            return ClassicalArithmetic(left, right)
        else:
            return KaucherArithmetic(left, right)

# single_type = (int, float, np.int_, np.float_, mpf)
single_type = (Number, mpf)
ARITHMETICS = (ClassicalArithmetic, KaucherArithmetic)
INTERVAL_CLASSES = (ClassicalArithmetic, KaucherArithmetic, ArrayInterval)


def Interval(*args, sortQ=True, midRadQ=False):
    '''
    Construct an interval or array of intervals from input arguments.
    
    This function creates either SingleInterval or ArrayInterval objects
    depending on the input format. It supports multiple input representations
    and automatically handles bound ordering.
    
    Parameters:
    -----------
    *args : variable
        Input arguments defining the interval(s). Can be:
        - Two numbers (lower, upper bound)
        - A single number (degenerate interval)
        - A sequence of two numbers [lower, upper]
        - An existing interval object
        - Array-like inputs for vectorized operations
        - Nested sequence for the standard interval vector representation
    
    sortQ : bool, optional
        If True (default), automatically sorts bounds to ensure a ≤ b.
        If False, preserves original order (may create improper intervals).
    
    midRadQ : bool, optional
        If True, interprets inputs as midpoint-radius representation.
        If False (default), interprets as lower-upper bounds.
    
    Returns:
    --------
    Interval object
        Returns either:
        - SingleInterval for scalar inputs
        - ArrayInterval for array-like inputs
    
    Raises:
    -------
    AssertionError
        If more than two main arguments are provided
    
    Examples:
    ---------
    >>> Interval(2, 5)                # Single interval [2, 5]
    >>> Interval([3, 7])              # Single interval from sequence
    >>> Interval(4)                   # Degenerate interval [4, 4]
    >>> Interval([1,2], [3,4])        # Array of intervals [1, 3] and [2, 4]
    >>> Interval([[1,3], [2,4]])      # Array of intervals [1, 3] and [3, 4]
    >>> Interval(x, y, midRadQ=True)  # Midpoint-radius interpretation
    '''

    n = len(args)
    assert n <= 2, "There cannot be more than two main arguments."

    if n == 2:
        if isinstance(args[0], single_type):
            return SingleInterval(args[0], args[1], sortQ=sortQ, midRadQ=midRadQ)
        left, right = np.array(args[0]), np.array(args[1])
        data = np.stack((left, right), axis=left.ndim)
    else:
        if isinstance(args[0], INTERVAL_CLASSES):
            return args[0]
        elif len(args[0]) == 2:
            if isinstance(args[0][0], single_type) and isinstance(args[0][1], single_type):
                return SingleInterval(args[0][0], args[0][1], sortQ=sortQ, midRadQ=midRadQ)
        elif isinstance(args[0], single_type):
            return SingleInterval(args[0], args[0], sortQ=sortQ, midRadQ=midRadQ)
        data = np.array(args[0])

    return ArrayInterval(np.vectorize(lambda l, r: SingleInterval(l, r, sortQ=sortQ, midRadQ=midRadQ))(data[..., 0], data[..., 1]))
