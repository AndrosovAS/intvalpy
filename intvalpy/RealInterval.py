import numpy as np

import mpmath
import math
from mpmath import mp, mpf

import itertools


def wrap_to_interval(func):
    def interval(x, y):
        if isinstance(y, ARITHMETICS):
            return func(x, y)
        else:
            return func(x, SingleInterval(y, y))
    return interval


class BaseTools(object):

    def __init__(self, left, right):
        self._a, self._b = left, right

    @property
    def a(self):
        """
        The largest number that is less than or equal to each of a given
        set of real numbers of an interval.
        """
        return self._a

    @property
    def b(self):
        """
        The smallest number that is greater than or equal to each of a given
        set of real numbers of an interval.
        """
        return self._b

    @property
    def inf(self):
        """
        The largest number that is less than or equal to each of a given
        set of real numbers of an interval.
        """
        return self._a

    @property
    def sup(self):
        """
        The smallest number that is greater than or equal to each of a given
        set of real numbers of an interval.
        """
        return self._b

    @property
    def copy(self):
        return type(self)(self._a, self._b)

    def to_float(self):
        return type(self)(np.float64(self._a), np.float64(self._b))

    @property
    def wid(self):
        """Width of the non-empty interval."""
        return abs(self._b - self._a)

    @property
    def rad(self):
        """Radius of the non-empty interval."""
        return 1/2 * self.wid

    @property
    def mid(self):
        """Midpoint of the non-empty interval."""
        return 1/2 * (self._b + self._a)

    @property
    def mig(self):
        """The smallest absolute value in the non-empty interval."""
        if 0 in self:
            return 0.0
        else:
            return min(abs(self._a), abs(self._b))

    @property
    def mag(self):
        """The greatest absolute value in the non-empty interval."""
        return max(abs(self._a), abs(self._b))

    @property
    def dual(self):
        if self._b <= self._a:
            return ClassicalArithmetic(self._b, self._a)
        else:
            return KaucherArithmetic(self._b, self._a)

    @property
    def pro(self):
        if isinstance(self, ClassicalArithmetic):
            return ClassicalArithmetic(self._a, self._b)
        else:
            return ClassicalArithmetic(min(self._a, self._b), max(self._a, self._b))

    @property
    def opp(self):
        if self._b <= self._a:
            return ClassicalArithmetic(-self._a, -self._b)
        else:
            return KaucherArithmetic(-self._a, -self._b)

    @property
    def inv(self):
        if self._b <= self._a:
            return ClassicalArithmetic(1/self._a, 1/self._b)
        else:
            return KaucherArithmetic(1/self._a, 1/self._b)

    @property
    def khi(self):
        if abs(self._a) <= abs(self._b):
            return self._a / self._b
        else:
            return self._b / self._a


    def __repr__(self):
        return "'[%.6g, %.6g]'" % (self._a, self._b)

    # Unary operation
    def __neg__(self):
        return type(self)(-self._b, -self._a)

    def __abs__(self):
        """Range of absolute value."""
        return type(self)(self.mig, self.mag)

    @wrap_to_interval
    def __contains__(self, other):
        return (self.a <= other.a) and (other.b <= self.b)

    @wrap_to_interval
    def __it__(self, other):
        return (self.a < other.a) and (self.b < other.b)

    @wrap_to_interval
    def __le__(self, other):
        return (self.a <= other.a) and (self.b <= other.b)

    @wrap_to_interval
    def __eq__(self, other):
        return (self.a == other.a) and (self.b == other.b)

    @wrap_to_interval
    def __ne__(self, other):
        return (self.a != other.a) or (self.b != other.b)

    @wrap_to_interval
    def __gt__(self, other):
        return (self.a > other.a) and (self.b > other.b)

    @wrap_to_interval
    def __ge__(self, other):
        return (self.a >= other.a) and (self.b >= other.b)

    def __array_ufunc__(self, *args):
        cls = type(self)
        if args[0].__name__ in ['add']:
            return self.__radd__(args[2])

        elif args[0].__name__ in ['subtract']:
            return cls(-self._b, -self._a).__radd__(args[2])

        elif args[0].__name__ in ['multiply']:
            return self.__rmul__(args[2])

        elif args[0].__name__ in ['true_divide']:
            return cls(1/self._b, 1/self._a).__rmul__(args[2])

        elif args[0].__name__ in ['sqrt']:
            return np.exp(1/2 * np.log(self))

        elif args[0].__name__ in ['exp']:
            try:
                return cls(math.exp(self._a), math.exp(self._b))
            except OverflowError:
                return cls(float('inf'), float('inf'))

        elif args[0].__name__ in ['log']:
            pro = self.pro
            _max, _min = max(pro.a, 0), pro.b
            if _max <= _min:
                inf = float('-inf') if _max==0 else math.log(_max)
                sup = math.log(_min)
                return cls(inf, sup) if self._a <= self._b else cls(sup, inf)
            else:
                return cls(float('nan'), float('nan'))

        elif args[0].__name__ in ['sin']:
            x = self.pro
            sin_inf, sin_sup = math.sin(x.a), math.sin(x.b)

            if math.ceil((x.a + math.pi / 2) / (2 * math.pi)) <= math.floor((x.b + math.pi / 2) / (2 * math.pi)):
                inf = -1.0
            else:
                inf = min(sin_inf, sin_sup)

            if math.ceil((x.a - math.pi / 2) / (2 * math.pi)) <= math.floor((x.b - math.pi / 2) / (2 * math.pi)):
                sup = 1.0
            else:
                sup = max(sin_inf, sin_sup)

            if self._a <= self._b:
                return cls(inf, sup)
            else:
                return cls(sup, inf)

        elif args[0].__name__ in ['cos']:
            x = self.pro
            cos_inf, cos_sup = math.cos(x.a), math.cos(x.b)

            if math.ceil((x.a - math.pi) / (2 * math.pi)) <= math.floor((x.b - math.pi) / (2 * math.pi)):
                inf = -1.0
            else:
                inf = min(cos_inf, cos_sup)

            if math.ceil(x.a / (2 * math.pi)) <= math.floor(x.b / (2 * math.pi)):
                sup = 1.0
            else:
                sup = max(cos_inf, cos_sup)

            if self._a <= self._b:
                return cls(inf, sup)
            else:
                return cls(sup, inf)

        else:
            raise Exception("Calculation of the {} function is not provided!".format(args[0].__name__))


class ClassicalArithmetic(BaseTools):

    def __add__(self, other):
        if isinstance(other, ARITHMETICS):
            return type(other)(self._a + other._a, self._b + other._b)
        else:
            return other + self

    def __sub__(self, other):
        if isinstance(other, ARITHMETICS):
            return type(other)(self._a - other._b, self._b - other._a)
        else:
            return -other + self

    def __mul__(self, other):
        if isinstance(other, ClassicalArithmetic):
            mul = np.array([self._a*other._a,
                            self._a*other._b,
                            self._b*other._a,
                            self._b*other._b])
            return ClassicalArithmetic(np.min(mul), np.max(mul))

        elif isinstance(other, KaucherArithmetic):
            _selfInfPlus, _selfSupPlus = max(self._a, 0), max(self._b, 0)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = max(other.a, 0), max(other.b, 0)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other.a, _otherSupPlus - other.b

            return KaucherArithmetic(_selfInfPlus*_otherInfPlus + _selfSupMinus*_otherSupMinus - _selfSupPlus*_otherInfMinus - _selfInfMinus*_otherSupPlus, \
                                    _selfSupPlus*_otherSupPlus + _selfInfMinus*_otherInfMinus - _selfInfPlus*_otherSupMinus - _selfSupMinus*_otherInfPlus)

        else:
            return other * self

    def __truediv__(self, other):

        if isinstance(other, ClassicalArithmetic):
            assert not (0 in other), 'It is impossible to divide by zero containing intervals!'

            div = np.array([self._a/other._a,
                            self._a/other._b,
                            self._b/other._a,
                            self._b/other._b])
            return ClassicalArithmetic(div.min(), div.max())

        if isinstance(other, KaucherArithmetic):
            assert not (0 in other), 'It is impossible to divide by zero containing intervals!'
            return self.__mul__(KaucherArithmetic(1/other.b, 1/other.a))

        else:
            return 1/other * self

    def __pow__(self, other):
        if isinstance(other, (int, np.int_)) and other >= 0:
            inf, sup = self._a**other, self._b**other
            if other % 2 == 0 and 0 in self:
                return ClassicalArithmetic(0, max(inf, sup))
            else:
                return ClassicalArithmetic(min(inf, sup), max(inf, sup))
        elif self >= 0:
            return np.exp(other * np.log(self))
        else:
            raise ValueError('If the base contains negative numbers, than the degree can only be a natural number.')

    def __radd__(self, other):
        if isinstance(other, single_type):
            return ClassicalArithmetic(self._a + other, self._b + other)
        else:
            return ArrayInterval(np.vectorize(lambda o: o + self)(other))

    def __rsub__(self, other):
        if isinstance(other, single_type):
            return ClassicalArithmetic(other - self._b, other - self._a)
        else:
            return ArrayInterval(np.vectorize(lambda o: o - self)(other))

    def __rmul__(self, other):
        def fmul(x, y):
            mul = np.array([x*y.a, x*y.b])
            return ClassicalArithmetic(min(mul), max(mul))

        if isinstance(other, single_type):
            return fmul(other, self)
        else:
            return ArrayInterval(np.vectorize(lambda o: o * self)(other))

    def __rtruediv__(self, other):
        assert not (0 in self), 'It is impossible to divide by zero containing intervals!'
        return other * ClassicalArithmetic(1/self._b, 1/self._a)

    def __iadd__(self, other):
        other = Interval(other)
        if isinstance(other, ClassicalArithmetic):
            self._a, self._b = self._a + other._a, self._b + other._b
            return self
        else:
            raise SyntaxError("It is not possible to change the type of the variable.")

    def __isub__(self, other):
        other = Interval(other)
        if isinstance(other, ClassicalArithmetic):
            self._a, self._b = self._a - other._b,  self._b - other._a
            return self
        else:
            raise SyntaxError("It is not possible to change the type of the variable.")

    def __imul__(self, other):
        other = Interval(other)
        if isinstance(other, ClassicalArithmetic):
            mul = np.array([self._a*other._a,
                            self._a*other._b,
                            self._b*other._a,
                            self._b*other._b])
            self._a, self._b = mul.min(), mul.max()
            return self
        else:
            raise SyntaxError("It is not possible to change the type of the variable.")

    def __itruediv__(self, other):
        other = Interval(other)
        if isinstance(other, ClassicalArithmetic):
            assert not (0 in other), 'It is impossible to divide by zero containing intervals!'

            div = np.array([self._a/other._a,
                            self._a/other._b,
                            self._b/other._a,
                            self._b/other._b])
            self._a, self._b = div.min(), div.max()
            return self
        else:
            raise SyntaxError("It is not possible to change the type of the variable.")


class KaucherArithmetic(BaseTools):

    def __add__(self, other):
        if isinstance(other, ARITHMETICS):
            return KaucherArithmetic(self._a + other.a, self._b + other.b)
        else:
            return other + self

    def __sub__(self, other):
        if isinstance(other, ARITHMETICS):
            return KaucherArithmetic(self._a - other.b, self._b - other.a)
        else:
            return -other + self

    def __mul__(self, other):
        if isinstance(other, ARITHMETICS):
            _selfInfPlus, _selfSupPlus = max(self._a, 0), max(self._b, 0)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = max(other.a, 0), max(other.b, 0)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other.a, _otherSupPlus - other.b

            return KaucherArithmetic(max(_selfInfPlus * _otherInfPlus, _selfSupMinus * _otherSupMinus) - max(_selfSupPlus * _otherInfMinus, _selfInfMinus * _otherSupPlus), \
                                    max(_selfSupPlus * _otherSupPlus, _selfInfMinus * _otherInfMinus) - max(_selfInfPlus * _otherSupMinus, _selfSupMinus * _otherInfPlus))

        else:
            return other * self

    def __truediv__(self, other):
        if isinstance(other, ClassicalArithmetic):
            assert not (0 in other), 'It is impossible to divide by zero containing intervals!'

            other = ClassicalArithmetic(1/other.b, 1/other.a)
            _selfInfPlus, _selfSupPlus = max(self._a, 0), max(self._b, 0)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = max(other.a, 0), max(other.b, 0)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other.a, _otherSupPlus - other.b

            return KaucherArithmetic(_selfInfPlus*_otherInfPlus + _selfSupMinus*_otherSupMinus - _selfSupPlus*_otherInfMinus - _selfInfMinus*_otherSupPlus, \
                                    _selfSupPlus*_otherSupPlus + _selfInfMinus*_otherInfMinus - _selfInfPlus*_otherSupMinus - _selfSupMinus*_otherInfPlus)

        if isinstance(other, KaucherArithmetic):
            assert not (0 in other), 'It is impossible to divide by zero containing intervals!'
            return self.__mul__(KaucherArithmetic(1/other.b, 1/other.a))

        else:
            return 1/other * self

    def __pow__(self, other):
        if isinstance(other, (int, np.int_)) and other >= 0:
            inf, sup = self._a**other, self._b**other
            if other % 2 == 0 and 0 in self:
                return KaucherArithmetic(0, max(inf, sup))
            else:
                return KaucherArithmetic(inf, sup)
        elif self >= 0:
            return np.exp(other * np.log(self))
        else:
            raise ValueError('If the base contains negative numbers, than the degree can only be a natural number.')

    def __radd__(self, other):
        if isinstance(other, single_type):
            return KaucherArithmetic(self._a + other, self._b + other)
        else:
            return ArrayInterval(np.vectorize(lambda o: o + self)(other))

    def __rsub__(self, other):
        if isinstance(other, single_type):
            return KaucherArithmetic(other - self._b, other - self._a)
        else:
            return ArrayInterval(np.vectorize(lambda o: o - self)(other))

    def __rmul__(self, other):
        def fmul(x, y):
            _selfInfPlus, _selfSupPlus = max(y.a, 0), max(y.b, 0)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - y.a, _selfSupPlus - y.b

            _otherInfPlus = max(x, 0)
            _otherSupPlus = _otherInfPlus
            _otherInfMinus, _otherSupMinus = _otherInfPlus - x, _otherSupPlus - x

            return KaucherArithmetic(_selfInfPlus*_otherInfPlus + _selfSupMinus*_otherSupMinus - _selfSupPlus*_otherInfMinus - _selfInfMinus*_otherSupPlus, \
                        _selfSupPlus*_otherSupPlus + _selfInfMinus*_otherInfMinus - _selfInfPlus*_otherSupMinus - _selfSupMinus*_otherInfPlus)

        if isinstance(other, single_type):
            return fmul(other, self)
        else:
            return ArrayInterval(np.vectorize(lambda o: o * self)(other))

    def __rtruediv__(self, other):
        assert not (0 in self), 'It is impossible to divide by zero containing intervals!'
        return other * KaucherArithmetic(1/self._b, 1/self._a)

    def __iadd__(self, other):
        other = Interval(other)
        if isinstance(other, ARITHMETICS):
            self._a, self._b = self._a + other.a, self._b + other.b
            return self
        else:
            raise SyntaxError("It is not possible to change the type of the variable.")

    def __isub__(self, other):
        other = Interval(other)
        if isinstance(other, ARITHMETICS):
            self._a, self._b = self._a - other.b,  self._b - other.a
            return self
        else:
            raise SyntaxError("It is not possible to change the type of the variable.")

    def __imul__(self, other):
        other = Interval(other)
        if isinstance(other, ARITHMETICS):
            _selfInfPlus, _selfSupPlus = max(self._a, 0), max(self._b, 0)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = max(other.a, 0), max(other.b, 0)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other.a, _otherSupPlus - other.b

            self._a = max(_selfInfPlus * _otherInfPlus, _selfSupMinus * _otherSupMinus) - max(_selfSupPlus * _otherInfMinus, _selfInfMinus * _otherSupPlus)
            self._b = max(_selfSupPlus * _otherSupPlus, _selfInfMinus * _otherInfMinus) - max(_selfInfPlus * _otherSupMinus, _selfSupMinus * _otherInfPlus)
            return self

        else:
            raise SyntaxError("It is not possible to change the type of the variable.")

    def __itruediv__(self, other):
        other = Interval(other)
        if isinstance(other, ClassicalArithmetic):
            assert (not 0 in other), 'It is impossible to divide by zero containing intervals!'

            other = ClassicalArithmetic(1/other.b, 1/other.a)
            _selfInfPlus, _selfSupPlus = max(self._a, 0), max(self._b, 0)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = max(other.a, 0), max(other.b, 0)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other.a, _otherSupPlus - other.b

            self._a = _selfInfPlus*_otherInfPlus + _selfSupMinus*_otherSupMinus - _selfSupPlus*_otherInfMinus - _selfInfMinus*_otherSupPlus
            self._b = _selfSupPlus*_otherSupPlus + _selfInfMinus*_otherInfMinus - _selfInfPlus*_otherSupMinus - _selfSupMinus*_otherInfPlus
            return self

        if isinstance(other, KaucherArithmetic):
            assert not (0 in other), 'It is impossible to divide by zero containing intervals!'

            div = self.__mul__(KaucherArithmetic(1/other.b, 1/other.a))
            self._a, self._b = div._a, div._b
            return self

        else:
            raise SyntaxError("It is not possible to change the type of the variable.")


class ArrayInterval:

    def __init__(self, intervals):
        self._data = intervals

        self._shape = self._data.shape
        self._ndim = self._data.ndim
        self._ranges = [range(0, self._shape[k]) for k in range(self._ndim)]

    def __repr__(self):
        return 'Interval' + self._data.__repr__()[5:-15] + ')'

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

    @property
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

        elif args[0].__name__ in ['true_divide']:
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
    increasedPrecisionQ = True
    mp.dps = 36

    def dps(_dps):
        mp.dps = _dps


def SingleInterval(left, right, sortQ=True, midRadQ=False):
    if precision.increasedPrecisionQ:
        left, right = mpf(str(left)), mpf(str(right))
    else:
        left, right = np.float64(left), np.float64(right)

    if midRadQ:
        assert right >= 0, "The radius of the interval cannot be negative."
        left, right = left - right, left + right

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

single_type = (int, float, np.int_, np.float_, mpf)
ARITHMETICS = (ClassicalArithmetic, KaucherArithmetic)
INTERVAL_CLASSES = (ClassicalArithmetic, KaucherArithmetic, ArrayInterval)


def Interval(*args, sortQ=True, midRadQ=False):

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
        elif isinstance(args[0], single_type):
            return SingleInterval(args[0], args[0], sortQ=sortQ, midRadQ=midRadQ)
        data = np.array(args[0])

    return ArrayInterval(np.vectorize(lambda l, r: SingleInterval(l, r, sortQ=sortQ, midRadQ=midRadQ))(data[..., 0], data[..., 1]))
