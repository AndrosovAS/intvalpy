import numbers
import numpy as np
import itertools

# from mpmath import *
from mpmath import mp, mpf, matrix
import sys


class precision:

    increasedPrecisionQ = True
    mp.dps = 32

    def dps(_dps):
        mp.dps = _dps

class BaseTools:

    @property
    def a(self):
        """Возвращает левый конец интервала."""
        return self._a

    @property
    def b(self):
        """Возвращает правый конец интервала."""
        return self._b

    @property
    def inf(self):
        """Возвращает левый конец интервала."""
        return self._a

    @property
    def sup(self):
        """Возвращает правый конец интервала."""
        return self._b

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    @property
    def ranges(self):
        return self._ranges

    @property
    def vertex(self):
        if self._ndim == 1:
            n = self._shape[0]
            result = np.zeros((2**n, n))
            k = 0
            for ends in itertools.product([0, 1], repeat=n):
                ends = np.array(ends)
                result[k][ends==0] = self._a[ends==0]
                result[k][ends==1] = self._b[ends==1]
                k += 1

        elif self._ndim == 2:
            n, m = self._shape
            result = np.zeros((2**(n*m), n, m))
            k = 0
            for ends in itertools.product(itertools.product([0, 1], repeat=m), repeat=n):
                ends = np.array(ends)
                result[k][ends==0] = self._a[ends==0]
                result[k][ends==1] = self._b[ends==1]
                k += 1
        else:
            raise Exception('Функция предусмотрена не более чем для двумерных массивов.')

        return result

    def __repr__(self):
        return self.__format__("%.6g")

    def __format__(self, fs):
        result = 'Interval(['
        if self._ndim > 1:
            result += '['
        i = 0
        for index in itertools.product(*self._ranges):
            if self._ndim > 1 and index[0] > i:
                result = result[:-2] + '],\n          ['
                i += 1
            result += "'[" + ', '.join(fs % x for x in [self._a[index], self._b[index]]) + "]', "

        return result[:-2] + ']])' if self._ndim > 1 else result[:-2] + '])'

    #     Эмуляция коллекций
    def __len__(self):
        return self._a.__len__()

    def __setitem__(self, key, value):
        if (isinstance(self, KaucherArithmetic) and isinstance(value, ARITHMETIC_TUPLE)) or \
        (isinstance(self, ClassicalArithmetic) and isinstance(value, ClassicalArithmetic)):
            self._a[key], self._b[key] = np.copy(value._a), np.copy(value._b)
        elif isinstance(self, ClassicalArithmetic) and isinstance(value, KaucherArithmetic):
            raise Exception('Объекты из разных арифметик!')
        else:
            if not precision.increasedPrecisionQ:
                self._a[key], self._b[key] = np.copy(value), np.copy(value)
            else:
                if isinstance(value, (matrix, mpf)):
                    self._a[key], self._b[key] = np.copy(value), np.copy(value)
                elif isinstance(value, (int, float)):
                    value = mpf(np.str(value))
                    self._a[key], self._b[key] = value, value
                else:
                    value = matrix(np.array(value, dtype=np.str))
                    self._a[key], self._b[key] = value, value


    def __delitem__(self, key):
        if isinstance(key, int):
            self._a = np.delete(self._a, key, axis=0)
            self._b = np.delete(self._b, key, axis=0)
            self._shape = self._a.shape
            self._ndim = self._a.ndim
            self._ranges = [range(0, self._shape[k]) for k in range(self._ndim)]

        elif isinstance(key, (slice, tuple, list, np.ndarray)):
            self._a = np.delete(self._a, key[-1], axis=len(key)-1)
            self._b = np.delete(self._b, key[-1], axis=len(key)-1)
            self._shape = self._a.shape
            self._ndim = self._a.ndim
            self._ranges = [range(0, self._shape[k]) for k in range(self._ndim)]

        else:
            msg = 'Indices must be integers / slice / tuple / list / np.ndarray'
            raise TypeError(msg)

    def __contains__(self, other):
        try:
            # s, o = self.pro, other.pro
            return ((self._a <= other._a) & (other._b <= self._b)).any()
        except:
            # s = self.pro
            return (np.array((self._a <= other)) & np.array((other <= self._b))).any()

    # Итерирование
    def __reversed__(self):
        return iter(self.storage[::-1])

    def __abs__(self):
        return np.maximum(np.abs(self._a), np.abs(self._b))

    #     Операторы сравнения
    def __it__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return np.array((self._a < other._a) & (self._b < other._b), dtype=np.bool)
        else:
            return np.array((self._a < other) & (self._b < other), dtype=np.bool)

    def __le__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return np.array((self._a <= other._a) & (self._b <= other._b), dtype=np.bool)
        else:
            return np.array((self._a <= other) & (self._b <= other), dtype=np.bool)

    def __eq__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return np.array((self._a == other._a) & (self._b == other._b), dtype=np.bool)
        else:
            return np.array((self._a == other) & (self._b == other), dtype=np.bool)

    def __ne__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return np.array((self._a != other._a) | (self._b != other._b), dtype=np.bool)
        else:
            return np.array((self._a != other) | (self._b != other), dtype=np.bool)

    def __gt__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return np.array((self._a > other._a) & (self._b > other._b), dtype=np.bool)
        else:
            return np.array((self._a > other) & (self._b > other), dtype=np.bool)

    def __ge__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return np.array((self._a >= other._a) & (self._b >= other._b), dtype=np.bool)
        else:
            return np.array((self._a >= other) & (self._b >= other), dtype=np.bool)

    # Характеристики интервала, такие как mid, rad и т.д.
    @property
    def rad(self):
        """Возвращает радиус интервала"""
        return 1/2 * (self._b - self._a)

    @property
    def wid(self):
        """Возвращает ширину интервала"""
        return self._b - self._a

    @property
    def mid(self):
        """Возвращает серидину интервала"""
        return (self._b + self._a)/2

    @property
    def mig(self):
        """Возвращает мигнитуду интервала"""
        if precision.increasedPrecisionQ:
            result = np.array(np.minimum(np.abs(self._a), np.abs(self._b)), dtype=np.object)
            result[self._a * self._b <= 0] = mpf('0')
        else:
            result = np.array(np.minimum(np.abs(self._a), np.abs(self._b)), dtype=np.float64)
            result[self._a * self._b <= 0] = 0
        return result

    def reshape(self, shape):
        if isinstance(self, ClassicalArithmetic):
            return ClassicalArithmetic(self._a.reshape(shape), self._b.reshape(shape), sortQ=False)
        else:
            return KaucherArithmetic(self._a.reshape(shape), self._b.reshape(shape))
        # return Interval(self._a.reshape(shape), self._b.reshape(shape), sortQ=False)

    @property
    def dual(self):
        if np.array(self._b <= self._a, dtype=np.bool).all():
            return ClassicalArithmetic(self._b, self._a, sortQ=False)
        else:
            return KaucherArithmetic(self._b, self._a)
        # return Interval(self._b, self._a, sortQ=False)

    @property
    def pro(self):
        if isinstance(self, ClassicalArithmetic):
            return ClassicalArithmetic(self._a, self._b, sortQ=False)
        else:
            return ClassicalArithmetic(self._a, self._b, sortQ=True)

    @property
    def opp(self):
        if np.array(self._b <= self._a, dtype=np.bool).all():
            return ClassicalArithmetic(-self._a, -self._b, sortQ=False)
        else:
            return KaucherArithmetic(-self._a, -self._b)
        # return Interval(-self._a, -self._b, sortQ=False)

    @property
    def inv(self):
        if np.array(self._b <= self._a, dtype=np.bool).all():
            return ClassicalArithmetic(1/self._a, 1/self._b, sortQ=False)
        else:
            return KaucherArithmetic(1/self._a, 1/self._b)
        # return Interval(1/self._a, 1/self._b, sortQ=False)

#########################################################################################################
#                  Согласование класса cls с другими библиотеками Python.                               #
#                                                                                                       #
#########################################################################################################

    #     Копирование объекта
    def __deepcopy__(self, memo):
        cls = type(self)
        return cls(np.copy(self._a), np.copy(self._b))

    # Преобразование в массив типа ndarray
    def asnumpy(self):
        return np.array([self[index] for index in itertools.product(*self._ranges)], dtype=np.object).reshape(self._shape)

    def __array__(self):
        return self.asnumpy()

    def _zeros(self, shape):
        if precision.increasedPrecisionQ:
            return np.array(np.zeros(shape) + mpf('0'), dtype=np.object)
        else:
            return np.zeros(shape)

    # Универсальная функция
    def __array_ufunc__(self, *args):
        cls = type(self)
        if args[0].__name__ in ['add']:
            return cls(self._a + args[2], self._b + args[2])

        elif args[0].__name__ in ['subtract']:
            return cls(args[2] - self._b, args[2] - self._a)

        elif args[0].__name__ in ['multiply']:
            tmp = np.array([args[2]*self._a,
                            args[2]*self._b])

            return cls(tmp.min(axis=0), tmp.max(axis=0))

        elif args[0].__name__ in ['true_divide']:
            if 0 in self:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')

            tmp = np.array([args[2]/self._a,
                            args[2]/self._b])
            return cls(tmp.min(axis=0), tmp.max(axis=0))

        elif args[0].__name__ in ['matmul']:
            return cls.__rmatmul__(self, args[2])

        elif args[0].__name__ in ['sqrt']:
            if np.array(self._a <= 0, dtype=np.bool).any() or np.array(self._b <= 0, dtype=np.bool).any():
                raise Exception("Невозможно взять квадратный корень из отрицательного числа!")
            return cls(np.sqrt(self._a), np.sqrt(self._b))

        elif args[0].__name__ in ['exp']:
            # return cls(args[0].__call__(self._a), args[0].__call__(self._b))
            return cls(np.e ** self._a, np.e ** self._b)

        elif args[0].__name__ in ['sin']:
            result = cls(self._zeros(self._shape), self._zeros(self._shape))
            kahan_index = self._a > self._b
            inf, sup = np.minimum(self._a, self._b), np.maximum(self._a, self._b)
            sin_inf, sin_sup = np.sin(np.array(inf, dtype=np.float64)), np.sin(np.array(sup, dtype=np.float64))

            sup_index = np.array(np.ceil((inf - (np.pi/2)) / (2*np.pi)) <= np.floor((sup - (np.pi/2)) / (2*np.pi)), dtype=np.bool)
            result._b[sup_index] = 1
            result._b[~sup_index] = np.maximum(sin_inf[~sup_index], sin_sup[~sup_index])

            inf_index = np.array(np.ceil((inf + (np.pi/2)) / (2*np.pi)) <= np.floor((sup + (np.pi/2)) / (2*np.pi)), dtype=np.bool)
            result._a[inf_index] = -1
            result._a[~inf_index] = np.minimum(sin_inf[~inf_index], sin_sup[~inf_index])

            try:
                result[kahan_index] = result[kahan_index].dual
            except:
                if kahan_index:
                    result = result.dual
            return result

        elif args[0].__name__ in ['cos']:
            result = cls(self._zeros(self._shape), self._zeros(self._shape))
            kahan_index = self._a > self._b
            inf, sup = np.minimum(self._a, self._b), np.maximum(self._a, self._b)
            cos_inf, cos_sup = np.cos(np.array(inf, dtype=np.float64)), np.cos(np.array(sup, dtype=np.float64))

            sup_index = np.array(np.ceil(2 * inf/np.pi) <= np.floor(2 * sup/np.pi), dtype=np.bool)
            result._b[sup_index] = 1
            result._b[~sup_index] = np.maximum(cos_inf[~sup_index], cos_sup[~sup_index])

            inf_index = np.array(np.ceil((inf - np.pi) / (2*np.pi)) <= np.floor((sup - np.pi) / (2*np.pi)), dtype=np.bool)
            result._a[inf_index] = -1
            result._a[~inf_index] = np.minimum(cos_inf[~inf_index], cos_sup[~inf_index])

            try:
                result[kahan_index] = result[kahan_index].dual
            except:
                if kahan_index:
                    result = result.dual
            return result

        elif args[0].__name__ in ['log']:
            if np.array((self._a <= 0) | (self._b <= 0), dtype=np.bool).any():
                raise Exception('Попытка вычислить логарифм от не положительного интервала!')
            else:
                return cls(np.log(np.array(self._a, dtype=np.float64)), np.log(np.array(self._b, dtype=np.float64)))

        else:
            raise Exception("Расчёт функции {} не предусмотрен!".format(args[0].__name__))


class ClassicalArithmetic(BaseTools):

    def __init__(self, left, right, sortQ=False):
        self._a, self._b = np.array(left), np.array(right)
        # self._a, self._b = left, right

        self._shape = self._a.shape
        self._ndim = self._a.ndim
        self._ranges = [range(0, self._shape[k]) for k in range(self._ndim)]

        if sortQ:
            self._a, self._b = np.array(np.minimum(self._a, self._b)), np.array(np.maximum(self._a, self._b))

    def __getitem__(self, key):
        if isinstance(key, (slice, numbers.Integral, tuple, list, np.ndarray)):
            # return ClassicalArithmetic(np.array(self._a[key]), np.array(self._b[key]), sortQ=False)
            return ClassicalArithmetic(self._a[key], self._b[key], sortQ=False)
        else:
            msg = 'Indices must be integers / slice / tuple / list / np.ndarray'
            raise TypeError(msg)

    # Унарные числовые операции
    def __neg__(self):
        return ClassicalArithmetic(-self._b, -self._a, sortQ=False)

    @property
    def copy(self):
        return ClassicalArithmetic(np.copy(self._a), np.copy(self._b), sortQ=False)

    @property
    def copyInKaucherArithmetic(self):
        return KaucherArithmetic(np.copy(self._a), np.copy(self._b))

    @property
    def T(self):
        return ClassicalArithmetic(self._a.T, self._b.T, sortQ=False)

    # Арифметические операторы
    def __add__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return type(other)(self._a + other._a, self._b + other._b)
        else:
            return ClassicalArithmetic(self._a + other, self._b + other)

    def __sub__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return type(other)(self._a - other._b, self._b - other._a)
        else:
            return ClassicalArithmetic(self._a - other, self._b - other)

    def __mul__(self, other):
        if isinstance(other, ClassicalArithmetic):
            tmp = np.array([self._a*other._a,
                            self._a*other._b,
                            self._b*other._a,
                            self._b*other._b])

            try:
                tmp = np.array([tmp[:, k] for k in range(tmp.shape[1])])
                return ClassicalArithmetic(tmp.min(axis=1), tmp.max(axis=1), sortQ=False)
            except:
                return ClassicalArithmetic(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

        elif isinstance(other, KaucherArithmetic):
            zero = self._zeros(self._shape)
            _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = np.maximum(other._a, zero), np.maximum(other._b, zero)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other._a, _otherSupPlus - other._b

            return KaucherArithmetic(_selfInfPlus*_otherInfPlus + _selfSupMinus*_otherSupMinus - _selfSupPlus*_otherInfMinus - _selfInfMinus*_otherSupPlus, \
                                    _selfSupPlus*_otherSupPlus + _selfInfMinus*_otherInfMinus - _selfInfPlus*_otherSupMinus - _selfSupMinus*_otherInfPlus)

        else:
            tmp = np.array([self._a*other,
                            self._b*other])
            return ClassicalArithmetic(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)


    def __truediv__(self, other):
        if isinstance(other, ClassicalArithmetic):
            if 0 in other:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            tmp = np.array([self._a/other._a,
                            self._a/other._b,
                            self._b/other._a,
                            self._b/other._b])
            try:
                tmp = np.array([tmp[:, k] for k in range(tmp.shape[1])])
                return ClassicalArithmetic(tmp.min(axis=1), tmp.max(axis=1), sortQ=False)
            except:
                return ClassicalArithmetic(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

        if isinstance(other, KaucherArithmetic):
            if 0 in other:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            return self.__mul__(KaucherArithmetic(1/other._b, 1/other._a))

        else:
            if 0 in np.asarray([other]):
                raise Exception('Нельзя делить на нуль!')
            tmp = np.array([self._a/other,
                            self._b/other])

            return ClassicalArithmetic(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

    def __pow__(self, other):
        if other == 1/2:
            return np.sqrt(self)
        elif (not isinstance(other, int)) or other < 0:
            raise Exception('Степень может быть только натуральным числом или равной 1/2.')

        if other % 2 == 0:
            index1 = np.array((self._a < 0) & (self._b <= 0), dtype=np.bool)
            index2 = np.array((self._a < 0) & (self._b >= 0), dtype=np.bool)
            index3 = np.array(~(index1 | index2), dtype=np.bool)

            left, right = self._zeros(self.shape), self._zeros(self.shape)

            left[index1], right[index1] = self[index1]._b**other, self[index1]._a**other
            right[index2] = abs(self[index2])**other
            left[index3], right[index3] = self[index3]._a**other, self[index3]._b**other
            return ClassicalArithmetic(left, right, sortQ=False)
        else:
            return ClassicalArithmetic(self._a**other, self._b**other, sortQ=False)

    def __matmul__(self, other):
        _ndim = (self._ndim, other.ndim)
        if _ndim == (2, 1) and self._shape[1] == other.shape[0]:
            return sum((self * other).T)

        elif self._shape == other.shape[::-1] and self._ndim == 2:
            n, _ = self._shape
            if isinstance(other, KaucherArithmetic):
                result = KaucherArithmetic(self._zeros((n, n)), self._zeros((n, n)))
            else:
                result = ClassicalArithmetic(self._zeros((n, n)), self._zeros((n, n)), sortQ=False)
            for k in range(n):
                result[k] = sum((self[k] * other.T).T)
            return result

        elif _ndim[0]*_ndim[1] <= 1:
            return sum(self * other)

        elif _ndim == (1, 2) and self._shape[0] == other.shape[0]:
            return sum((self * other.T).T)

        else:
            msg = 'Входные операнды имеют неправильную размерность.'
            raise TypeError(msg)


    # Инверсные арифметические операторы
    def __radd__(self, other):
        return ClassicalArithmetic(other + self._a, other + self._b, sortQ=False)

    def __rsub__(self, other):
        return ClassicalArithmetic(other - self._b, other - self._a, sortQ=False)

    def __rmul__(self, other):
        tmp = np.array([other*self._a,
                        other*self._b])
        return ClassicalArithmetic(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

    def __rtruediv__(self, other):
        if 0 in self:
            raise Exception('Нельзя делить на нуль содержащие интервалы!')

        tmp = np.array([other/self._a,
                        other/self._b])
        return ClassicalArithmetic(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

    def __rmatmul__(self, other):
        other = np.asarray(other)
        ndim = (other.ndim, self._ndim)
        if (ndim == (2, 1) and other.shape[1] == self._shape[0]) or \
           (self._shape == other.shape[::-1] and other.ndim == 2):
            om = other @ self.mid
            aor = np.abs(other) @ self.rad
            return ClassicalArithmetic(om-aor, om+aor, sortQ=False)

        elif ndim[0]*ndim[1] <= 1:
            return sum(self * other)

        elif ndim == (1, 2) and other.shape[0] == self._shape[0]:
            return sum((other * self).T)

        else:
            msg = 'Входные операнды имеют неправильную размерность.'
            raise TypeError(msg)


    #  Арифметические операторы присваивания
    def __iadd__(self, other):
        if isinstance(other, ClassicalArithmetic):
            self._a, self._b = self._a + other._a, self._b + other._b
        elif isinstance(other, KaucherArithmetic):
            raise Exception('Данная функция не поддерживает сложение с интервалами из разных арифметик.')
        else:
            self._a, self._b = self._a + other, self._b + other
        return self

    def __isub__(self, other):
        if isinstance(other, ClassicalArithmetic):
            self._a, self._b = self._a - other._b,  self._b - other._a
        elif isinstance(other, KaucherArithmetic):
            raise Exception('Данная функция не поддерживает вычитание с интервалами из разных арифметик.')
        else:
            self._a, self._b = self._a - other,  self._b - other
        return self

    def __imul__(self, other):
        if isinstance(other, ClassicalArithmetic):
            tmp = np.array([self._a*other._a,
                            self._a*other._b,
                            self._b*other._a,
                            self._b*other._b])
            try:
                tmp = np.array([tmp[:, k] for k in range(tmp.shape[1])])
                self._a, self._b = tmp.min(axis=1), tmp.max(axis=1)
            except:
                self._a, self._b = tmp.min(axis=0), tmp.max(axis=0)

        elif isinstance(other, KaucherArithmetic):
            raise Exception('Данная функция не поддерживает умножение с интервалами из разных арифметик.')

        else:
            tmp = np.array([self._a*other,
                            self._b*other])
            self._a, self._b = tmp.min(axis=0), tmp.max(axis=0)
        return self

    def __itruediv__(self, other):
        if isinstance(other, ClassicalArithmetic):
            if 0 in other:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            tmp = np.array([self._a/other._a,
                            self._a/other._b,
                            self._b/other._a,
                            self._b/other._b])
            try:
                tmp = np.array([tmp[:, k] for k in range(tmp.shape[1])])
                self._a, self._b = tmp.min(axis=1), tmp.max(axis=1)
            except:
                self._a, self._b = tmp.min(axis=0), tmp.max(axis=0)

        elif isinstance(other, KaucherArithmetic):
            raise Exception('Данная функция не поддерживает деление с интервалами из разных арифметик.')

        else:
            if 0 in np.asarray([other]):
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            tmp = np.array([self._a/other,
                            self._b/other])
            self._a, self._b = tmp.min(axis=0), tmp.max(axis=0)
        return self


class KaucherArithmetic(BaseTools):

    def __init__(self, left, right):
        self._a, self._b = np.array(left), np.array(right)
        # self._a, self._b = left, right

        self._shape = self._a.shape
        self._ndim = self._a.ndim
        self._ranges = [range(0, self._shape[k]) for k in range(self._ndim)]

    def __getitem__(self, key):
        if isinstance(key, (slice, numbers.Integral, tuple, list, np.ndarray)):
            # return KaucherArithmetic(np.array(self._a[key]), np.array(self._b[key]))
            return KaucherArithmetic(self._a[key], self._b[key])
        else:
            msg = 'Indices must be integers / slice / tuple / list / np.ndarray'
            raise TypeError(msg)

    #     Унарные числовые операции
    def __neg__(self):
        return KaucherArithmetic(-self._b, -self._a)

    @property
    def copy(self):
        return KaucherArithmetic(np.copy(self._a), np.copy(self._b))

    @property
    def T(self):
        return KaucherArithmetic(self._a.T, self._b.T)

    #     Арифметические операторы
    def __add__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return KaucherArithmetic(self._a + other._a, self._b + other._b)
        else:
            return KaucherArithmetic(self._a + other, self._b + other)


    def __sub__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return KaucherArithmetic(self._a - other._b, self._b - other._a)
        else:
            return KaucherArithmetic(self._a - other, self._b - other)


    def __mul__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            zero = self._zeros(self._shape)
            _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = np.maximum(other._a, zero), np.maximum(other._b, zero)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other._a, _otherSupPlus - other._b

            return KaucherArithmetic(np.maximum(_selfInfPlus * _otherInfPlus, _selfSupMinus * _otherSupMinus) - np.maximum(_selfSupPlus * _otherInfMinus, _selfInfMinus * _otherSupPlus), \
                                    np.maximum(_selfSupPlus * _otherSupPlus, _selfInfMinus * _otherInfMinus) - np.maximum(_selfInfPlus * _otherSupMinus, _selfSupMinus * _otherInfPlus))

        else:
            zero = self._zeros(self._shape)
            _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus = np.maximum(other, zero)
            _otherSupPlus = _otherInfPlus
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other, _otherSupPlus - other

            return KaucherArithmetic(np.maximum(_selfInfPlus * _otherInfPlus, _selfSupMinus * _otherSupMinus) - np.maximum(_selfSupPlus * _otherInfMinus, _selfInfMinus * _otherSupPlus), \
                                    np.maximum(_selfSupPlus * _otherSupPlus, _selfInfMinus * _otherInfMinus) - np.maximum(_selfInfPlus * _otherSupMinus, _selfSupMinus * _otherInfPlus))

    def __truediv__(self, other):
        if isinstance(other, ClassicalArithmetic):
            if 0 in other:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')

            other = ClassicalArithmetic(1/other.b, 1/other.a, sortQ=False)
            zero = self._zeros(self._shape)
            _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = np.maximum(other._a, zero), np.maximum(other._b, zero)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other._a, _otherSupPlus - other._b

            return KaucherArithmetic(_selfInfPlus*_otherInfPlus + _selfSupMinus*_otherSupMinus - _selfSupPlus*_otherInfMinus - _selfInfMinus*_otherSupPlus, \
                                    _selfSupPlus*_otherSupPlus + _selfInfMinus*_otherInfMinus - _selfInfPlus*_otherSupMinus - _selfSupMinus*_otherInfPlus)

        if isinstance(other, KaucherArithmetic):
            if 0 in other:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            return self.__mul__(KaucherArithmetic(1/other.b, 1/other.a))

        else:
            if 0 in np.asarray([other]):
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            return self.__mul__(1/other)



    def __pow__(self, other):
        if other == 1/2:
            return np.sqrt(self)
        elif (not isinstance(other, int)) or other < 0:
            raise Exception('Степень может быть только натуральным числом или равной 1/2.')

        if other % 2 == 0:
            index1 = np.array((self._a < 0) & (self._b <= 0))
            index2 = np.array((self._a < 0) & (self._b >= 0))
            index3 = np.array(~(index1 | index2), dtype=np.bool)

            left, right = self._zeros(self.shape), self._zeros(self.shape)

            left[index1], right[index1] = self[index1]._b**other, self[index1]._a**other
            right[index2] = abs(self[index2])**other
            left[index3], right[index3] = self[index3]._a**other, self[index3]._b**other
            return KaucherArithmetic(left, right)
        else:
            return KaucherArithmetic(self._a**other, self._b**other)

    def __matmul__(self, other):
        _ndim = (self._ndim, other.ndim)
        if _ndim == (2, 1) and self._shape[1] == other.shape[0]:
            return sum((self * other).T)

        elif self._shape == other.shape[::-1] and self._ndim == 2:
            n, _ = self._shape
            if isinstance(other, ARITHMETIC_TUPLE):
                result = KaucherArithmetic(self._zeros((n, n)), self._zeros((n, n)))
            for k in range(n):
                result[k] = sum((self[k] * other.T).T)
            return result

        elif _ndim[0]*_ndim[1] <= 1:
            return sum(self * other)

        elif _ndim == (1, 2) and self._shape[0] == other.shape[0]:
            return sum((self * other.T).T)

        else:
            msg = 'Входные операнды имеют неправильную размерность.'
            raise TypeError(msg)


    # Инверсные арифметические операторы
    def __radd__(self, other):
        return KaucherArithmetic(other + self._a, other + self._b)

    def __rsub__(self, other):
        return KaucherArithmetic(other - self._b, other - self._a)

    def __rmul__(self, other):
        zero = self._zeros(self._shape)
        _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
        _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

        _otherInfPlus = np.maximum(other, zero)
        _otherSupPlus = _otherInfPlus
        _otherInfMinus, _otherSupMinus = _otherInfPlus - other, _otherSupPlus - other

        return KaucherArithmetic(_selfInfPlus*_otherInfPlus + _selfSupMinus*_otherSupMinus - _selfSupPlus*_otherInfMinus - _selfInfMinus*_otherSupPlus, \
                                _selfSupPlus*_otherSupPlus + _selfInfMinus*_otherInfMinus - _selfInfPlus*_otherSupMinus - _selfSupMinus*_otherInfPlus)

    def __rtruediv__(self, other):
        if 0 in self:
            raise Exception('Нельзя делить на нуль содержащие интервалы!')
        return KaucherArithmetic(1/self.b, 1/self.a).__rmul__(other)


    def __rmatmul__(self, other):
        other = np.asarray(other)
        ndim = (other.ndim, self._ndim)
        if (ndim == (2, 1) and other.shape[1] == self._shape[0]) or \
           (self._shape == other.shape[::-1] and other.ndim == 2):
            om = other @ self.mid
            aor = np.abs(other) @ self.rad
            return KaucherArithmetic(om-aor, om+aor)

        elif ndim[0]*ndim[1] <= 1:
            return sum(self * other)

        elif ndim == (1, 2) and other.shape[0] == self._shape[0]:
            return sum((other * self).T)

        else:
            msg = 'Входные операнды имеют неправильную размерность.'
            raise TypeError(msg)

    #  Арифметические операторы присваивания
    def __iadd__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            self._a, self._b = self._a + other._a, self._b + other._b
        else:
            self._a, self._b = self._a + other, self._b + other
        return self

    def __isub__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            self._a, self._b = self._a - other._b,  self._b - other._a
        else:
            self._a, self._b = self._a - other,  self._b - other
        return self

    def __imul__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            zero = self._zeros(self._shape)
            _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = np.maximum(other._a, zero), np.maximum(other._b, zero)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other._a, _otherSupPlus - other._b

            self._a = np.maximum(_selfInfPlus * _otherInfPlus, _selfSupMinus * _otherSupMinus) - np.maximum(_selfSupPlus * _otherInfMinus, _selfInfMinus * _otherSupPlus)
            self._b = np.maximum(_selfSupPlus * _otherSupPlus, _selfInfMinus * _otherInfMinus) - np.maximum(_selfInfPlus * _otherSupMinus, _selfSupMinus * _otherInfPlus)
        else:
            zero = self._zeros(self._shape)
            _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus = np.maximum(other, zero)
            _otherSupPlus = _otherInfPlus
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other, _otherSupPlus - other

            self._a = np.maximum(_selfInfPlus * _otherInfPlus, _selfSupMinus * _otherSupMinus) - np.maximum(_selfSupPlus * _otherInfMinus, _selfInfMinus * _otherSupPlus)
            self._b = np.maximum(_selfSupPlus * _otherSupPlus, _selfInfMinus * _otherInfMinus) - np.maximum(_selfInfPlus * _otherSupMinus, _selfSupMinus * _otherInfPlus)
        return self

    def __itruediv__(self, other):
        if isinstance(other, ClassicalArithmetic):
            if 0 in other:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')

            other = ClassicalArithmetic(1/other.b, 1/other.a, sortQ=False)
            zero = self._zeros(self._shape)
            _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = np.maximum(other._a, zero), np.maximum(other._b, zero)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other._a, _otherSupPlus - other._b

            self._a = _selfInfPlus*_otherInfPlus + _selfSupMinus*_otherSupMinus - _selfSupPlus*_otherInfMinus - _selfInfMinus*_otherSupPlus
            self._b = _selfSupPlus*_otherSupPlus + _selfInfMinus*_otherInfMinus - _selfInfPlus*_otherSupMinus - _selfSupMinus*_otherInfPlus

        if isinstance(other, KaucherArithmetic):
            if 0 in other:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            tmp = self.__mul__(KaucherArithmetic(1/other.b, 1/other.a))
            self._a, self._b = tmp._a, tmp._b

        else:
            if 0 in np.asarray([other]):
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            tmp = self.__mul__(1/other)
            self._a, self._b = tmp._a, tmp._b

        return self


ARITHMETIC_TUPLE = (ClassicalArithmetic, KaucherArithmetic)
def Interval(*args, sortQ=True, midRadQ=False):

    n = len(args)
    assert n <= 2, "Основных аргументов не может быть больше двух."

    if precision.increasedPrecisionQ:
        # задаём левые и правые концы интервала-(ов)
        if n == 2:
            left = np.array(np.array(args[0], dtype=np.str), dtype=np.object)
            right = np.array(np.array(args[1], dtype=np.str), dtype=np.object)
        else:
            data = np.array(args[0], dtype=np.str)
            if data.ndim == 1:
                left, right = np.array(data[0], dtype=np.object), np.array(data[1], dtype=np.object)
            elif data.ndim == 2:
                left, right = np.array(data[:, 0], dtype=np.object), np.array(data[:, 1], dtype=np.object)
            else:
                left, right = np.array(data[:, :, 0], dtype=np.object), np.array(data[:, :, 1], dtype=np.object)

        ranges = [range(0, left.shape[k]) for k in range(left.ndim)]
        isnan = False
        for index in itertools.product(*ranges):
            try:
                left[index], right[index] = mpf(left[index]), mpf(right[index])
            except:
                isnan = True
                left[index], right[index] = None, None

        # если оставить тип np.object, то невозможно совершать операции +,-,*,/ и т.д.
        # предполагается, что если в массиве встречается пустой интервал, то дальнейшая работа либо прекращается,
        # либо переходит в точность double
        if isnan:
            left, right = np.array(left, np.float64), np.array(right, np.float64)
        if midRadQ:
            left, right = left - right, left + right

    else:
        # задаём левые и правые концы интервала-(ов)
        if n == 2:
            left = np.array(args[0], dtype=np.float64)
            right = np.array(args[1], dtype=np.float64)
        else:
            data = np.array(args[0], dtype=np.float64)
            if data.ndim == 1:
                left, right = data[0], data[1]
            elif data.ndim == 2:
                left, right = data[:, 0], data[:, 1]
            else:
                left, right = data[:, :, 0], data[:, :, 1]

        if midRadQ:
            left, right = left - right, left + right

    if sortQ:
        return ClassicalArithmetic(left, right, sortQ)

    else:
        if (left > right).any():
            return KaucherArithmetic(left, right)
        else:
            return ClassicalArithmetic(left, right, sortQ)
