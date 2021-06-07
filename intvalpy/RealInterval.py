import numbers
import numpy as np
import itertools

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
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return self._ndim

    @property
    def ranges(self):
        return self._ranges

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
            result += "'[" + ', '.join(fs % x for x in [self[index]._a, self[index]._b]) + "]', "

        return result[:-2] + ']])' if self._ndim > 1 else result[:-2] + '])'

    #     Эмуляция коллекций
    def __len__(self):
        return self._a.__len__()

    def __setitem__(self, key, value):
        if (isinstance(self, KauherArithmetic) and isinstance(value, ARITHMETIC_TUPLE)) or \
        (isinstance(self, ClassicalArithmetic) and isinstance(value, ClassicalArithmetic)):
            self._a[key], self._b[key] = value._a, value._b
        elif isinstance(self, ClassicalArithmetic) and isinstance(value, KauherArithmetic):
            raise Exception('Объекты из разных арифметик!')
        else:
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
            s, o = self.pro, other.pro
            return ((s._a <= o._a) & (o._b <= s._b)).any()
        except:
            s = self.pro
            return ((s._a <= other) & (other <= s._b)).any()

    # Итерирование
    def __reversed__(self):
        return iter(self.storage[::-1])

    def __abs__(self):
        return np.maximum(np.abs(self._a), np.abs(self._b))

    #     Операторы сравнения
    def __eq__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return (self._a == other._a) & (self._b == other._b)
        else:
            return (self._a == other) & (self._b == other)

    def __ne__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return (self._a != other._a) | (self._b != other._b)
        else:
            return (self._a != other) | (self._b != other)

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
        result = np.minimum(np.abs(self._a), np.abs(self._b))
        result[(self._a * self._b) <= np.zeros(self._shape)] = 0
        return result

    def reshape(self, shape):
        return Interval(self._a.reshape(shape), self._b.reshape(shape), sortQ=False)

    @property
    def dual(self):
        return Interval(self._b, self._a, sortQ=False)

    @property
    def pro(self):
        if isinstance(self, ClassicalArithmetic):
            return ClassicalArithmetic(self._a, self._b, sortQ=False)
        else:
            return ClassicalArithmetic(self._a, self._b, sortQ=True)

    @property
    def opp(self):
        return Interval(-self._a, -self._b, sortQ=False)

    @property
    def inv(self):
        return Interval(1/self._a, 1/self._b, sortQ=False)

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
        cls = type(self)
        tmp = np.zeros(self.shape, dtype=object)
        for index in itertools.product(*self.ranges):
            tmp[index] = cls(self._a[index], self._b[index])
        return tmp

    def __array__(self):
        return self.asnumpy()

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

        elif args[0].__name__ in ['exp']:
            return cls(args[0].__call__(self._a), args[0].__call__(self._b))

        elif args[0].__name__ in ['sin']:
            result = cls(np.zeros(self._shape), np.zeros(self._shape))
            for index in itertools.product(*self._ranges):
                if self._b[index] - self._a[index] >= 3*np.pi/2:
                    result[index] = cls(-1, 1)
                    continue

                offset = max(self._a[index]//(2*np.pi), self._b[index]//(2*np.pi))
                _a = self._a[index] - (2*np.pi)*offset
                _b = self._b[index] - (2*np.pi)*offset

                if (_a <= -3*np.pi/2 and _b <= -3*np.pi/2) or \
                   (-3*np.pi/2 <= _a and _a <= -np.pi/2 and -3*np.pi/2 <= _b and _b <= -np.pi/2) or \
                   (-np.pi/2 <= _a and _a <= np.pi/2 and -np.pi/2 <= _b and _b <= np.pi/2) or \
                   (np.pi/2 <= _a and _a <= 3*np.pi/2 and np.pi/2 <= _b and _b <= 3*np.pi/2) or \
                   (3*np.pi/2 <= _a and 3*np.pi/2 <= _b):
                    result[index] = cls(np.sin(_a), np.sin(_b))

                elif (-3*np.pi/2 <= _a and _a <= -np.pi/2 and -np.pi/2 <= _b and _b <= np.pi/2) or \
                     (np.pi/2 <= _a and _a <= 3*np.pi/2 and 3*np.pi/2 <= _b):
                    result[index] = cls(-1, max(np.sin(_a), np.sin(_b)))

                elif (_a <= -3*np.pi/2 and -3*np.pi/2 <= _b and _b <= -np.pi/2) or \
                     (-np.pi/2 <= _a and _a <= np.pi/2 and np.pi/2 <= _b and _b <= 3*np.pi/2):
                    result[index] = cls(min(np.sin(_a), np.sin(_b)), 1)

                else:
                    result[index] = cls(-1, 1)
            return result

        elif args[0].__name__ in ['cos']:
            result = cls(np.zeros(self._shape), np.zeros(self._shape))
            for index in itertools.product(*self._ranges):
                if self._b[index] - self._a[index] >= 3*np.pi/2:
                    result[index] = cls(-1, 1)
                    continue

                offset = max(self._a[index]//(2*np.pi), self._b[index]//(2*np.pi))
                _a = self._a[index] - (2*np.pi)*offset
                _b = self._b[index] - (2*np.pi)*offset

                if (_a <= -np.pi and _b <= -np.pi) or \
                   (-np.pi <= _a and _a <= 0 and -np.pi <= _b and _b <= 0) or \
                   (0 <= _a and _a <= np.pi and 0 <= _b and _b <= np.pi) or \
                   (np.pi <= _a and _a <= 2*np.pi and np.pi <= _b and _b <= 2*np.pi):
                    result[index] = cls(np.cos(_a), np.cos(_b))

                elif (_a <= -np.pi and -np.pi <= _b and _b <= 0) or \
                     (0 <= _a and _a <= np.pi and np.pi <= _b and _b <= 2*np.pi):
                    result[index] = cls(-1, max(np.cos(_a), np.cos(_b)))

                elif -np.pi <= _a and _a <= 0 and 0 <= _b and _b <= np.pi:
                    result[index] = cls(min(np.cos(_a), np.cos(_b)), 1)

                else:
                    result[index] = cls(-1, 1)
            return result

        elif args[0].__name__ in ['log']:
            if ((self._a <= 0) | (self._b <= 0)).any():
                raise Exception('Попытка вычислить логарифм от не положительного интервала!')
            else:
                return cls(np.log(self._a), np.log(self._b))

        else:
            raise Exception("Расчёт функции {} не предусмотрен!".format(args[0].__name__))


class ClassicalArithmetic(BaseTools):

    def __init__(self, left, right, sortQ=False):
        self._a, self._b = np.copy(left), np.copy(right)

        self._shape = self._a.shape
        self._ndim = self._a.ndim
        self._ranges = [range(0, self._shape[k]) for k in range(self._ndim)]

        if sortQ:
            self._a, self._b = np.minimum(self._a, self._b), np.maximum(self._a, self._b)

    def __getitem__(self, key):
        if isinstance(key, (slice, numbers.Integral, tuple, list, np.ndarray)):
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
        return KauherArithmetic(np.copy(self._a), np.copy(self._b))

    @property
    def T(self):
        return ClassicalArithmetic(self._a.T, self._b.T, sortQ=False)

    # Арифметические операторы
    def __add__(self, other):
        if isinstance(other, ClassicalArithmetic):
            return ClassicalArithmetic(self._a + other._a, self._b + other._b, sortQ=False)
        elif isinstance(other, KauherArithmetic):
            return KauherArithmetic(self._a + other._a, self._b + other._b)
        else:
            return ClassicalArithmetic(self._a + other, self._b + other, sortQ=False)

    def __sub__(self, other):
        if isinstance(other, ClassicalArithmetic):
            return ClassicalArithmetic(self._a - other._b, self._b - other._a, sortQ=False)
        elif isinstance(other, KauherArithmetic):
            return KauherArithmetic(self._a - other._b, self._b - other._a)
        else:
            return ClassicalArithmetic(self._a - other, self._b - other, sortQ=False)

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

        elif isinstance(other, KauherArithmetic):
            zero = np.zeros(self._shape)
            _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = np.maximum(other._a, zero), np.maximum(other._b, zero)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other._a, _otherSupPlus - other._b

            return KauherArithmetic(_selfInfPlus*_otherInfPlus + _selfSupMinus*_otherSupMinus - _selfSupPlus*_otherInfMinus - _selfInfMinus*_otherSupPlus, \
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

        if isinstance(other, KauherArithmetic):
            if 0 in other:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            return self.__mul__(KauherArithmetic(1/other._b, 1/other._a))

        else:
            if 0 in np.asarray([other]):
                raise Exception('Нельзя делить на нуль!')
            tmp = np.array([self._a/other,
                            self._b/other])

            return ClassicalArithmetic(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

    def __pow__(self, other):
        if (not isinstance(other, int)) or other < 0:
            raise Exception('Степень может быть только натуральным числом.')

        if other % 2 == 0:
            tmp_a = np.zeros(self.shape)
            tmp_b = np.zeros(self.shape)

            for index in itertools.product(*self.ranges):
                if self._a[index] < 0 and self._b[index] <= 0:
                    tmp_a[index] = self._b[index]**other
                    tmp_b[index] = self._a[index]**other
                elif self._a[index] < 0 and self._b[index] >= 0:
                    tmp_a[index] = 0
                    tmp_b[index] = abs(self[index])**other
                else:
                    tmp_a[index] = self._a[index]**other
                    tmp_b[index] = self._b[index]**other
            return ClassicalArithmetic(tmp_a, tmp_b, sortQ=False)
        else:
            return ClassicalArithmetic(self._a**other, self._b**other, sortQ=False)

    def __matmul__(self, other):
        _ndim = (self._ndim, other.ndim)
        if _ndim == (2, 1) and self._shape[1] == other.shape[0]:
            return sum((self * other).T)

        elif self._shape == other.shape[::-1] and self._ndim == 2:
            n, _ = self._shape
            if isinstance(other, KauherArithmetic):
                result = KauherArithmetic(np.zeros((n, n)), np.zeros((n, n)))
            else:
                result = ClassicalArithmetic(np.zeros((n, n)), np.zeros((n, n)), sortQ=False)
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
        elif isinstance(other, KauherArithmetic):
            raise Exception('Данная функция не поддерживает сложение с интервалами из разных арифметик.')
        else:
            self._a, self._b = self._a + other, self._b + other
        return self

    def __isub__(self, other):
        if isinstance(other, ClassicalArithmetic):
            self._a, self._b = self._a - other._b,  self._b - other._a
        elif isinstance(other, KauherArithmetic):
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

        elif isinstance(other, KauherArithmetic):
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

        elif isinstance(other, KauherArithmetic):
            raise Exception('Данная функция не поддерживает деление с интервалами из разных арифметик.')

        else:
            if 0 in np.asarray([other]):
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            tmp = np.array([self._a/other,
                            self._b/other])
            self._a, self._b = tmp.min(axis=0), tmp.max(axis=0)
        return self


class KauherArithmetic(BaseTools):

    def __init__(self, left, right):
        self._a, self._b = np.copy(left), np.copy(right)

        self._shape = self._a.shape
        self._ndim = self._a.ndim
        self._ranges = [range(0, self._shape[k]) for k in range(self._ndim)]

    def __getitem__(self, key):
        if isinstance(key, (slice, numbers.Integral, tuple, list, np.ndarray)):
            return KauherArithmetic(self._a[key], self._b[key])
        else:
            msg = 'Indices must be integers / slice / tuple / list / np.ndarray'
            raise TypeError(msg)

    #     Унарные числовые операции
    def __neg__(self):
        return KauherArithmetic(-self._b, -self._a)

    @property
    def copy(self):
        return KauherArithmetic(np.copy(self._a), np.copy(self._b))

    @property
    def T(self):
        return KauherArithmetic(self._a.T, self._b.T)

    #     Арифметические операторы
    def __add__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return KauherArithmetic(self._a + other._a, self._b + other._b)
        else:
            return KauherArithmetic(self._a + other, self._b + other)

    def __sub__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            return KauherArithmetic(self._a - other._b, self._b - other._a)
        else:
            return KauherArithmetic(self._a - other, self._b - other)

    def __mul__(self, other):
        if isinstance(other, ARITHMETIC_TUPLE):
            zero = np.zeros(self._shape)
            _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = np.maximum(other._a, zero), np.maximum(other._b, zero)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other._a, _otherSupPlus - other._b

            return KauherArithmetic(np.maximum(_selfInfPlus * _otherInfPlus, _selfSupMinus * _otherSupMinus) - np.maximum(_selfSupPlus * _otherInfMinus, _selfInfMinus * _otherSupPlus), \
                                    np.maximum(_selfSupPlus * _otherSupPlus, _selfInfMinus * _otherInfMinus) - np.maximum(_selfInfPlus * _otherSupMinus, _selfSupMinus * _otherInfPlus))

        else:
            zero = np.zeros(self._shape)
            _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus = np.maximum(other, zero)
            _otherSupPlus = _otherInfPlus
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other, _otherSupPlus - other

            return KauherArithmetic(np.maximum(_selfInfPlus * _otherInfPlus, _selfSupMinus * _otherSupMinus) - np.maximum(_selfSupPlus * _otherInfMinus, _selfInfMinus * _otherSupPlus), \
                                    np.maximum(_selfSupPlus * _otherSupPlus, _selfInfMinus * _otherInfMinus) - np.maximum(_selfInfPlus * _otherSupMinus, _selfSupMinus * _otherInfPlus))

    # Разобраться с делением!!! провести тесты
    def __truediv__(self, other):
        if isinstance(other, ClassicalArithmetic):
            if 0 in other:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')

            other = ClassicalArithmetic(1/other.b, 1/other.a, sortQ=False)
            zero = np.zeros(self._shape)
            _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = np.maximum(other._a, zero), np.maximum(other._b, zero)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other._a, _otherSupPlus - other._b

            return KauherArithmetic(_selfInfPlus*_otherInfPlus + _selfSupMinus*_otherSupMinus - _selfSupPlus*_otherInfMinus - _selfInfMinus*_otherSupPlus, \
                                    _selfSupPlus*_otherSupPlus + _selfInfMinus*_otherInfMinus - _selfInfPlus*_otherSupMinus - _selfSupMinus*_otherInfPlus)

        if isinstance(other, KauherArithmetic):
            if 0 in other:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            return self.__mul__(KauherArithmetic(1/other.b, 1/other.a))

        else:
            if 0 in np.asarray([other]):
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            return self.__mul__(1/other)



    def __pow__(self, other):
        if (not isinstance(other, int)) or other < 0:
            raise Exception('Степень может быть только натуральным числом.')

        if other % 2 == 0:
            tmp_a = np.zeros(self.shape)
            tmp_b = np.zeros(self.shape)

            for index in itertools.product(*self.ranges):
                if self._a[index] < 0 and self._b[index] <= 0:
                    tmp_a[index] = self._b[index]**other
                    tmp_b[index] = self._a[index]**other
                elif self._a[index] < 0 and self._b[index] >= 0:
                    tmp_a[index] = 0
                    tmp_b[index] = abs(self[index])**other
                else:
                    tmp_a[index] = self._a[index]**other
                    tmp_b[index] = self._b[index]**other
            return KauherArithmetic(tmp_a, tmp_b)
        else:
            return KauherArithmetic(self._a**other, self._b**other)

    def __matmul__(self, other):
        _ndim = (self._ndim, other.ndim)
        if _ndim == (2, 1) and self._shape[1] == other.shape[0]:
            return sum((self * other).T)

        elif self._shape == other.shape[::-1] and self._ndim == 2:
            n, _ = self._shape
            if isinstance(other, ARITHMETIC_TUPLE):
                result = KauherArithmetic(np.zeros((n, n)), np.zeros((n, n)))
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
        return KauherArithmetic(other + self._a, other + self._b)

    def __rsub__(self, other):
        return KauherArithmetic(other - self._b, other - self._a)

    def __rmul__(self, other):
        zero = np.zeros(self._shape)
        _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
        _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

        _otherInfPlus = np.maximum(other, zero)
        _otherSupPlus = _otherInfPlus
        _otherInfMinus, _otherSupMinus = _otherInfPlus - other, _otherSupPlus - other

        return KauherArithmetic(_selfInfPlus*_otherInfPlus + _selfSupMinus*_otherSupMinus - _selfSupPlus*_otherInfMinus - _selfInfMinus*_otherSupPlus, \
                                _selfSupPlus*_otherSupPlus + _selfInfMinus*_otherInfMinus - _selfInfPlus*_otherSupMinus - _selfSupMinus*_otherInfPlus)

    def __rtruediv__(self, other):
        if 0 in self:
            raise Exception('Нельзя делить на нуль содержащие интервалы!')
        return KauherArithmetic(1/self.b, 1/self.a).__rmul__(other)


    def __rmatmul__(self, other):
        other = np.asarray(other)
        ndim = (other.ndim, self._ndim)
        if (ndim == (2, 1) and other.shape[1] == self._shape[0]) or \
           (self._shape == other.shape[::-1] and other.ndim == 2):
            om = other @ self.mid
            aor = np.abs(other) @ self.rad
            return KauherArithmetic(om-aor, om+aor)

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
            zero = np.zeros(self._shape)
            _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = np.maximum(other._a, zero), np.maximum(other._b, zero)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other._a, _otherSupPlus - other._b

            self._a = np.maximum(_selfInfPlus * _otherInfPlus, _selfSupMinus * _otherSupMinus) - np.maximum(_selfSupPlus * _otherInfMinus, _selfInfMinus * _otherSupPlus)
            self._b = np.maximum(_selfSupPlus * _otherSupPlus, _selfInfMinus * _otherInfMinus) - np.maximum(_selfInfPlus * _otherSupMinus, _selfSupMinus * _otherInfPlus)
        else:
            zero = np.zeros(self._shape)
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
            zero = np.zeros(self._shape)
            _selfInfPlus, _selfSupPlus = np.maximum(self._a, zero), np.maximum(self._b, zero)
            _selfInfMinus, _selfSupMinus = _selfInfPlus - self._a, _selfSupPlus - self._b

            _otherInfPlus, _otherSupPlus = np.maximum(other._a, zero), np.maximum(other._b, zero)
            _otherInfMinus, _otherSupMinus = _otherInfPlus - other._a, _otherSupPlus - other._b

            self._a = _selfInfPlus*_otherInfPlus + _selfSupMinus*_otherSupMinus - _selfSupPlus*_otherInfMinus - _selfInfMinus*_otherSupPlus
            self._b = _selfSupPlus*_otherSupPlus + _selfInfMinus*_otherInfMinus - _selfInfPlus*_otherSupMinus - _selfSupMinus*_otherInfPlus

        if isinstance(other, KauherArithmetic):
            if 0 in other:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            tmp = self.__mul__(KauherArithmetic(1/other.b, 1/other.a))
            self._a, self._b = tmp._a, tmp._b

        else:
            if 0 in np.asarray([other]):
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            tmp = self.__mul__(1/other)
            self._a, self._b = tmp._a, tmp._b

        return self


ARITHMETIC_TUPLE = (ClassicalArithmetic, KauherArithmetic)
def Interval(left, right, sortQ=True):
    left = np.asarray(left, dtype='float64')
    right = np.asarray(right, dtype='float64')

    if sortQ:
        return ClassicalArithmetic(left, right, sortQ)

    else:
        if (left > right).any():
            return KauherArithmetic(left, right)
        else:
            return ClassicalArithmetic(left, right, sortQ)
