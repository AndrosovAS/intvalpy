import numbers
import numpy as np
import itertools


class Interval:
    """
    Класс Interval создаёт интервал или массив интервалов.

    Input:
            left: int, float, list, ndarray
                Левый конец интервала или интервалов.

            right: int, float, list, ndarray
                Правый конец интервала или интервалов.

            sortQ: bool
                Необязательный параметр для проверки правильности интервала.
                В случае, если интервал неправильный, то создаётся правильный интервал
                путём перестановки концов интервала.
                Значением по умолчанию является True.

    Methods:
            self.a -- возвращает левый конец интервала или интервалов.
            self.b -- возвращает правый конец интервала или интервалов.

            self.rad -- возвращает радиус интервала или интервалов.
            self.wid -- возвращает ширину интервала или интервалов.
            self.mid -- возвращает середину интервала или интервалов.
            self.mig -- возвращает мигнитуду интервала или интервалов.

            self.copy -- создаёт новый объект.

            self.invbar -- создаёт алгебраически обратный к self интревал.
            self.opp -- создаёт алгебраически противоположный к self интревал.
    """

    def __init__(self, left, right, sortQ=True):
        self._a = np.asarray(left, dtype='float64')
        self._b = np.asarray(right, dtype='float64')
        self.shape = self._a.shape
        self.ndim = self._a.ndim
        self.ranges = [range(0, self.shape[k]) for k in range(self.ndim)]

        if sortQ:
            for index in itertools.product(*self.ranges):
                if self._a[index] > self._b[index]:
                    self._a[index], self._b[index] = self._b[index], self._a[index]

    @property
    def a(self):
        """Возвращает левый конец интервала."""
        return self._a

    @property
    def b(self):
        """Возвращает правый конец интервала."""
        return self._b

    def __repr__(self):
        result = 'interval('

        if self.shape == ():
            return '[%.7g, %.7g]' % (self._a, self._b)

        elif self.ndim == 1:
            try:
                result += str(['[%.7g, %.7g]' % (self._a[k], self._b[k]) for k in range(self.shape[0])]) + ')'
            except:
                result += str(['[%.7g, %.7g]' % (self._a, self._b)]) + ')'
        else:
            for l in range(self.shape[0]-1):
                result += str(['[%.7g, %.7g]' % (self._a[l, k], self._b[l, k]) for k in range(self.shape[1])]) + '\n      '

            result += str(['[%.7g, %.7g]' % (self._a[self.shape[0]-1, k], self._b[self.shape[0]-1, k]) \
                           for k in range(self.shape[1])]) + ')\n'

        return result

#########################################################################################################
#                  Описываем операции для классической интервальной арифметики.                         #
#                                                                                                       #
#########################################################################################################

    #     Эмуляция коллекций
    def __len__(self):
        return self._a.__len__()

    def __setitem__(self, key, value):
        if isinstance(value, Interval):
            self._a[key], self._b[key] = value._a, value._b
        else:
            self._a[key], self._b[key] = value, value

    def __delitem__(self, key):
        if isinstance(key, int):
            self._a = np.delete(self._a, key, axis=0)
            self._b = np.delete(self._b, key, axis=0)
            self.shape = self._a.shape
            self.ndim = self._a.ndim
            self.ranges = [range(0, self.shape[k]) for k in range(self.ndim)]

        elif isinstance(key, (slice, tuple, list, np.ndarray)):
            self._a = np.delete(self._a, key[-1], axis=len(key)-1)
            self._b = np.delete(self._b, key[-1], axis=len(key)-1)
            self.shape = self._a.shape
            self.ndim = self._a.ndim
            self.ranges = [range(0, self.shape[k]) for k in range(self.ndim)]

        else:
            msg = '{Interval.__name__} indices must be integers'
            raise TypeError(msg.format(cls=Interval))

    def __getitem__(self, key):
        if isinstance(key, (slice, numbers.Integral, tuple, list, np.ndarray)):
            return Interval(self._a[key], self._b[key], sortQ=False)
        else:
            msg = '{Interval.__name__} indices must be integers'
            raise TypeError(msg.format(cls=Interval))

    def __contains__(self, other):
        try:
            tmp1 = self._a <= other._a
            tmp2 = other._b <= self._b
        except:
            tmp1 = self._a <= other
            tmp2 = other <= self._b
        return (tmp1 & tmp2).any()

    # Итерирование
    def __reversed__(self):
        return iter(self.storage[::-1])

    #     Унарные числовые операции
    def __neg__(self):
        return Interval(-self._b, -self._a, sortQ=False)

    def __abs__(self):
        if self.shape == (1, ):
            result = max([abs(self._a), abs(self._b)])
        else:
            result = np.zeros(self.shape)
            for index in itertools.product(*self.ranges):
                result[index] = max([abs(self._a[index]), abs(self._b[index])])
        return result

#     Операторы сравнения
    def __eq__(self, other):
        if isinstance(other, Interval):
            tmp1 = self._a == other._a
            tmp2 = self._b == other._b
        else:
            tmp1 = self._a == other
            tmp2 = self._b == other
        return tmp1 & tmp2

    def __ne__(self, other):
        if isinstance(other, Interval):
            tmp1 = self._a != other._a
            tmp2 = self._b != other._b
        else:
            tmp1 = self._a != other
            tmp2 = self._b != other
        return tmp1 | tmp2

    #     Арифметические операторы
    def __add__(self, other):
        if isinstance(other, Interval):
            return Interval(self._a + other._a, self._b + other._b, sortQ=False)
        else:
            return Interval(self._a + other, self._b + other, sortQ=False)

    def __sub__(self, other):
        if isinstance(other, Interval):
            return Interval(self._a - other._b, self._b - other._a, sortQ=False)
        else:
            return Interval(self._a - other, self._b - other, sortQ=False)

    def __mul__(self, other):
        if isinstance(other, Interval):
            tmp = np.array([self._a*other._a,
                            self._a*other._b,
                            self._b*other._a,
                            self._b*other._b])
            try:
                tmp = np.array([tmp[:, k] for k in range(tmp.shape[1])])
                return Interval(tmp.min(axis=1), tmp.max(axis=1), sortQ=False)
            except:
                return Interval(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

        else:
            tmp = np.array([self._a*other,
                            self._b*other])

            return Interval(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

    def __truediv__(self, other):
        if isinstance(other, Interval):
            if 0 in other:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            tmp = np.array([self._a/other._a,
                            self._a/other._b,
                            self._b/other._a,
                            self._b/other._b])
            try:
                tmp = np.array([tmp[:, k] for k in range(tmp.shape[1])])
                return Interval(tmp.min(axis=1), tmp.max(axis=1), sortQ=False)
            except:
                return Interval(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

        else:
            if 0 in np.asarray([other]):
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            tmp = np.array([self._a/other,
                            self._b/other])

            return Interval(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

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
            return Interval(tmp_a, tmp_b, sortQ=False)
        else:
            return Interval(self._a**other, self._b**other, sortQ=False)

    def __matmul__(self, other):
        ndim = (self.ndim, other.ndim)
        if ndim == (2, 1) and self.shape[1] == other.shape[0]:
            n, _ = self.shape
            tmp = self * other
            result = Interval(np.zeros(n), np.zeros(n), sortQ=False)
            for k in range(n):
                result[k] += sum(tmp[k])
            return result

        elif self.shape == other.shape[::-1] and self.ndim is 2:
            n, _ = self.shape
            result = Interval(np.zeros((n, n)), np.zeros((n, n)), sortQ=False)

            for k in range(n):
                for l in range(n):
                    result[k, l] += sum(self[k] * other[:, l])
            return result

        elif ndim[0]*ndim[1] <= 1:
            return sum(self * other)

        elif ndim == (1, 2) and self.shape[0] == other.shape[0]:
            _, m = other.shape
            result = Interval(np.zeros(m), np.zeros(m), sortQ=False)
            for k in range(m):
                result[k] += sum(self * other[:, k])
            return result

        else:
            msg = 'Входные операнды имеют неправильную размерность.'
            raise TypeError(msg)

    #     Инверсные арифметические операторы
    def __radd__(self, other):
        if isinstance(other, Interval):
            return Interval(other._a + self._a, other._b + self._b, sortQ=False)
        else:
            return Interval(other + self._a, other + self._b, sortQ=False)

    def __rsub__(self, other):
        if isinstance(other, Interval):
            return Interval(other._a - self._b, other._b - self._a, sortQ=False)
        else:
            return Interval(other - self._b, other - self._a, sortQ=False)

    def __rmul__(self, other):
        if isinstance(other, Interval):
            tmp = np.array([other._a*self._a,
                            other._a*self._b,
                            other._b*self._a,
                            other._b*self._b])
            try:
                tmp = np.array([tmp[:, k] for k in range(tmp.shape[1])])
                return Interval(tmp.min(axis=1), tmp.max(axis=1), sortQ=False)
            except:
                return Interval(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

        else:
            tmp = np.array([other*self._a,
                            other*self._b])

            return Interval(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

    def __rtruediv__(self, other):
        if 0 in self:
            raise Exception('Нельзя делить на нуль содержащие интервалы!')
        if isinstance(other, Interval):
            tmp = np.array([other._a/self._a,
                            other._a/self._b,
                            other._b/self._a,
                            other._b/self._b])
            try:
                tmp = np.array([tmp[:, k] for k in range(tmp.shape[1])])
                return Interval(tmp.min(axis=1), tmp.max(axis=1), sortQ=False)
            except:
                return Interval(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

        else:
            tmp = np.array([other/self._a,
                            other/self._b])

            return Interval(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

    def __rmatmul__(self, other):
        other = np.asarray(other)
        ndim = (other.ndim, self.ndim)
        if ndim == (2, 1) and other.shape[1] == self.shape[0]:
            n, _ = other.shape
            result = Interval(np.zeros(n), np.zeros(n), sortQ=False)
            tmp = other * self
            for k in range(n):
                result[k] += sum(tmp[k])
            return result

        elif self.shape == other.shape[::-1] and other.ndim is 2:
            n, _ = other.shape
            result = Interval(np.zeros((n,n)), np.zeros((n,n)), sortQ=False)
            for k in range(n):
                for l in range(n):
                    result[k, l] += sum(other[k] * self[:, l])
            return result

        elif ndim[0]*ndim[1] <= 1:
            return sum(self * other)

        elif ndim == (1, 2) and other.shape[0] == self.shape[0]:
            _, m = self.shape
            result = Interval(np.zeros(m), np.zeros(m), sortQ=False)
            tmp = other * self
            for k in range(m):
                result[k] += sum(tmp[k])
            return result

        else:
            msg = 'Входные операнды имеют неправильную размерность.'
            raise TypeError(msg)

#     Арифметические операторы присваивания
    def __iadd__(self, other):
        if isinstance(other, Interval):
            self._a, self._b = self._a + other._a, self._b + other._b
        else:
            self._a, self._b = self._a + other, self._b + other
        return self

    def __isub__(self, other):
        if isinstance(other, Interval):
            self._a, self._b = self._a - other._b,  self._b - other._a
        else:
            self._a, self._b = self._a - other,  self._b - other
        return self

    def __imul__(self, other):
        if isinstance(other, Interval):
            tmp = np.array([self._a*other._a,
                            self._a*other._b,
                            self._b*other._a,
                            self._b*other._b])
            try:
                tmp = np.array([tmp[:, k] for k in range(tmp.shape[1])])
                self._a, self._b = tmp.min(axis=1), tmp.max(axis=1)
            except:
                self._a, self._b = tmp.min(axis=0), tmp.max(axis=0)

        else:
            tmp = np.array([self._a*other,
                            self._b*other])
            self._a, self._b = tmp.min(axis=0), tmp.max(axis=0)
        return self

    def __itruediv__(self, other):
        if isinstance(other, Interval):
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

        else:
            if 0 in np.asarray([other]):
                raise Exception('Нельзя делить на нуль содержащие интервалы!')
            tmp = np.array([self._a/other,
                            self._b/other])
            self._a, self._b = tmp.min(axis=0), tmp.max(axis=0)
        return self

#########################################################################################################
#                  Согласование класса Interval с другими библиотеками Python.                          #
#                                                                                                       #
#########################################################################################################

    #     Копирование объекта
    def __deepcopy__(self, memo):
        return Interval(np.copy(self._a), np.copy(self._b), sortQ=False)

    # Преобразование в массив типа ndarray
    def asnumpy(self):
        tmp = np.zeros(self.shape, dtype=object)
        for index in itertools.product(*self.ranges):
            tmp[index] = Interval(self._a[index], self._b[index], sortQ=False)
        return tmp

    def __array__(self):
        return self.asnumpy()

    # Универсальная функция
    def __array_ufunc__(self, *args):
        if args[0].__name__ in ['add']:
            return Interval(self._a + args[2], self._b + args[2], sortQ=False)

        elif args[0].__name__ in ['subtract']:
            return Interval(args[2] - self._b, args[2] - self._a, sortQ=False)

        elif args[0].__name__ in ['multiply']:
            tmp = np.array([args[2]*self._a,
                            args[2]*self._a,
                            args[2]*self._b,
                            args[2]*self._b])

            return Interval(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

        elif args[0].__name__ in ['true_divide']:
            if 0 in self:
                raise Exception('Нельзя делить на нуль содержащие интервалы!')

            tmp = np.array([args[2]/self._a,
                            args[2]/self._a,
                            args[2]/self._b,
                            args[2]/self._b])
            return Interval(tmp.min(axis=0), tmp.max(axis=0), sortQ=False)

        elif args[0].__name__ in ['matmul']:
            ndim = (args[2].ndim, self.ndim)
            if ndim == (2, 1) and args[2].shape[1] == self.shape[0]:
                _, m = args[2].shape
                tmp = self * args[2]
                result = Interval(np.zeros(m), np.zeros(m), sortQ=False)
                for k in range(m):
                    result[k] += sum(tmp[k])
                return result

            elif args[2].shape == self.shape[::-1] and self.ndim is 2:
                n, _ = args[2].shape
                result = Interval(np.zeros((n, n)), np.zeros((n, n)), sortQ=False)

                for k in range(n):
                    for l in range(n):
                        result[k, l] += sum(args[2][k] * self[:, l])
                return result

            elif ndim == (1, 2) and args[2].shape[0] == self.shape[0]:
                _, m = self.shape
                result = Interval(np.zeros(m), np.zeros(m), sortQ=False)
                for k in range(m):
                    result[k] += sum(self[:, k] * args[2])
                return result

            elif ndim[0]*ndim[1] <=1:
                return sum(self * args[2])

            else:
                msg = 'Входные операнды имеют неправильную размерность.'
                raise TypeError(msg)

        elif args[0].__name__ in ['exp']:
            return Interval(args[0].__call__(self._a), args[0].__call__(self._b), sortQ=False)

        elif args[0].__name__ in ['sin']:
            result = Interval(np.zeros(self.shape), np.zeros(self.shape))
            for index in itertools.product(*self.ranges):
                if self._b[index] - self._a[index] >= 3*np.pi/2:
                    result[index] = Interval(-1, 1, sortQ=False)
                    continue

                offset = max(self._a[index]//(2*np.pi), self._b[index]//(2*np.pi))
                _a = self._a[index] - (2*np.pi)*offset
                _b = self._b[index] - (2*np.pi)*offset

                if (_a <= -3*np.pi/2 and _b <= -3*np.pi/2) or \
                   (-3*np.pi/2 <= _a and _a <= -np.pi/2 and -3*np.pi/2 <= _b and _b <= -np.pi/2) or \
                   (-np.pi/2 <= _a and _a <= np.pi/2 and -np.pi/2 <= _b and _b <= np.pi/2) or \
                   (np.pi/2 <= _a and _a <= 3*np.pi/2 and np.pi/2 <= _b and _b <= 3*np.pi/2) or \
                   (3*np.pi/2 <= _a and 3*np.pi/2 <= _b):
                    result[index] = Interval(np.sin(_a), np.sin(_b))

                elif (-3*np.pi/2 <= _a and _a <= -np.pi/2 and -np.pi/2 <= _b and _b <= np.pi/2) or \
                     (np.pi/2 <= _a and _a <= 3*np.pi/2 and 3*np.pi/2 <= _b):
                    result[index] = Interval(-1, max(np.sin(_a), np.sin(_b)), sortQ=False)

                elif (_a <= -3*np.pi/2 and -3*np.pi/2 <= _b and _b <= -np.pi/2) or \
                     (-np.pi/2 <= _a and _a <= np.pi/2 and np.pi/2 <= _b and _b <= 3*np.pi/2):
                    result[index] = Interval(min(np.sin(_a), np.sin(_b)), 1, sortQ=False)

                else:
                    result[index] = Interval(-1, 1, sortQ=False)
            return result

        elif args[0].__name__ in ['cos']:
            result = Interval(np.zeros(self.shape), np.zeros(self.shape))
            for index in itertools.product(*self.ranges):
                if self._b[index] - self._a[index] >= 3*np.pi/2:
                    result[index] = Interval(-1, 1, sortQ=False)
                    continue

                offset = max(self._a[index]//(2*np.pi), self._b[index]//(2*np.pi))
                _a = self._a[index] - (2*np.pi)*offset
                _b = self._b[index] - (2*np.pi)*offset

                if (_a <= -np.pi and _b <= -np.pi) or \
                   (-np.pi <= _a and _a <= 0 and -np.pi <= _b and _b <= 0) or \
                   (0 <= _a and _a <= np.pi and 0 <= _b and _b <= np.pi) or \
                   (np.pi <= _a and _a <= 2*np.pi and np.pi <= _b and _b <= 2*np.pi):
                    result[index] = Interval(np.cos(_a), np.cos(_b))

                elif (_a <= -np.pi and -np.pi <= _b and _b <= 0) or \
                     (0 <= _a and _a <= np.pi and np.pi <= _b and _b <= 2*np.pi):
                    result[index] = Interval(-1, max(np.cos(_a), np.cos(_b)), sortQ=False)

                elif -np.pi <= _a and _a <= 0 and 0 <= _b and _b <= np.pi:
                    result[index] = Interval(min(np.cos(_a), np.cos(_b)), 1, sortQ=False)

                else:
                    result[index] = Interval(-1, 1, sortQ=False)
            return result

        else:
            raise Exception("Расчёт функции {} не предусмотрен!".format(args[0].__name__))

#     Характеристики интервала, такие как mid, rad и т.д.
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
        tmp = self._a * self._b
        if self.shape is (1, ):
            result = 0 if tmp <= 0 else min([abs(self._a), abs(self._b)])
        else:
            result = np.zeros(self.shape)
            for index in itertools.product(*self.ranges):
                result[index] = 0 if tmp[index] <= 0 else min([abs(self._a[index]), abs(self._b[index])])
        return result

    @property
    def copy(self):
        return Interval(np.copy(self._a), np.copy(self._b), sortQ=False)

    @property
    def invbar(self):
        return Interval(np.copy(self._b), np.copy(self._a), sortQ=False)

    @property
    def opp(self):
        return Interval(-np.copy(self._a), -np.copy(self._b), sortQ=False)
