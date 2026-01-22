# cython: language_level=3
# distutils: language = c++

import numpy as np
import math
from libc.math cimport fabs, exp, log, sin, cos, ceil, floor, INFINITY, NAN
from numbers import Number
cimport cython

import math
cimport cython
from libc.math cimport nextafter, nexttoward
from libc.float cimport DBL_MAX, DBL_MIN
from cython cimport fused_type


# Типы для Cython
ctypedef double double_t
ctypedef bint bool_t

# Константы
cdef double PI = 3.14159265358979323846
cdef double INF = float('inf')
cdef double NEGINF = float('-inf')
ARITHMETICS = (ClassicalArithmetic, KaucherArithmetic)


#######################################################################################
# import tounding dunction
cdef extern from "fenv.h":
    int fegetround()
    int fesetround(int rounding_mode)
    int FE_DOWNWARD
    int FE_UPWARD
    int FE_TONEAREST
    int FE_TOWARDZERO

cdef class RoundingContext:
    cdef int old_mode
    cdef int new_mode
    
    def __init__(self, int mode):
        self.new_mode = mode
        
    def __enter__(self):
        self.old_mode = fegetround()
        fesetround(self.new_mode)
        
    def __exit__(self, *args):
        fesetround(self.old_mode)

cdef double round_down(double x) nogil:
    """
    Округление вниз (к -∞).
    Возвращает наибольшее число ≤ x, представимое в double.
    """
    if x == 0.0:
        return -0.0
    elif x > 0.0:
        return nextafter(x, -DBL_MAX)
    else:
        return x

cdef double round_up(double x) nogil:
    """
    Округление вверх (к +∞).
    Возвращает наименьшее число ≥ x, представимое в double.
    """
    if x == 0.0:
        return 0.0
    elif x > 0.0:
        return x
    else:
        return nextafter(x, DBL_MAX)    
#######################################################################################


cdef class BaseTools:
    cdef int _a_int, _b_int
    cdef double_t _a_double, _b_double
    cdef bint _a_doubleQ, _b_doubleQ
   
    
    def __cinit__(self, left, right, roundQ=True):
        if roundQ:
            if isinstance(left, int):
                self._a_int = left
                self._a_doubleQ = False
            else:
                self._a_double = round_down(left)
                self._a_doubleQ = True
                
            if isinstance(right, int):
                self._b_int = right
                self._b_doubleQ = False
            else:
                self._b_double = round_up(right)
                self._b_doubleQ = True
        else:
            self._a_double = left
            self._a_doubleQ = True
            self._b_double = right
            self._b_doubleQ = True
    
    @property
    def a(self):
        """
        The largest number that is less than or equal to each of a given
        set of real numbers of an interval.
        """
        return self._a_double if self._a_doubleQ else self._a_int
    
    @property
    def b(self):
        """
        The smallest number that is greater than or equal to each of a given
        set of real numbers of an interval.
        """
        return self._b_double if self._b_doubleQ else self._b_int
    
    @property
    def inf(self):
        """
        The largest number that is less than or equal to each of a given
        set of real numbers of an interval.
        """
        return self._a_double if self._a_doubleQ else self._a_int
    
    @property
    def sup(self):
        """
        The smallest number that is greater than or equal to each of a given
        set of real numbers of an interval.
        """
        return self._b_double if self._b_doubleQ else self._b_int
    
    def copy(self, deep=True):
        if deep:
            return type(self)(self.a, self.b)
        else:
            return self
    
    @property
    def wid(self):
        """Width of the non-empty interval."""
        return self.b - self.a
    
    @property
    def rad(self):
        """Radius of the non-empty interval."""
        return 0.5 * self.wid
    
    @property
    def mid(self):
        """Midpoint of the non-empty interval."""
        return 0.5 * (self.b + self.a)
    
    @property
    def mig(self):
        """The smallest absolute value in the non-empty interval."""
        if 0 in self:
            return 0.0
        else:
            return min(fabs(self.a), fabs(self.b))
    
    @property
    def mag(self):
        """The greatest absolute value in the non-empty interval."""
        return max(fabs(self.a), fabs(self.b))
    
    @property
    def dual(self):
        if self.b <= self.a:
            return ClassicalArithmetic(self.b, self.a, roundQ=False)
        else:
            return KaucherArithmetic(self.b, self.a, roundQ=False)
    
    @property
    def pro(self):
        if isinstance(self, ClassicalArithmetic):
            return ClassicalArithmetic(self.a, self.b, roundQ=False)
        else:
            return ClassicalArithmetic(min(self.a, self.b), max(self.a, self.b), roundQ=False)
    
    @property
    def opp(self):
        if self.b <= self.a:
            return ClassicalArithmetic(-self.a, -self.b, roundQ=False)
        else:
            return KaucherArithmetic(-self.a, -self.b, roundQ=False)
    
    @property
    def inv(self):
        cdef double_t inf, sup
        with RoundingContext(FE_DOWNWARD):
            inf = 1.0/self.a
        with RoundingContext(FE_UPWARD):
            sup = 1.0/self.b
        if self.b <= self.a:
            return ClassicalArithmetic(inf, sup)
        else:
            return KaucherArithmetic(inf, sup)
    
    @property
    def khi(self):
        if fabs(self.a) <= fabs(self.b):
            return self.a / self.b
        else:
            return self.b / self.a
    
    def __repr__(self):
        return "'[%.6g, %.6g]'" % (self.a, self.b)
    
    # Unary operations
    def __neg__(self):
        return type(self)(-self.b, -self.a, roundQ=False)
    
    def __abs__(self):
        return type(self)(self.mig, self.mag)
    
    def __contains__(self, other):
        if isinstance(other, Number):
            return (self.a <= other) and (other <= self.b)
        elif isinstance(other, ARITHMETICS):
            return (self.a <= other.a) and (other.b <= self.b)
        else:
            TypeError('Not supported type.')
                        
    def __lt__(self, other):
        if isinstance(other, Number):
            return (self.a < other) and (self.b < other)
        elif isinstance(other, ARITHMETICS):
            return (self.a < other.a) and (self.b < other.b)
        else:
            raise TypeError('Not supported type.')
    
    def __le__(self, other):
        if isinstance(other, Number):
            return (self.a <= other) and (self.b <= other)
        elif isinstance(other, ARITHMETICS):
            return (self.a <= other.a) and (self.b <= other.b)
        else:
            raise TypeError('Not supported type.')
    
    def __eq__(self, other):
        if isinstance(other, Number):
            return (self.a == other) and (self.b == other)
        elif isinstance(other, ARITHMETICS):
            return (self.a == other.a) and (self.b == other.b)
        else:
            raise TypeError('Not supported type.')
    
    def __ne__(self, other):
        if isinstance(other, Number):
            return (self.a != other) or (self.b != other)
        elif isinstance(other, ARITHMETICS):
            return (self.a != other.a) or (self.b != other.b)
        else:
            raise TypeError('Not supported type.')
    
    def __gt__(self, other):
        if isinstance(other, Number):
            return (self.a > other) and (self.b > other)
        elif isinstance(other, ARITHMETICS):
            return (self.a > other.a) and (self.b > other.b)
        else:
            raise TypeError('Not supported type.')
    
    def __ge__(self, other):
        if isinstance(other, Number):
            return (self.a >= other) and (self.b >= other)
        elif isinstance(other, ARITHMETICS):
            return (self.a >= other.a) and (self.b >= other.b)
        else:
            raise TypeError('Not supported type.')
    
    def sqrt(self):
        return np.exp(0.5 * np.log(self))

    def exp(self):
        try:
            return type(self)(math.exp(self.a), math.exp(self.b))
        except OverflowError:
            return type(self)(INF, INF)

    def log(self):
        pro = self.pro
        _max, _min = max(pro.a, 0.0), pro.b
        if _max <= _min:
            inf = NEGINF if _max == 0.0 else math.log(_max)
            sup = NEGINF if _min == 0.0 else math.log(_min)
            if self.a <= self.b:
                return type(self)(inf, sup)
            else:
                return type(self)(sup, inf)
        else:
            return type(self)(NAN, NAN)

    def sin(self):
        x = self.pro
        sin_inf, sin_sup = math.sin(x.a), math.sin(x.b)
        
        if ceil((x.a + PI/2.0) / (2.0*PI)) <= floor((x.b + PI/2.0) / (2.0*PI)):
            inf = -1
        else:
            inf = min(sin_inf, sin_sup)
        
        if ceil((x.a - PI/2.0) / (2.0*PI)) <= floor((x.b - PI/2.0) / (2.0*PI)):
            sup = 1
        else:
            sup = max(sin_inf, sin_sup)
        
        if self.a <= self.b:
            return type(self)(inf, sup)
        else:
            return type(self)(sup, inf)

    def cos(self):
        x = self.pro
        cos_inf, cos_sup = math.cos(x.a), math.cos(x.b)
        
        if ceil((x.a - PI) / (2.0*PI)) <= floor((x.b - PI) / (2.0*PI)):
            inf = -1
        else:
            inf = min(cos_inf, cos_sup)
        
        if ceil(x.a / (2.0*PI)) <= floor(x.b / (2.0*PI)):
            sup = 1
        else:
            sup = max(cos_inf, cos_sup)
        
        if self.a <= self.b:
            return type(self)(inf, sup)
        else:
            return type(self)(sup, inf)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): # TODO, debug
        cdef double_t inf, sup
        if ufunc.__name__ == 'add':
            return self.__radd__(inputs[1])
        elif ufunc.__name__ == 'subtract':
            return type(self)(-self.b, -self.a).__radd__(inputs[1])
        elif ufunc.__name__ == 'multiply':
            return self.__rmul__(inputs[1])
        elif ufunc.__name__ in ['true_divide', 'divide']:
            with RoundingContext(FE_DOWNWARD):
                inf = 1.0/self.b
            with RoundingContext(FE_UPWARD):
                sup = 1.0/self.a
            return type(self)(inf, sup, roundQ=False).__rmul__(inputs[1])
        elif ufunc.__name__ == 'sqrt': 
            return self.sqrt()
        elif ufunc.__name__ == 'exp':
            return self.exp()
        elif ufunc.__name__ == 'log': 
            return self.log()
        elif ufunc.__name__ == 'sin':
            return self.sin()
        elif ufunc.__name__ == 'cos':
            return self.cos()        
        else:
            raise NotImplementedError(f"Calculation of the {ufunc.__name__} function is not provided!")


cdef class ClassicalArithmetic(BaseTools):
    def __add__(self, other):
        cdef double_t inf, sup
        if isinstance(other, (ClassicalArithmetic, KaucherArithmetic)):
            with RoundingContext(FE_DOWNWARD):
                inf = self.a + other.a
            with RoundingContext(FE_UPWARD):
                sup = self.b + other.b
            return type(other)(inf, sup, roundQ=False)
        else:
            return other + self
    
    def __sub__(self, other):
        cdef double_t inf, sup
        if isinstance(other, (ClassicalArithmetic, KaucherArithmetic)):
            with RoundingContext(FE_DOWNWARD):
                inf = self.a - other.b
            with RoundingContext(FE_UPWARD):
                sup = self.b - other.a
            return type(other)(inf, sup, roundQ=False)
        else:
            return -other + self
    
    def __mul__(self, other):
        cdef double_t inf, sup
        cdef double_t selfInfPlus, selfSupPlus, otherInfPlus, otherSupPlus
        cdef double_t selfInfMinus, selfSupMinus, otherInfMinus, otherSupMinus
        
        if isinstance(other, ClassicalArithmetic):
            with RoundingContext(FE_DOWNWARD):
                inf = min(self.a * other.a, self.a * other.b, self.b * other.a, self.b * other.b)
            with RoundingContext(FE_UPWARD):
                sup = max(self.a * other.a, self.a * other.b, self.b * other.a, self.b * other.b)
            return ClassicalArithmetic(inf, sup, roundQ=False)
        
        elif isinstance(other, KaucherArithmetic):
            selfInfPlus = max(self.a, 0.0)
            selfSupPlus = max(self.b, 0.0)
            selfInfMinus = selfInfPlus - self.a
            selfSupMinus = selfSupPlus - self.b
            
            otherInfPlus = max(other.a, 0.0)
            otherSupPlus = max(other.b, 0.0)
            otherInfMinus = otherInfPlus - other.a
            otherSupMinus = otherSupPlus - other.b
            
            with RoundingContext(FE_DOWNWARD):
                inf = selfInfPlus * otherInfPlus + selfSupMinus * otherSupMinus - selfSupPlus * otherInfMinus - selfInfMinus * otherSupPlus
            with RoundingContext(FE_UPWARD):
                sup = selfSupPlus * otherSupPlus + selfInfMinus * otherInfMinus - selfInfPlus * otherSupMinus - selfSupMinus * otherInfPlus
            return KaucherArithmetic(inf, sup, roundQ=False)
        
        else:
            return other * self
    
    def __truediv__(self, other):
        cdef double_t inf, sup
        if isinstance(other, ClassicalArithmetic):
            if 0.0 in other:
                raise ValueError('It is impossible to divide by zero containing intervals.')
            with RoundingContext(FE_DOWNWARD):
                inf = min(self.a / other.a, self.a / other.b, self.b / other.a, self.b / other.b)
            with RoundingContext(FE_UPWARD):
                sup = max(self.a / other.a, self.a / other.b, self.b / other.a, self.b / other.b)
            return ClassicalArithmetic(inf, sup, roundQ=False)
        
        if isinstance(other, KaucherArithmetic):
            if 0.0 in other:
                raise ValueError('It is impossible to divide by zero containing intervals.')
            with RoundingContext(FE_DOWNWARD):
                inf = 1.0/other.b
            with RoundingContext(FE_UPWARD):
                sup = 1.0/other.a
            return self.__mul__(KaucherArithmetic(inf, sup, roundQ=False))
        
        else:
            return 1.0/other * self
    
    def __pow__(self, other):
        if isinstance(other, (int, np.int_)) and other >= 0:
            with RoundingContext(FE_DOWNWARD):
                inf = min(self.a ** other, self.b ** other)
            with RoundingContext(FE_UPWARD):
                sup = max(self.a ** other, self.b ** other)
            if (other % 2 == 0) and (0 in self):
                return ClassicalArithmetic(0.0, sup, roundQ=False)
            else:
                return ClassicalArithmetic(inf, sup, roundQ=False)
        elif self >= 0:
            return np.exp(other * np.log(self))
        else:
            raise ValueError('If the base contains negative numbers, then the degree can only be a natural number.')
    
    def __radd__(self, other):
        cdef double_t inf, sup
        if isinstance(other, Number):
            with RoundingContext(FE_DOWNWARD):
                inf = self.a + other
            with RoundingContext(FE_UPWARD):
                sup = self.b + other
            return ClassicalArithmetic(inf, sup, roundQ=False)
        else:
            # Для массивов
            return np.vectorize(lambda o: o + self)(other)
    
    def __rsub__(self, other):
        cdef double_t inf, sup
        if isinstance(other, Number):
            with RoundingContext(FE_DOWNWARD):
                inf = other - self.b
            with RoundingContext(FE_UPWARD):
                sup = other - self.a
            return ClassicalArithmetic(inf, sup, roundQ=False)
        else:
            return np.vectorize(lambda o: o - self)(other)
    
    def __rmul__(self, other):
        cdef double_t inf, sup
        if isinstance(other, Number):
            with RoundingContext(FE_DOWNWARD):
                inf = min(other * self.a, other * self.b)
            with RoundingContext(FE_UPWARD):
                sup = max(other * self.a, other * self.b)
            return ClassicalArithmetic(inf, sup, roundQ=False)
        else:
            return np.vectorize(lambda o: o * self)(other)
    
    def __rtruediv__(self, other):
        cdef double_t inf, sup
        if 0.0 in self:
            raise ValueError('It is impossible to divide by zero containing intervals!')
        with RoundingContext(FE_DOWNWARD):
            inf = 1.0/self.b
        with RoundingContext(FE_UPWARD):
            sup = 1.0/self.a
        return other * ClassicalArithmetic(inf, sup, roundQ=False)


cdef class KaucherArithmetic(BaseTools):
    def _add__(self, other):
        cdef double_t inf, sup
        if isinstance(other, (ClassicalArithmetic, KaucherArithmetic)):
            with RoundingContext(FE_DOWNWARD):
                inf = self.a + other.a
            with RoundingContext(FE_UPWARD):
                sup = self.b + other.b
            return KaucherArithmetic(inf, sup, roundQ=False)
        else:
            return other + self
    
    def __sub__(self, other):
        cdef double_t inf, sup
        if isinstance(other, (ClassicalArithmetic, KaucherArithmetic)):
            with RoundingContext(FE_DOWNWARD):
                inf = self.a - other.b
            with RoundingContext(FE_UPWARD):
                sup = self.b - other.a
            return KaucherArithmetic(inf, sup, roundQ=False)
        else:
            return -other + self
    
    def __mul__(self, other):
        cdef double_t inf, sup
        cdef double_t selfInfPlus, selfSupPlus, otherInfPlus, otherSupPlus
        cdef double_t selfInfMinus, selfSupMinus, otherInfMinus, otherSupMinus
        
        if isinstance(other, (ClassicalArithmetic, KaucherArithmetic)):
            selfInfPlus = max(self.a, 0.0)
            selfSupPlus = max(self.b, 0.0)
            selfInfMinus = selfInfPlus - self.a
            selfSupMinus = selfSupPlus - self.b
            
            otherInfPlus = max(other.a, 0.0)
            otherSupPlus = max(other.b, 0.0)
            otherInfMinus = otherInfPlus - other.a
            otherSupMinus = otherSupPlus - other.b
            
            with RoundingContext(FE_DOWNWARD):
                inf = max(selfInfPlus * otherInfPlus, selfSupMinus * otherSupMinus) - max(selfSupPlus * otherInfMinus, selfInfMinus * otherSupPlus)
            with RoundingContext(FE_UPWARD):
                sup = max(selfSupPlus * otherSupPlus, selfInfMinus * otherInfMinus) - max(selfInfPlus * otherSupMinus, selfSupMinus * otherInfPlus)
            return KaucherArithmetic(inf, sup, roundQ=False)
        
        else:
            return other * self
    
    def __truediv__(self, other):
        cdef double_t inf, sup
        cdef double_t selfInfPlus, selfSupPlus, otherInfPlus, otherSupPlus
        cdef double_t selfInfMinus, selfSupMinus, otherInfMinus, otherSupMinus
        
        if isinstance(other, ClassicalArithmetic):
            if 0.0 in other:
                raise ValueError('It is impossible to divide by zero containing intervals!')
            
            with RoundingContext(FE_DOWNWARD):
                inf = 1.0/other.b
            with RoundingContext(FE_UPWARD):
                sup = 1.0/other.a
            other = ClassicalArithmetic(inf, sup, roundQ=False)
            
            selfInfPlus = max(self.a, 0.0)
            selfSupPlus = max(self.b, 0.0)
            selfInfMinus = selfInfPlus - self.a
            selfSupMinus = selfSupPlus - self.b
            
            otherInfPlus = max(other.a, 0.0)
            otherSupPlus = max(other.b, 0.0)
            otherInfMinus = otherInfPlus - other.a
            otherSupMinus = otherSupPlus - other.b
            
            with RoundingContext(FE_DOWNWARD):
                inf = selfInfPlus * otherInfPlus + selfSupMinus * otherSupMinus - selfSupPlus * otherInfMinus - selfInfMinus * otherSupPlus
            with RoundingContext(FE_UPWARD):
                sup = selfSupPlus * otherSupPlus + selfInfMinus * otherInfMinus - selfInfPlus * otherSupMinus - selfSupMinus * otherInfPlus
            return KaucherArithmetic(inf, sup, roundQ=False)
        
        if isinstance(other, KaucherArithmetic):
            if 0.0 in other:
                raise ValueError('It is impossible to divide by zero containing intervals!')
            with RoundingContext(FE_DOWNWARD):
                inf = 1.0/other.b
            with RoundingContext(FE_UPWARD):
                sup = 1.0/other.a
            return self.__mul__(KaucherArithmetic(inf, sup, roundQ=False))
        
        else:
            return 1/other * self
    
    def __pow__(self, other): #TODO, debug
        cdef double_t inf, sup
        if isinstance(other, (int, np.int_)) and other >= 0:
            with RoundingContext(FE_DOWNWARD):
                inf = self.a ** other
            with RoundingContext(FE_UPWARD):
                sup = self.b ** other
            if (other % 2 == 0) and (0 in self):
                return ClassicalArithmetic(0.0, max(inf, sup))
            elif (other % 2 == 0) and (0 in self.pro):
                return KaucherArithmetic(max(inf, sup), 0.0)
            else:
                if inf > sup:
                    return KaucherArithmetic(inf, sup)
                else:
                    return ClassicalArithmetic(inf, sup)
        elif self >= 0:
            return np.exp(other * np.log(self))
        else:
            raise ValueError('If the base contains negative numbers, then the degree can only be a natural number.')
    
    def __radd__(self, other):
        cdef double_t inf, sup
        if isinstance(other, Number):
            with RoundingContext(FE_DOWNWARD):
                inf = self.a + other
            with RoundingContext(FE_UPWARD):
                sup = self.b + other
            return KaucherArithmetic(inf, sup, roundQ=False)
        else:
            return np.vectorize(lambda o: o + self)(other)
    
    def __rsub__(self, other):
        cdef double_t inf, sup
        if isinstance(other, Number):
            with RoundingContext(FE_DOWNWARD):
                inf = other - self.b
            with RoundingContext(FE_UPWARD):
                sup = other - self.a
            return KaucherArithmetic(inf, sup, roundQ=False)
        else:
            return np.vectorize(lambda o: o - self)(other)
    
    def __rmul__(self, other):
        cdef double_t inf, sup
        cdef double_t selfInfPlus, selfSupPlus, otherInfPlus, otherSupPlus
        cdef double_t selfInfMinus, selfSupMinus, otherInfMinus, otherSupMinus
        
        if isinstance(other, Number):
            selfInfPlus = max(self.a, 0.0)
            selfSupPlus = max(self.b, 0.0)
            selfInfMinus = selfInfPlus - self.a
            selfSupMinus = selfSupPlus - self.b
            
            otherInfPlus = max(other, 0.0)
            otherSupPlus = otherInfPlus
            otherInfMinus = otherInfPlus - other
            otherSupMinus = otherSupPlus - other
            
            with RoundingContext(FE_DOWNWARD):
                inf = selfInfPlus * otherInfPlus + selfSupMinus * otherSupMinus - selfSupPlus * otherInfMinus - selfInfMinus * otherSupPlus
            with RoundingContext(FE_UPWARD):
                sup = selfSupPlus * otherSupPlus + selfInfMinus * otherInfMinus - selfInfPlus * otherSupMinus - selfSupMinus * otherInfPlus
            return KaucherArithmetic(inf, sup, roundQ=False)
        else:
            return np.vectorize(lambda o: o * self)(other)
    
    def __rtruediv__(self, other):
        cdef double_t inf, sup
        if 0.0 in self:
            raise ValueError('It is impossible to divide by zero containing intervals!')
        with RoundingContext(FE_DOWNWARD):
            inf = 1.0/self.b
        with RoundingContext(FE_UPWARD):
            sup = 1.0/self.a
        return other * KaucherArithmetic(inf, sup, roundQ=False)
