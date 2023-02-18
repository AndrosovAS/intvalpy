General purpose functions
=========================

In this section, we present an overview of functions for working with interval quantities as well as some functions
for creating interval objects.

Run the following commands to connect the necessary modules

    >>> import intvalpy as ip
    >>> import numpy as np

.. Contents::


Converting data to interval type
--------------------------------

**def asinterval(a)**

To convert the input data to the interval type, use the `asinterval` function:

**Parameters**:

* a : int, float, array_like
        Input data in any form that can be converted to an interval data type.
        These include int, float, list and ndarrays.

**Returns**:

* out : Interval
    The conversion is not performed if the input is already of type Interval.
    Otherwise an object of interval type is returned.


**Examples**:

>>> ip.asinterval(3)
'[3, 3]'
>>> ip.asinterval([1/2, ip.Interval(-2, 5), 2])
Interval(['[0.5, 0.5]', '[-2, 5]', '[2, 2]'])


Interval scatterplot
--------------------

Математическая диаграмма, изображающая значения двух переменных в виде брусов на декартовой плоскости.

**Parameters**:

* x : Interval
            Интервальный вектор положения данных на оси OX.

* y : Interval
            Интервальный вектор положения данных на оси OY.

* title: str, optional
            Верхняя легенда графика.

* color: str, optional
            Цвет отображения брусов.

* alpha: float, optional
            Прозрачность брусов.

* s: float, optional
            Насколько велики точки вершин.

* size: tuple, optional
            Размер отрисовочного окна.

* save: bool, optional
            Если значение True, то график сохраняется.


**Returns**:

* out: None
            A scatterplot is displayed.


**Examples**:

>>> x = ip.Interval(np.array([1.06978355, 1.94152571, 1.70930717, 2.94775725, 4.55556349, 6, 6.34679035, 6.62305275]), \
>>>                 np.array([1.1746937 , 2.73256075, 1.95913956, 3.61482169, 5.40818299, 6, 7.06625362, 7.54738552]))
>>> y = ip.Interval(np.array([0.3715678 , 0.37954135, 0.38124681, 0.39739009, 0.42010472, 0.45, 0.44676075, 0.44823645]), \
>>>                 np.array([0.3756708 , 0.4099036 , 0.3909104 , 0.42261893, 0.45150898, 0.45, 0.47255936, 0.48118948]))
>>> ip.scatter_plot(x, y)


Intersection of intervals
-------------------------

Функция ``intersection`` осуществляет пересечение интервальных данных. В случае, если на вход поданы массивы, то осуществляется покомпонентное пересечение.

Parameters:
            A, B: ``Interval``
                В случае, если операнды не являются интервальным типом, то
                они преобразуются функцией ``asinterval``.

Returns:
            out: ``Interval``
                Возвращается массив пересечённых интервалов.
                Если некоторые интервалы не пересекаются, то на их месте
                выводится интервал ``Interval(float('-inf'), float('-inf'))``.

Примеры:

>>> import intvalpy as ip
>>> f = ip.Interval([-3., -6., -2.], [0., 5., 6.])
>>> s = ip.Interval(-1, 10)
>>> ip.intersection(f, s)
interval(['[-1.0, 0.0]', '[-1.0, 5.0]', '[-1.0, 6.0]'])

>>> f = ip.Interval([-3., -6., -2.], [0., 5., 6.])
>>> s = -2
>>> ip.intersection(f, s)
interval(['[-2.0, -2.0]', '[-2.0, -2.0]', '[-2.0, -2.0]'])

>>> f = ip.Interval([-3., -6., -2.], [0., 5., 6.])
>>> s = ip.Interval([ 2., -8., -6.], [6., 7., 0.])
>>> ip.intersection(f, s)
interval(['[-inf, -inf]', '[-6.0, 5.0]', '[-2.0, 0.0]'])


Distance
------------

**def dist(x, y, order=float('inf'))**

To calculate metrics or multimetrics in interval spaces, the `dist` function is provided.
The mathematical formula for distance is given as follows:
dist\ :sub:`order` = (sum\ :sub:`ij` ||x\ :sub:`ij` - y\ :sub:`ij` ||\ :sup:`order` )\ :sup:`1/order`.

It is important to note that this formula involves an algebraic difference, not the usual interval difference.

**Parameters**:

* a, b : Interval
          The intervals between which you need to calculate the distance. In the case of multidimensional
          operands a multimetric is calculated.

* order : int, optional
          The order of the metric is set. By default, setting is Chebyshev distance.


**Returns**:

* out: float
          The distance between the input operands is returned.


**Examples**:

>>> f = ip.Interval([
        [[0, 1], [2, 3]],
        [[4, 5], [6, 7]],
    ])
>>> s = ip.Interval([
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]],
    ])
>>> ip.dist(f, s)
1.0

The detailed information about various metrics can be found in the referenced `monograph <http://www.nsc.ru/interval/Library/InteBooks/SharyBook.pdf>`_.


Zero intervals
--------------

**def zeros(shape)**

To create an interval array where each element is point and equal to zero, the function `zeros` is provided:

**Parameters**:

* shape : int, tuple
            Shape of the new interval array, e.g., (2, 3) or 4.

**Returns**:

* out : Interval
            An interval array of zeros with a given shape


**Examples**:

>>> ip.zeros((2, 3))
Interval([['[0, 0]', '[0, 0]', '[0, 0]'],
          ['[0, 0]', '[0, 0]', '[0, 0]']])
>>> ip.zeros(4)
Interval(['[0, 0]', '[0, 0]', '[0, 0]', '[0, 0]'])


Identity interval matrix
--------------

**def eye(N, M=None, k=0)**

Return a 2-D interval array with ones on the diagonal and zeros elsewhere.

**Parameters**:

* N : int
          Shape of the new interval array, e.g., (2, 3) or 4.

* M : int, optional
          Number of columns in the output. By default, M = N.

* k : int, optional
          Index of the diagonal: 0 refers to the main diagonal, a positive value refers
          to an upper diagonal, and a negative value to a lower diagonal. By default, k = 0.


**Returns**:

* out : Interval of shape (N, M)
          An interval array where all elements are equal to zero, except for the k-th diagonal,
          whose values are equal to one.


**Examples**:

>>> ip.eye(3, M=2, k=-1)
Interval([['[0, 0]', '[0, 0]'],
          ['[1, 1]', '[0, 0]'],
          ['[0, 0]', '[1, 1]']])


Diagonal of the interval matrix
--------------

**def diag(v, k=0)**

Extract a diagonal or construct a diagonal interval array.

**Parameters**:

* v : Interval
          If v is a 2-D interval array, return a copy of its k-th diagonal.
          If v is a 1-D interval array, return a 2-D interval array with v on the k-th diagonal.

* k : int, optional
          Diagonal in question. Use k>0 for diagonals above the main diagonal, and k<0 for diagonals
          below the main diagonal. By default, k=0.


**Returns**:

* out : Interval
          The extracted diagonal or constructed diagonal interval array.


**Examples**:

>>> A, b = ip.Shary(3)
>>> ip.diag(A)
Interval(['[2, 3]', '[2, 3]', '[2, 3]'])
>>> ip.diag(b)
Interval([['[-2, 2]', '[0, 0]', '[0, 0]'],
          ['[0, 0]', '[-2, 2]', '[0, 0]'],
          ['[0, 0]', '[0, 0]', '[-2, 2]']])


Elementary mathematical functions
---------------------
This section presents the basic elementary mathematical functions that are most commonly encountered
in various kinds of applied problems.


The square root
~~~~~~~~~~~~~~~~

**def sqrt(x)**

Interval enclosure of the square root intrinsic over an interval.

**Parameters**:

* x : Interval
        The values whose square-roots are required.


**Returns**:

* out : Interval
        An array of the same shape as x, containing the interval enclosure of the square root
        of each element in x.


**Examples**:

>>> f = ip.Interval([[-3, -1], [-3, 2], [0, 4]])
>>> ip.sqrt(f)
Interval(['[nan, nan]', '[0, 1.41421]', '[0, 2]'])


The exponent
~~~~~~~~~~~~~~~~

**def exp(x)**

Interval enclosure of the exponential intrinsic over an interval.

**Parameters**:

* x : Interval
        The values to take the exponent from.


**Returns**:

* out : Interval
        An array of the same shape as x, containing the interval enclosure of the exponential
        of each element in x.


**Examples**:

>>> f = ip.Interval([[-3, -1], [-3, 2], [0, 4]])
>>> ip.exp(f)
Interval(['[0.0497871, 0.367879]', '[0.0497871, 7.38906]', '[1, 54.5982]'])


The natural logarithm
~~~~~~~~~~~~~~~~

**def log(x)**

Interval enclosure of the natural logarithm intrinsic over an interval.

**Parameters**:

* x : Interval
        The values to take the natural logarithm from.


**Returns**:

* out : Interval
        An array of the same shape as x, containing the interval enclosure of the natural logarithm
        of each element in x.


**Examples**:

>>> f = ip.Interval([[-3, -1], [-3, 2], [1, 4]])
>>> ip.log(f)
Interval(['[nan, nan]', '[-inf, 0.693147]', '[0, 1.38629]'])


The sine function
~~~~~~~~~~~~~~~~

**def sin(x)**

Interval enclosure of the sin intrinsic over an interval.

**Parameters**:

* x : Interval
        The values to take the sin from.


**Returns**:

* out : Interval
        An array of the same shape as x, containing the interval enclosure of the sin
        of each element in x.


**Examples**:

>>> f = ip.Interval([[-3, -1], [-3, 2], [0, 4]])
>>> ip.sin(f)
Interval(['[-1, -0.14112]', '[-1, 1]', '[-0.756802, 1]'])


The cosine function
~~~~~~~~~~~~~~~~

**def cos(x)**

Interval enclosure of the cos intrinsic over an interval.

**Parameters**:

* x : Interval
        The values to take the cos from.


**Returns**:

* out : Interval
        An array of the same shape as x, containing the interval enclosure of the cos
        of each element in x.


**Examples**:

>>> f = ip.Interval([[-3, -1], [-3, 2], [0, 4]])
>>> ip.cos(f)
Interval(['[-0.989992, 0.540302]', '[-0.989992, 1]', '[-1, 1]'])



Test interval systems
---------------------
To check the performance of each implemented algorithm, it is tested on well-studied test systems.
This subsection describes some of these systems, for which the properties of the solution sets are known,
and their analytical characteristics and the complexity of numerical procedures have been previously studied.


The Shary system
~~~~~~~~~~~~~~~~

**def Shary(n, N=None, alpha=0.23, beta=0.35)**

One of the popular test systems is the Shary system. Due to its symmetry, it is quite simple to determine
the structure of its united solution set as well as other solution sets. Changing the values of the system
parameters, you can get an extensive family of interval linear systems for testing the numerical algorithms.
As the parameter beta decreases, the matrix of the system becomes more and more singular, and the united solution
set enlarges  indefinitely.

**Parameters**:

* n : int
            Dimension of the interval system. It may be greater than or equal to two.

* N : float, optional
            A real number not less than (n − 1). By default, N = n.

* alpha : float, optional
            A parameter used for specifying the lower endpoints of the elements in the interval matrix.
            The parameter is limited to 0 < alpha <= beta <= 1. By default, alpha = 0.23.

* beta : float, optional
            A parameter used for specifying the upper endpoints of the elements in the interval matrix.
            The parameter is limited to 0 < alpha <= beta <= 1. By default, beta = 0.35.


**Returns**:

* out: Interval, tuple
            The interval matrix and interval vector of the right side are returned, respectively.


**Examples**:

>>> A, b = ip.Shary(3)
>>> print('A: ', A)
>>> print('b: ', b)
A:  Interval([['[2, 3]', '[-0.77, 0.65]', '[-0.77, 0.65]'],
          ['[-0.77, 0.65]', '[2, 3]', '[-0.77, 0.65]'],
          ['[-0.77, 0.65]', '[-0.77, 0.65]', '[2, 3]']])
b:  Interval(['[-2, 2]', '[-2, 2]', '[-2, 2]'])


The Neumaier-Reichmann system
~~~~~~~~~~~~~~~~~~~~~~~~~

**def Neumeier(n, theta, infb=None, supb=None)**

This system is a parametric interval linear system, first proposed by K. Reichmann [2], and then
slightly modified by A. Neumaier. The matrix of the system can be regular, but not strongly regular
for some values of the diagonal parameter. It is shown that n × n matrices are non-singular
for theta > n provided that n is even, and, for odd order n, the matrices are non-singular
for theta > sqrt(n^2 - 1).

**Parameters**:

* n : int
            Dimension of the interval system. It may be greater than or equal to two.

* theta : float, optional
            Nonnegative real parameter, which is the number that stands on the main diagonal of the matrix А.

* infb : float, optional
            A real parameter that specifies the lower endpoints of the components of the right-hand
            side vector. By default, infb = -1.

* supb : float, optional
            A real parameter that specifies the upper endpoints of the components of the right-hand
            side vector. By default, supb = 1.


**Returns**:

* out: Interval, tuple
            The interval matrix and interval vector of the right side are returned, respectively.


**Examples**:

>>> A, b = ip.Neumaier(2, 3.5)
>>> print('A: ', A)
>>> print('b: ', b)
A:  Interval([['[3.5, 3.5]', '[0, 2]'],
          ['[0, 2]', '[3.5, 3.5]']])
b:  Interval(['[-1, 1]', '[-1, 1]'])



References
~~~~~~~~~~

[1] S.P. Shary - `On optimal solution of interval linear equations <http://www-sbras.nsc.ru/interval/shary/Papers/SharySINUM.pdf>`_ // SIAM Journal on Numerical Analysis. – 1995. – Vol. 32, No. 2. – P. 68–630.

[2] Reichmann K. Abbruch beim Intervall-Gauß-Algorithmus // Computing. – 1979. – Vol. 22, Issue 4. – P. 355–361.

[3] С.П. Шарый - `Конечномерный интервальный анализ <http://www.nsc.ru/interval/Library/InteBooks/SharyBook.pdf>`_.
    Sergey P. Shary, `Finite-Dimensional Interval Analysis`_.
