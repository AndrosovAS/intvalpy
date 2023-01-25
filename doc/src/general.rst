General purpose functions
=========================

In this section, we present an overview of functions for working with interval quantities as well as some functions for creating interval objects.

Run the following commands to connect the necessary modules

    >>> import intvalpy as ip
    >>> import numpy as np

.. Contents::

Converting data to interval type
--------------------------------

To convert the input data to the interval type, use the ``asinterval`` function:
  
Parameters:
            a: ``array_like``
                Input data, in any form, that can be converted to an array of intervals. 
                These include ``int``, ``float``, ``list`` and ``ndarrays``. 

Returns:
            out: ``Interval``
                The conversion is not performed if the input is already of type ``Interval``.
                If a is ``int``, ``float``, ``list`` or ``ndarrays``, then an object is returned 
                of the base class ``Interval``.
                

Examples: 

>>> import intvalpy as ip
>>> data = 3
>>> ip.asinterval(data)
[3.000000, 3.000000]

>>> data = [1/3, ip.Interval(-2, 5), 2]
>>> ip.asinterval(data)
interval(['[0.333333, 0.333333]', '[-2.0, 5.0]', '[2.0, 2.0]'])


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


Метрика
------------

Для вычисления метрики или мультиметрики в интервальных пространствах предусмотрена функция ``dist``:


Parameters:
            a, b: ``Interval``
                Интервалы между которыми необходимо рассчитать ``dist``.
                В случае многомерности операндов вычисляется мультиметрика.

            order: ``int``
                Задаются различные метрики. По умолчанию используется Чебышёвское расстояние.

Returns:
            out: ``float``
                Возвращается расстояние между входными операндами.

Пример:

>>> import intvalpy as ip
>>> f = ip.Interval([[0, 2], [4, 6]],
>>>                 [[1, 3], [5, 7]])
>>> s = ip.Interval([[1, 3], [5, 7]],
>>>                 [[2, 4], [6, 8]])
>>> ip.dist(f, s)
1.0

The detailed information about various metrics can be found in the referenced `monograph <http://www.nsc.ru/interval/Library/InteBooks/SharyBook.pdf>`_.


Zero intervals 
--------------

To create an interval array where each element is point and equal to zero, the function ``zeros`` is provided: 

>>> import intvalpy as ip
>>> ip.zeros((2, 3))
interval([['[0.0, 0.0]', '[0.0, 0.0]', '[0.0, 0.0]'],
          ['[0.0, 0.0]', '[0.0, 0.0]', '[0.0, 0.0]']])


Test interval systems
---------------------
To check the performance of each implemented algorithm, it is tested on well-studied test systems. This subsection describes some of these systems, for which the properties of the solution sets are known, and their analytical characteristics and the complexity of numerical procedures have been previously studied. 


The Shary system
~~~~~~~~~~~~~~~~

One of the popular test systems is the Shary system. Due to its symmetry, it is quite simple to determine the structure of its united solution set as well as other solution sets. Changing the values of the system parameters, you can get an extensive family of interval linear systems for testing the numerical algorithms. As the parameter beta decreases, the matrix of the system becomes more and more singular, and the united solution set enlarges  indefinitely. 

**Parameters**:

* n : int
            Dimension of the interval system. It may be greater than or equal to two. 

* N : float, optional
            A real number not less than (n − 1). By default, N = n. 

* alpha : float, optional
            A parameter used for specifying the lower endpoints of the elements in the interval matrix. The parameter is limited 
            to 0 < alpha <= beta <= 1. By default, alpha = 0.23. 

* beta : float, optional
            A parameter used for specifying the upper endpoints of the elements in the interval matrix. The parameter is limited 
            to 0 < alpha <= beta <= 1. By default, beta = 0.35. 
          

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


Neumaier-Reichmann system
~~~~~~~~~~~~~~~~~~~~~~~~~

This system is a parametric interval linear system, first proposed by K. Reichmann, and then slightly modified by A. Neumeier. The matrix of the system can be both regular and not strongly regular for some values of the diagonal parameter. 
It is shown that n × n matrices are non-singular for theta > n provided that n is even, and, for odd order n, the matrices are non-singular for theta > sqrt(n^2 - 1). 
  
**Parameters**:

* n : int
            Dimension of the interval system. It may be greater than or equal to two. 

* theta : float, optional
            Nonnegative real parameter, which is the number that stands on the main diagonal of the matrix А.

* infb : float, optional
            A real parameter that specifies the lower endpoints of the components of the right-hand side vector. By default, infb = -1.

* supb : float, optional
            A real parameter that specifies the upper endpoints of the components of the right-hand side vector. By default, supb = 1. 


**Returns**:

* out: Interval, tuple
            The interval matrix and interval vector of the right side are returned, respectively.


**Examples**:

>>> A, b = ip.Neumeier(2, 3.5)
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
