General purpose functions
=========================

In this section, we present an overview of functions for working with interval quantities as well as some functions for creating interval objects.

Run the following commands to connect the necessary modules

    >>> import intvalpy as ip
    >>> import numpy as np

.. Содержание::

Преобразование данных в интервальный тип
------------

Для того, чтобы преобразовать входные данные в интервальный тип следует воспользоваться функцией ``asinterval``:

Parameters:
            a: ``array_like``
                Входные данные, в любой форме, которые могут быть преобразованы в массив интервалов.
                Это включает в себя ``int``, ``float``, ``list`` и ``ndarrays``.

Returns:
            out: ``Interval``
                Преобразование не выполняется, если входные данные уже являются типом ``Interval``.
                Если a - ``int``, ``float``, ``list`` или ``ndarrays``, то возвращается
                базовый класс ``Interval``.

Примеры:

>>> import intvalpy as ip
>>> data = 3
>>> ip.asinterval(data)
[3.000000, 3.000000]

>>> data = [1/3, ip.Interval(-2, 5), 2]
>>> ip.asinterval(data)
interval(['[0.333333, 0.333333]', '[-2.0, 5.0]', '[2.0, 2.0]'])


Интервальная диаграмма рассеяния
------------

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
            Отображается диаграмма рассеяния.


**Examples**:

>>> x = ip.Interval(np.array([1.06978355, 1.94152571, 1.70930717, 2.94775725, 4.55556349, 6, 6.34679035, 6.62305275]), \
>>>                 np.array([1.1746937 , 2.73256075, 1.95913956, 3.61482169, 5.40818299, 6, 7.06625362, 7.54738552]))
>>> y = ip.Interval(np.array([0.3715678 , 0.37954135, 0.38124681, 0.39739009, 0.42010472, 0.45, 0.44676075, 0.44823645]), \
>>>                 np.array([0.3756708 , 0.4099036 , 0.3909104 , 0.42261893, 0.45150898, 0.45, 0.47255936, 0.48118948]))
>>> ip.scatter_plot(x, y)


Пересечение интервалов
------------

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

Наиболее подробную информацию о различных метриках можно узнать из указанной `монографии <http://www.nsc.ru/interval/Library/InteBooks/SharyBook.pdf>`_.


Интервал из нулей
------------

Для создания интервального массива данных, где каждый интервал точечный и имеет значение нуль, предусмотрена функция ``zeros``:

>>> import intvalpy as ip
>>> ip.zeros((2, 3))
interval([['[0.0, 0.0]', '[0.0, 0.0]', '[0.0, 0.0]'],
          ['[0.0, 0.0]', '[0.0, 0.0]', '[0.0, 0.0]']])


Test interval systems
---------------------
Для проверки работоспособности каждый реализованный алгоритм тестируется на хорошо изученных тестовых системах. В данном подразделе предложены
некоторые из таких систем, в каждой из которых известны свойства, аналитическое решение, а также трудоёмкость решения.


The Shary system
~~~~~~~~~~~~~~~~

Первой предложенной системой является система С.П. Шарого. В силу симметрии достаточно просто определить структуру объединённого множества решений.
А с помощью изменения значений параметров системы можно получить обширный набор ИСЛАУ для тестирования реализованных алгоритмов. Видно, что при
уменьшении параметра beta матрица становится все больше особенной, а множество решений неограниченно увеличивается.

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


Система Ноймайера-Райхмана
~~~~~~~~~~~~~~~~~~

Данная система является параметрической системой, которая была предложена Ноймайером-Райхманом. Класс матриц, которые составляют левую часть,
способны продемонстрировать, что результат произведения двух неособенных матриц может дать особенную матрицу — невозможная ситуация в классической линейной алгебре.
Показано, что матрицы чётных размеров n × n неособенны при theta > n, а для нечётного порядка n матрицы неособенны при theta > sqrt(n^2 - 1).

**Parameters**:

* n : int
            Размерность интервальной системы. Может быть больше либо равным двум.

* theta : float, optional
            Неотрицательный вещественный параметр, который является значением стоящим на главной диагонали матрицы А.

* infb : float, optional
            Вещественный параметр который совпадает с каждым левым концом из вектора правой части. По умолчанию infb = -1.

* supb : float, optional
            Вещественный параметр который совпадает с каждым правым концом из вектора правой части. По умолчанию supb = 1.


**Returns**:

* out: Interval, tuple
            Возвращаются интервальная матрица и интервальный вектор правой части соответсвенно.


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
