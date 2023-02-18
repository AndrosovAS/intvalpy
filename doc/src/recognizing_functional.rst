Recognizing functionals
===============

This paragraph presents an overview of functions for investigating solvability
of the set of solutions of interval linear systems.

Run the following commands to connect the necessary modules

    >>> import intvalpy as ip
    >>> ip.precision.extendedPrecisionQ = False
    >>> import numpy as np

Before we start solving a system of equations with interval data it is necessary to understand
whether it is solvable or not. To do this we consider the problem of decidability recognition,
i.e. non-emptiness of the set of solutions. In the case of an interval linear (m x n)-system
of equations, we will need to solve no more than 2\ :sup:`n` linear inequalities of size 2m+n.
This follows from the fact of convexity and polyhedra of the intersection of the sets of solutions
interval system of linear algebraic equations (ISLAU) with each of the orthants of **R**\ :sup:`n` space.
Reducing the number of inequalities is fundamentally impossible, which follows from the fact
that the problem is intractable, i.e. its NP-hardness. It is clear that the above described
method is applicable only for small dimensionality of the problem, that is why the *recognizing
functional method* was proposed.

.. Content::


Tol for linear systems
------------

**class Tol**

To check the interval system of linear equations for its strong compatibility,
the recognizing functional Tol should be used.


Value at the point
~~~~~~~~~~~~~~~~~~

** class Tol.value(A, b, x, weight=None)**

The function computes the value of the recognizing functional at the point x.

**Parameters**:

* A: Interval
    The input interval matrix of ISLAE, which can be either square or rectangular.

* b: Interval
    The interval vector of the right part of the ISLAE.

* x: np.array, optional
    The point at which the recognizing functional is calculated.

* weight: float, np.array, optional
    The vector of weight coefficients for each forming of the recognizing functional.
    By default, it is a vector consisting of ones.


**Returns**:

* out: float
    The value of the recognizing functional at the point x.


**Examples**:

As an example, consider the well-known interval system proposed by Barth-Nuding:

>>> A = ip.Interval([
        [[2, 4], [-1, 2]],
        [[-2, 1], [2, 4]]
    ])
>>> b = ip.Interval([[-2, 2], [-2, 2]])

To get the value of a function at a specific point, perform the following instruction

>>> x = np.array([1, 2])
>>> ip.linear.Tol.value(A, b, x)
-7.0

The point x does not lie in the tolerable solution set for this system, because the value
of the recognizing functional is negative.


Finding a global maximum
~~~~~~~~~~~~~~~~~~

** class Tol.maximize(A, b, x0=None, weight=None, linear_constraint=None, **kwargs)**

The function is intended for finding the global maximum of the recognizing functional.
The ralgb5 subgradient method is used for optimization.

**Parameters**:

* A: Interval
    The input interval matrix of ISLAE, which can be either square or rectangular.

* b: Interval
    The interval vector of the right part of the ISLAE.

* x0: np.array, optional
    The initial assumption is at what point the maximum is reached. By default, x0
    is equal to the vector which is the solution (pseudo-solution) of the system
    mid(A) x = mid(b).

* weight: float, np.array, optional
    The vector of weight coefficients for each forming of the recognizing functional.
    By default, it is a vector consisting of ones.

* linear_constraint: LinearConstraint, optional
    System (lb <= C <= ub) describing linear dependence between parameters.
    By default, the problem of unconditional maximization is being solved.

* kwargs: optional params
    The ralgb5 function uses additional parameters to adjust its performance.
    These parameters include the step size, the stopping criteria, the maximum number
    of iterations and others. Specified in the function description ralgb5.


**Returns**:

* out: tuple
    The function returns the following values in the specified order:
    1. the vector solution at which the recognition functional reaches its maximum,
    2. the value of the recognition functional,
    3. the number of iterations taken by the algorithm,
    4. the number of calls to the calcfg function,
    5. the exit code of the algorithm (1 = tolf, 2 = tolg, 3 = tolx, 4 = maxiter, 5 = error).


**Examples**:

To identify whether the data is strong compatibility, optimization must be performed:

>>> A = ip.Interval([
        [[2, 4], [-1, 2]],
        [[-2, 1], [2, 4]]
    ])
>>> b = ip.Interval([[-2, 2], [-2, 2]])
>>> ip.linear.Tol.maximize(A, b)
(array([0., 0.]), 2.0, 29, 46, 1)

The distinguishing feature of the `Tol` functional from the `Uni` and `Uss` functional
is that regardless of whether the matrix **A** interval or point matrix, the functional
always has only one extremum. Thus it does not matter which initial guess to start the search with.
However, if one specifies an initial point, the search for a global maximum can be accelerated.

In addition, conditional optimization with linear constraints has been implemented using
the penalty function method.

>>> A = ip.Interval([
        [[2, 4], [10, 11.99999]],
        [[-2, 1], [2, 4]]
    ])
>>> b = ip.Interval([[-2, 2], [-2, 2]]) + 0.15

>>> C = np.array([
        [1, 0],
        [0, 1]
    ])
>>> ub = np.array([5, 5])
>>> lb = np.array([0, 0.1])

>>> linear_constraint = ip.LinearConstraint(C, ub=ub, lb=lb)
>>> ip.linear.Tol.maximize(A, b, linear_constraint=linear_constraint, tolx=1e-20)
(array([3.48316025e-17, 1.00000000e-01]), 0.9500009999999999, 114, 288, 1)



Uni for linear systems
------------

**class Uni**

To check the interval system of linear equations for its weak compatibility,
the recognizing functional Uni should be used.


Value at the point
~~~~~~~~~~~~~~~~~~

** class Uni.value(A, b, x, weight=None)**

The function computes the value of the recognizing functional at the point x.

**Parameters**:

* A: Interval
    The input interval matrix of ISLAE, which can be either square or rectangular.

* b: Interval
    The interval vector of the right part of the ISLAE.

* x: np.array, optional
    The point at which the recognizing functional is calculated.

* weight: float, np.array, optional
    The vector of weight coefficients for each forming of the recognizing functional.
    By default, it is a vector consisting of ones.


**Returns**:

* out: float
    The value of the recognizing functional at the point x.


**Examples**:

As an example, consider the well-known interval system proposed by Barth-Nuding:

>>> A = ip.Interval([
        [[2, 4], [-1, 2]],
        [[-2, 1], [2, 4]]
    ])
>>> b = ip.Interval([[-2, 2], [-2, 2]])

To get the value of a function at a specific point, perform the following instruction

>>> x = np.array([1, 2])
>>> ip.linear.Uni.value(A, b, x)
0.0

The point x does lie in the united solution set for this system, because the value
of the recognizing functional is not negative.


Finding a global maximum
~~~~~~~~~~~~~~~~~~

** class Uni.maximize(A, b, x0=None, weight=None, linear_constraint=None, **kwargs)**

The function is intended for finding the global maximum of the recognizing functional.
The ralgb5 subgradient method is used for optimization.

**Parameters**:

* A: Interval
    The input interval matrix of ISLAE, which can be either square or rectangular.

* b: Interval
    The interval vector of the right part of the ISLAE.

* x0: np.array, optional
    The initial assumption is at what point the maximum is reached. By default, x0
    is equal to the vector which is the solution (pseudo-solution) of the system
    mid(A) x = mid(b).

* weight: float, np.array, optional
    The vector of weight coefficients for each forming of the recognizing functional.
    By default, it is a vector consisting of ones.

* linear_constraint: LinearConstraint, optional
    System (lb <= C <= ub) describing linear dependence between parameters.
    By default, the problem of unconditional maximization is being solved.

* kwargs: optional params
    The ralgb5 function uses additional parameters to adjust its performance.
    These parameters include the step size, the stopping criteria, the maximum number
    of iterations and others. Specified in the function description ralgb5.


**Returns**:

* out: tuple
    The function returns the following values in the specified order:
    1. the vector solution at which the recognition functional reaches its maximum,
    2. the value of the recognition functional,
    3. the number of iterations taken by the algorithm,
    4. the number of calls to the calcfg function,
    5. the exit code of the algorithm (1 = tolf, 2 = tolg, 3 = tolx, 4 = maxiter, 5 = error).


**Examples**:

To identify whether the data is weak compatibility, optimization must be performed:

>>> A = ip.Interval([
        [[2, 4], [-1, 2]],
        [[-2, 1], [2, 4]]
    ])
>>> b = ip.Interval([[-2, 2], [-2, 2]])
>>> ip.linear.Uni.maximize(A, b)
(array([0., 0.]), 2.0, 29, 45, 1)

However, we know from theory that even in the linear case the recognizing function Uni
is not a concave function on the whole investigated space. Thus, there is no guarantee
that the global maximum of the function, and not the local extremum, was found using
the optimization algorithm.

As some solution, the user can specify an initial guess, based, for example, on the features
of the matrix. This can also speed up the process of finding the global maximum.

In addition, conditional optimization with linear constraints has been implemented using
the penalty function method.

>>> A = ip.Interval([
        [[2, 4], [10, 11.99999]],
        [[-2, 1], [2, 4]]
    ])
>>> b = ip.Interval([[-2, 2], [-2, 2]]) + 0.15

>>> C = np.array([
        [1, 0],
        [0, 1]
    ])
>>> ub = np.array([5, 5])
>>> lb = np.array([0, 0.1])

>>> linear_constraint = ip.LinearConstraint(C, ub=ub, lb=lb)
>>> ip.linear.Uni.maximize(A, b, linear_constraint=linear_constraint, tolx=1e-20)
(array([1.47928518e-17, 1.00000000e-01]), 1.15, 110, 259, 1)


References
~~~~~~~~~~~~~~~~~~

[1] С.П. Шарый - `Разрешимость интервальных линейных уравнений и анализ данных с неопределённостями <http://www.nsc.ru/interval/shary/Papers/SharyAiT.pdf>`_ // Автоматика и Телемеханика, No 2, 2012

[2] С.П. Шарый, И.А. Шарая - `Распознавание разрешимости интервальных уравнений и его приложения к анализу данных <http://www.nsc.ru/interval/shary/Papers/Sharys-JCT2013.pdf>`_ // Вычислительные технологии, Том 18, No 3, 2013, стр. 80-109.

[3] С.П. Шарый - `Сильная согласованность в задаче восстановления зависимостей при интервальной неопределённости данных <http://www.nsc.ru/interval/shary/Papers/SShary-JCT-2017.pdf>`_ // Вычислительные технологии, Том 22, No 2, 2017, стр. 150-172.
