Using Intervals
===============

This section provides an overview of interval classes and demonstrates how to work with interval data.

To begin, import the necessary modules:

.. code-block:: python

    >>> import numpy as np
    >>> from intvalpy import Interval

This imports the `Interval` class from the `IntvalPy` library, which should be installed beforehand 
using `pip install intvalpy`.

.. Contents::

The `Interval` class is designed in accordance with the IEEE 1788-2015 standard for floating-point arithmetic 
and implements **directed rounding** (rounding downward and upward). This guarantees that the computed 
intervals always enclose the exact mathematical result, ensuring reliability in numerical computations 
and providing rigorous error bounds.

The IntvalPy library supports both classical interval arithmetic and full Kaucher interval arithmetic. 
Each arithmetic operates with different types of intervals (for example, classical arithmetic considers 
only "proper" intervals), which means that the arithmetic operations may differ between the two approaches. 
Therefore, it was decided to implement two separate classes, one for each arithmetic.

However, there are many common characteristics that these classes can inherit from a shared base class, 
such as extracting the endpoints of an interval, computing its width or radius, and various other operations. 
This is why the parent class ``BaseTools`` was created separately.


Basic Characteristics
---------------------

**class BaseTools(left, right)**

A parent class that contains methods for calculating the basic interval characteristics for any interval arithmetic. 
It is used in the `ClassicalArithmetic` and `KaucherArithmetic` classes.

**Parameters**:

* **left** : int, float
    The lower (left) limit of the interval.
* **right** : int, float
    The upper (right) limit of the interval.

**Methods**:

1.  **a, inf**: Returns the lower (left) bound of the interval.
2.  **b, sup**: Returns the upper (right) bound of the interval.
3.  **copy()**: Creates a deep copy of the interval.
4.  **wid()**: Width of the non-empty interval.
5.  **rad()**: Radius of the non-empty interval.
6.  **mid()**: Midpoint of the non-empty interval.
7.  **mig()**: The smallest absolute value in the non-empty interval (mignitude).
8.  **mag()**: The greatest absolute value in the non-empty interval (magnitude).
9. **dual()**: Returns the dual interval.
10. **pro()**: Returns the correct projection of an interval.
11. **opp()**: Returns the algebraically opposite interval.
12. **inv()**: Returns the algebraically inverse interval.
13. **khi()**: Returns Ratschek's functional of an interval.


Interval Vectors and Matrices
-----------------------------

**class ArrayInterval(intervals)**

It is often necessary to consider intervals not as separate objects, but as interval vectors or matrices. 
It is important to note that different intervals can belong to different interval arithmetics, which leads 
to additional checks when performing arithmetic operations. The IntvalPy library, using the `ArrayInterval` 
class, allows you to create such arrays of any nesting. In practice, this class uses arrays created with 
the `numpy` library.

This class utilizes all the methods of the `BaseTools` class, but also provides additional features common 
to interval vectors and matrices.

**Parameters**:

* **intervals** : ndarray
    A numpy array with objects of type `ClassicalArithmetic` and `KaucherArithmetic`.

**Methods**:

1.  **data**: An array of interval data of type `ndarray`.
2.  **shape**: The shape tuple, giving the lengths of the corresponding interval array dimensions.
3.  **ndim**: The number of interval array dimensions.
4.  **ranges**: A list of index ranges for each dimension.
5.  **vertex()**: Returns the set of extreme points (vertices) of an interval vector.
6.  **T**: Returns a view of the transposed interval array.
7.  **reshape(new_shape)**: Gives a new shape to an interval array without changing its data.

**Examples**:

**Matrix Product**

.. code-block:: python

    >>> f = Interval([
    ...   [[-1, 3], [-2, 5]],
    ...   [[-7, -4], [-5, 7]]
    ... ])
    >>> s = Interval([
    ...   [[-3, -2], [4, 4]],
    ...   [[-7, 3], [-8, 0]]
    ... ])
    >>> f @ s
    Interval([['[-44.0, 18.0]', '[-44.0, 28.0]'],
              ['[-41.0, 56.0]', '[-84.0, 24.0]']])

**Transpose**

.. code-block:: python

    >>> f.T
    Interval([['[-1, 3]', '[-7, -4]'],
              ['[-2, 5]', '[-5, 7]']])


Creating Intervals
------------------

**def Interval(*args, sortQ=True, midRadQ=False)**

When creating an interval, you must consider which interval arithmetic it belongs to and how it is defined: 
by left and right endpoints, by midpoint and radius, or as a single array object. For this purpose, a universal 
function `Interval` has been implemented. It also has a parameter for automatic conversion of the interval 
endpoints, ensuring that the user works with the classical type of intervals by default.

**Parameters**:

* ***args** : int, float, list, ndarray
    - If a single argument is provided, the intervals are set as a single object. To do this, you must create 
      an array where each element is an ordered pair of the lower and upper bound of an interval.
    - If two arguments are provided, the `midRadQ` flag is taken into account. If `midRadQ=True`, the interval 
      is set by its midpoint and radius. Otherwise, the first argument represents the lower endpoints, 
      and the second argument represents the upper endpoints.

* **sortQ** : bool, optional
    Determines whether the automatic conversion of the interval ends should be performed to ensure `left <= right`. 
    The default is `True`.

* **midRadQ** : bool, optional
    Defines whether the interval is set by its midpoint and radius. The default is `False`.

**Examples**:

**Creating intervals by specifying arrays of left and right endpoints**

.. code-block:: python

    >>> a = [2, 5, -3]
    >>> b = [4, 7, 1]
    >>> Interval(a, b)
    Interval(['[2, 4]', '[5, 7]', '[-3, 1]'])

**Creating the same interval vector in a different way**

.. code-block:: python

    >>> Interval([ [2, 4], [5, 7], [-3, 1] ])
    Interval(['[2, 4]', '[5, 7]', '[-3, 1]'])

**Creating an interval from Kaucher arithmetic (disabling automatic endpoint sorting)**

.. code-block:: python

    >>> Interval(5, -2, sortQ=False)
    '[5, -2]'

**Working with Vectors and Matrices**

.. code-block:: python

    >>> f = Interval([ [2, 4], [5, 7], [-3, 1] ])
    >>> len(f)
    3
    >>> f[1]
    [5, 7]
    >>> f[1:]
    Interval(['[5, 7]', '[-3, 1]'])
    >>> f[1:] = Interval([ [-5, 5], [-10, 10] ])
    >>> f
    Interval(['[2, 4]', '[-5, 5]', '[-10, 10]'])
    >>> del f[1]
    >>> f
    Interval(['[2, 4]', '[-10, 10]'])
