Using intervals
===============

This section gives an overview of the use of interval classes and examples of working with interval data.

Follow the instructions:

    >>> import numpy as np
    >>> from intvalpy import Interval, precision

It connects the interval function `Interval` from the implemented `IntvalPy` library, which was previously installed
with the command `pip install intvalpy`.

.. Contents::

The class `Interval` was created in accordance with IEEE Standard 754-2008, where rounding occurs to the nearest even number.
This allows significantly to accelerate the computational upper-level functions and reduce the computation time.
However, in tasks where you need more precision, you can switch from the standard representation of the number as
`double float` type to `mpf` type. To do this, run the following command:

    >>> precision.extendedPrecisionQ = True

You can also set the working precision (after which decimal place rounding will take place):

    >>> precision.dps(50)

The default setting is increased accuracy to 36th decimal place.


The IntvalPy library supports classical arithmetic and full Kaucher interval arithmetic.
Each arithmetic has different types of intervals (for example, classical arithmetic considers only "correct" intervals),
which means that the arithmetic operations can differ from each other. Therefore, it was decided to develop two different classes
for each of the arithmetics. However, there are not few common characteristics that these classes could inherit
from some third class. These could be the operations of taking the ends of an interval, the width or radius, and many other things.
This is why the parent class ``BaseTools'' was made separately.


Basic сharacteristics
------------

**class BaseTools(left, right)**

A parent class that contains methods that can be used to calculate the basic interval characteristics of any interval arithmetic.
Used in the `ClassicalArithmetic` and `KaucherArithmetic` classes.

**Parameters**:

* left : int, float
          The lower (left) limit of the interval.

* right : int, float
          The upper (right) limit of the interval.


**Methods**:

1. a, inf:              The operation of taking the lower (left) bound of the interval.

2. b, sup:              The operation of taking the upper (right) bound of the interval.

3. copy:                Creates a deep copy of the interval.

4. to_float:            If increased accuracy is enabled, it is sometimes necessary to convert to standard accuracy (float64)

5. wid:                 Width of the non-empty interval.

6. rad:                 Radius of the non-empty interval.

7. mid:                 Midpoint of the non-empty interval.

8. mig:                 The smallest absolute value in the non-empty interval.

9. mag:                 The greatest absolute value in the non-empty interval.

10. dual:               Dual interval.

11. pro:                Correct projection of an interval.

12. opp:                Algebraically opposite interval.

13. inv:                Algebraically opposite interval.

14. khi:                Ratschek's functional of an interval.



Interval vectors and matrices
------------

**class ArrayInterval(intervals)**

It is often necessary to consider intervals not as separate objects, but as interval vectors or matrices.
It is important to note that different intervals can be from different interval arithmetics, which leads to additional
checks when performing arithmetic operations. The IntvalPy library, using the ArrayInterval class, allows you to create
such arrays of any nesting. In fact, we use arrays created using the `numpy` library.

This class uses all the methods of the BaseTools class, but there are additional features that are common to
interval vectors and matrices.


**Parameters**:

* intervals : ndarray
          The numpy array with objects of type ClassicalArithmetic and KaucherArithmetic.


**Methods**:

1. data:                An array of interval data of type `ndarray`.

2. shape:               The elements of the shape tuple give the lengths of the corresponding interval array dimensions.

3. ndim:                Number of interval array dimensions.

4. ranges:              A list of indexes for each dimension.

5. vertex:              The set of extreme points of an interval vector.

6. T:                   View of the transposed interval array.

7. reshape(new_shape):  Gives a new shape to an interval array without changing its data.


**Examples**:

Matrix product

>>> f = Interval([
      [[-1, 3], [-2, 5]],
      [[-7, -4], [-5, 7]]
    ])
>>> s = Interval([
      [[-3, -2], [4, 4]],
      [[-7, 3], [-8, 0]]
    ])
>>> f @ s
# Interval([['[-44.0, 18.0]', '[-44.0, 28.0]']
            ['[-41.0, 56.0]', '[-84.0, 24.0]']])


Transpose

>>> f.T
# Interval([['[-1, 3]', '[-7, -4]'],
            ['[-2, 5]', '[-5, 7]']])


Create intervals
------------

**def Interval(*args, sortQ=True, midRadQ=False)**

When creating an interval, you must consider which interval arithmetic it belongs to, and how it is defined:
by means of the left and right values, through the middle and radius, or as a single object.
For this purpose, a universal function `Interval` has been implemented, which can take into account all the aspects described above.
In addition, it has a parameter for automatic conversion of the ends of an interval, so that when the user creates it, he can be sure,
that he works with the classical type of intervals.


**Parameters**:

* args : int, float, list, ndarray
          If the argument is a single one, then the intervals are set as single objects. To do this you must create
          array, each element of which is an ordered pair of the lower and upper bound of the interval.

          If the arguments are two, then the flag of the `midRadQ` parameter is taken into account. If the value is `True`,
          then the interval is set through the middle of the interval and its radius. Otherwise, the first argument will
          stand for the lower ends, and the second argument the upper ends.

* sortQ : bool, optional
          Parameter determines whether the automatic conversion of the interval ends should be performed.
          The default is `True`.

* midRadQ : bool, optional
          The parameter defines whether the interval is set through its middle and radius.
          The default is `False`.


**Examples**:

Creating intervals by specifying arrays of left and right ends of intervals

>>> a = [2, 5, -3]
>>> b = [4, 7, 1]
>>> Interval(a, b)
# Interval(['[2, 4]', '[5, 7]', '[-3, 1]'])

Now let's create the same interval vector, but in a different way

>>> Interval([ [2, 4], [5, 7], [-3, 1] ])
# Interval(['[2, 4]', '[5, 7]', '[-3, 1]'])

In case it is necessary to work with an interval object from Kaucher arithmetic, it is necessary to disable
automatic converting ends

>>> Interval(5, -2, sortQ=False)
# '[5, -2]'

As mentioned earlier, the IntvalPy library allows you to work with vectors and matrices. This automatically generates
the need to calculate the length of the array, as well as the possibility of working with collections.

>>> f = Interval([ [2, 4], [5, 7], [-3, 1] ])
>>> len(f)
# 3

To get the N-th value or several values (in the future we will call it a slice of the array) you can use quite usual tools.
Moreover, since the class `ArrayInterval` is changeable, it is also possible to change or delete elements:

>>> f[1]
# [5, 7]
>>> f[1:]
# Interval(['[5, 7]', '[-3, 1]'])
>>> f[1:] = Interval([ [-5, 5], [-10, 10] ])
>>> f
# Interval(['[2, 4]', '[-5, 5]', '[-10, 10]'])
>>> del f[1]
>>> f
# Interval([’[2, 4]’, ’[-10, 10]’])
