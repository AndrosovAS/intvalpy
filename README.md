# Interval library in Python

The Python module implements an algebraically closed system for working with intervals, solving interval systems of both
linear and nonlinear equations, and visualizing multiple solutions.

For details, see the complete documentation on [API](https://intvalpy.readthedocs.io/ru/latest/index.html).

## Installation

Make sure you have all the system-wide dependencies, then install the module itself:
```
pip install intvalpy
```

## Examples

### Visualizing solutions

We can calculate the list of vertices of the convex set described by a point the system of inequalities ``A * x >= b`` or
if an interval system of equations is considered ``A * x = b`` as well as visualize this set:

```python
import intvalpy as ip

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=2, figsize=(15,8))

A, b = ip.Shary(2)
vertices1 = ip.IntLinIncR2(A, b, show=False)
vertices2 = ip.IntLinIncR2(A, b, consistency='tol', show=False)

A = -np.array([[-3, -1],
              [-2, -2],
              [-1, -3],
              [1, -3],
              [2, -2],
              [3, -1],
              [3, 1],
              [2, 2],
              [1, 3],
              [-1, 3],
              [-2, 2],
              [-3, 1]])
b = -np.array([18,16,18,18,16,18,18,16,18,18,16,18])
vertices3 = ip.lineqs(A, b, show=False)

for k in range(len(vertices1)):
    if len(vertices1[k])>0:
        x, y = vertices1[k][:,0], vertices1[k][:,1]
        ax[0].fill(x, y, linestyle = '-', linewidth = 1, color='gray', alpha=0.5)
        ax[0].scatter(x, y, s=0, color='black', alpha=1)

for k in range(len(vertices2)):
    if len(vertices2[k])>0:
        x, y = vertices2[k][:,0], vertices2[k][:,1]
        ax[0].fill(x, y, linestyle = '-', linewidth = 1, color='blue', alpha=0.3)
        ax[0].scatter(x, y, s=10, color='black', alpha=1)

ax[0].text(-4.5, -5.5, 'United and Tolerance sets of the system Shary',
           rotation = 0,
           fontsize = 15)      

x, y = vertices3[:,0], vertices3[:,1]
ax[1].fill(x, y, linestyle = '-', linewidth = 1, color='peru', alpha=0.3)
ax[1].scatter(x, y, s=10, color='black', alpha=1)
ax[1].text(-1.5, -7.77, 'Duodecagon',
           rotation = 0,
           fontsize = 15)
```
![SolSet](https://raw.githubusercontent.com/AndrosovAS/intvalpy/master/examples/SolSet.png)

It is also possible to make a three-dimensional (two-dimensional) slice of an N-dimensional figure and see what the set of solutions looks like
with fixed N-3 (N-2) parameters. A specific implementation of the algorithm can be found in the examples.
As a result, a gif image of the united set of solutions of the system proposed by S.P. Sharym is shown below, during the evolution of the 4th unknown.

![Shary4Uni](https://raw.githubusercontent.com/AndrosovAS/intvalpy/master/examples/Shary4Uni.gif)

### Recognizing functionals:

Before we start solving a system of equations with interval data it is necessary to understand whether it is solvable or not.
To do this we consider the problem of decidability recognition, i.e. non-emptiness of the set of solutions.
In the case of an interval linear (m x n)-system of equations, we will need to solve no more than 2\ :sup:`n`
linear inequalities of size 2m+n. This follows from the fact of convexity and polyhedra of the intersection of the sets of solutions
interval system of linear algebraic equations (ISLAE) with each of the orthants of **R**\ :sup:`n` space.
Reducing the number of inequalities is fundamentally impossible, which follows from the fact that the problem is intractable,
i.e. its NP-hardness. It is clear that the above described method is applicable only for small dimensionality of the problem,
that is why the *recognizing functional method* was proposed.

After global optimization, if the value of the functional is non-negative, then the system is solvable. If the value is negative,
then the set of parameters consistent with the data is empty, but the argument delivering the maximum of the functional minimizes this inconsistency.

As an example, it is proposed to investigate the Bart-Nuding system for the emptiness/non-emptiness of the tolerance set of solutions:

```python
import intvalpy as ip

A = ip.Interval([
  [[2, 4], [-2, 1]],
  [[-1, 2], [2, 4]]
])
b = ip.Interval([[-2, 2], [-2, 2]])

tol = ip.linear.Tol(A, b, maxQ=True)
print(tol)
```

### External decision evaluation:

To obtain an optimal external estimate of the united set of solutions of an interval system linear of algebraic equations (ISLAE),
a hybrid method of splitting PSS solutions is implemented. Since the task is NP-hard, the process can be stopped by the number of iterations completed.
PSS methods are consistently guaranteeing, i.e. when the process is interrupted at any number of iterations, an approximate estimate of the solution satisfies the required estimation method.

```python
import intvalpy as ip

A, b = ip.Shary(12, N=12, alpha=0.23, beta=0.35)
pss = ip.linear.PSS(A, b)
print('pss: ', pss)
```

### Interval system of nonlinear equations:

For nonlinear systems, the simplest multidimensional interval methods of Kravchik and Hansen-Sengupta are implemented for solving nonlinear systems:

```python
import intvalpy as ip

epsilon = 0.1
def f(x):
    return ip.asinterval([x[0]**2 + x[1]**2 - 1 - ip.Interval(-epsilon, epsilon),
                          x[0] - x[1]**2])

def J(x):    
    result = [[2*x[0], 2*x[1]],
              [1, -2*x[1]]]
    return ip.asinterval(result)

ip.nonlinear.HansenSengupta(f, J, ip.Interval([0.5,0.5],[1,1]))
```

The library also provides the simplest interval global optimization:

```python
import intvalpy as ip

def levy(x):
    z = 1 + (x - 1) / 4
    t1 = np.sin( np.pi * z[0] )**2
    t2 = sum(((x - 1) ** 2 * (1 + 10 * np.sin(np.pi * x + 1) ** 2))[:-1])
    t3 = (z[-1] - 1) ** 2 * (1 + np.sin(2*np.pi * z[-1]) ** 2)
    return t1 + t2 + t3

N = 2
x = ip.Interval([-5]*N, [5]*N)
ip.nonlinear.globopt(levy, x, tol=1e-14)
```

Links
-----

* [Homepage](<https://github.com/AndrosovAS/intvalpy>)

* [Online documentation](<https://intvalpy.readthedocs.io/ru/latest/#>)

* [PyPI package](<https://pypi.org/project/intvalpy/>)

* A detailed [monograph](<http://www.nsc.ru/interval/Library/InteBooks/SharyBook.pdf>) on interval theory
