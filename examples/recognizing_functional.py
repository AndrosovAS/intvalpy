# Before solving a system of equations, we want to know if there is a solution.
# For this purpose, there are recognizing functionals.
#
# In this file, we consider two recognizing functionals. One for the united set of
# solutions, and one for the tolerance set.
#
#
# You can read the theory used to algorithmize the process in the following sources:
# [1] http://www.nsc.ru/interval/shary/Papers/SharyAiT.pdf (ru version)
# [2] http://www.nsc.ru/interval/shary/Papers/Sharys-JCT2013.pdf (ru version)
# [3] http://www.nsc.ru/interval/shary/Papers/SShary-JCT-2017.pdf (ru version)
# [4] http://www.nsc.ru/interval/shary/Slides/SShary-WeakStrong.pdf (en version)

from intvalpy import Interval, Uni, Tol
import numpy as np

# As an example, consider the Barth-Nuding system:
A = Interval([[2, -2],[-1, 2]], [[4,1],[2,4]])
b = Interval([-2, -2], [2, 2])

# Take a random point in space and see if it is a solution
x = np.random.uniform(0, 1, 2)
print('Uni: ', Uni(A, b, x))
print('Tol: ', Tol(A, b, x))

# However, we are interested not just in a random point, but in whether the system
# as a whole is solvable. To do this, we maximize the functionals.
# Since the system is linear, the functionals are concave and any method that does
# not use a gradient can be used as an optimization method.
# This code uses the Nelder-Mead method.

print('Uni: ', Uni(A, b, maxQ=True))
print('Tol: ', Tol(A, b, maxQ=True))
