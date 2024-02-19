# This file will suggest several systems for a fuller understanding.
#
# In the case of a point system of linear inequalities A x >= b to visualize the
# set of solutions the function lineqs is provided.
#
# For interval systems, the function IntLinIncR2 is used to construct the united
# or tolerance set of solutions the function IntLinIncR2 is used.
#
#
# You can read the theory used to algorithmize the process in the following sources:
# [1] http://www.nsc.ru/interval/sharaya/Papers/ReliableComputing-19-pp-435-467.pdf (en version)
# [2] http://www.nsc.ru/interval/sharaya/Papers/Sharaya-JCT2015.pdf (ru version)
# [3] http://www-sbras.nsc.ru/interval/Library/InteBooks/SharyBook.pdf (ru version)


from intvalpy import Interval
from intvalpy import lineqs, IntLinIncR2

import numpy as np
import matplotlib.pyplot as plt

# To begin with, consider a point system:
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

vertices = lineqs(A, b, color='blue', alpha=0.2, size=(10,12))
plt.show()


# As the following example, consider the bounded set of solutions that is obtained
# from the Barth-Nuding system:
A = Interval([[2, -2],[-1, 2]], [[4,1],[2,4]])
b = Interval([-2, -2], [2, 2])

vertices = IntLinIncR2(A, b, title='Barth-Nuding', size=(10,12))
plt.show()


# In the previous examples the set of solutions was always finite, now consider a
# system that gives an infinite solution:
A = Interval([[-1, -1],[-1, -1]], [[1,1], [-1,1]])
b = Interval([1,-2], [1,2])

vertices = IntLinIncR2(A, b, title='Infinite solution', size=(10,12))
plt.show()


# Finally, consider a set of solutions degenerate into a single point:
A = Interval([[1,3],[2,5]], [[1,4],[2,5]])
b = Interval([1,2], [1,2])

vertices = IntLinIncR2(A, b, title='Single point', size=(10,12))
plt.show()
