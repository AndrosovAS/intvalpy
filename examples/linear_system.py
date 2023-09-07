# To solve interval systems of linear algebraic equations you can use the
# following methods: Gauss, Gauss-Seidel, Rohn method, and PSS method
#
# In this file we will consider both square matrices and overdetermined systems.
#
#
# You can read the theory used to algorithmize the process in the following sources:
# [1] http://www-sbras.nsc.ru/interval/Library/InteBooks/SharyBook.pdf (ru version)
# [2] https://www.researchgate.net/publication/220252801_Enclosing_solutions_of_overdetermined_systems_of_linear_interval_equations
#
# Articles related to empirical data:
# [3] Ostanina, T. N. Modelling the dynamic growth of copper and zinc
#     dendritic deposits under the galvanostatic electrolysis conditions / T.
#     N. Ostanina, V. M. Rudoi, A. V. Patrushev, A. B. Darintseva, A. S.
#     Farlenkov // J. Electroanal. Chem. – 2015. – Vol. 750. – P. 9-18.
#
# [4] Ostanina, T. N. Determination of the surface of dendritic electrolytic
#     zinc powders and evaluation of its fractal dimension / T.N. Ostanina,
#     V.M. Rudoy, V.S. Nikitin, A.B. Darintseva, O.L. Zalesova, N.M.
#     Porotnikova // Russ. J. Non-Ferr. Met. – 2016. – Vol. 57 – P. 47–51.
#     DOI: 10.3103/S1067821216010120.

from intvalpy import Interval, zeros
from intvalpy.linear import Gauss, Gauss_Seidel, Rohn, PSS
import numpy as np

from intvalpy.linear.overdetermined import TolSolSetEstimation

# First, consider the Gauss and Gauss-Seidel methods for solving quadratic systems:
A = Interval([[2, -2],[-1, 2]], [[4, 1],[2, 4]])
b = Interval([-2, -2], [2, 2])

print('Gauss: ', Gauss(A, b))
print('Gauss_Seidel: ', Gauss_Seidel(A, b))
x = TolSolSetEstimation.Neumaier(A, b, np.array([0.0, 0.0]), np.array([1, 2]))

# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+ #
# Gauss(A, b)                                                         #
# interval(['[-5.000000, 5.000000]', '[-4.000000, 4.000000]'])        #
#                                                                     #
# Gauss-Seidel(A, b, P=True)                                          #
# interval(['[-14.000000, 14.000000]', '[-14.000000, 14.000000]'])    #
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+ #
print(A @ x in b)

A = Interval([[0.5, -0.456], [-0.438, 0.624]],
              [[1.176, 0.448], [0.596, 1.36]])
b = Interval([0.316, 0.27], [0.632, 0.624])

print('Gauss: ', Gauss(A, b))
print('Gauss_Seidel: ', Gauss_Seidel(A, b))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+ #
# Gauss(A, b)                                                         #
# interval(['[-11.094065, 13.199459]', '[-5.371444, 13.087127]'])     #
#                                                                     #
# Gauss-Seidel(A, b, P=False)                                         #
# interval(['[-4.266757, 6.076814]', '[-5.371444, 5.265456]'])        #
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+ #


# To solve overdetermined systems we can use the algorithm proposed by J. Rohn in [2]:

# Consider empirical data obtained from the description of a loose metal powder precipitate:
t = np.array([31, 69, 144, 198, 359, 446, 536, 626, 716, 809, 903, 1039, 1161, \
              1316, 1536, 2029, 2400, 29, 64, 135, 189, 261, 342, 432, 517, 613, \
              699, 792, 888, 1020, 1142, 1301, 1511, 2017, 2400, 65, 96, 521, 617, \
              705, 794, 892, 1024, 1148, 1309, 1520, 2003, 2400, 27, 83, 121, 173, \
              234, 307, 394, 467, 553, 636, 715, 806, 915, 1026, 1170, 1364, 1500, \
              2400, 77, 112, 155, 216, 367, 458, 541, 639, 741, 825, 936, 1061, \
              1189, 1200, 1426, 1902, 2400, 67, 102, 145, 206, 357, 448, 531, 629, \
              731, 815, 900, 1030, 1115, 1326, 1554, 2015, 2400]);
data = np.array([0.8, 0.84, 0.85, 0.89, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, 0.93, \
                 0.95, 0.95, 0.96, 0.97, 0.99, 1, 0.79, 0.82, 0.85, 0.89, 0.91, 0.92, \
                 0.93, 0.93, 0.94, 0.93, 0.93, 0.94, 0.95, 0.95, 0.96, 0.97, 0.99, 1, \
                 0.84, 0.84, 0.94, 0.94, 0.94, 0.95, 0.95, 0.96, 0.96, 0.97, 0.96, \
                 0.97, 1, 0.78, 0.81, 0.84, 0.89, 0.9, 0.92, 0.93, 0.92, 0.93, 0.93, \
                 0.92, 0.93, 0.95, 0.95, 0.96, 0.97, 0.98, 1, 0.8, 0.82, 0.86, 0.9, \
                 0.91, 0.93, 0.93, 0.94, 0.94, 0.93, 0.94, 0.95, 0.95, 0.96, 0.96, \
                 0.98, 1, 0.81, 0.83, 0.86, 0.9, 0.91, 0.93, 0.93, 0.94, 0.94, 0.93, \
                 0.93, 0.94, 0.94, 0.97, 0.96, 0.98, 1])

t = t + Interval(0, 0)
data = data + Interval(-0.0255, 0.0255)

A = zeros((100, 2))
A[:, 0] += Interval(1, 1)
A[:, 1] -= data
b = t * (data - 1)

print('Rohn: ', Rohn(A, b))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+ #
# overdetermined(A, b)                                                      #
# interval(['[-396.621157, 575.293503]', '[-418.434473, 687.961243]'])      #
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+ #


# To obtain the best estimate of the united set of solutions, you can use the
# method of splitting solutions, but this is a difficult NP-problem and may
# take some time.

print('PSS: ', PSS(A, b))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+ #
# PSS(A, b)                                               #
# Interval(['[155.257, 195.744]', '[205.42, 253.398]'])   #
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+ #


