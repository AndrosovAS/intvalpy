# To solve interval nonlinear systems you can use the Krawczyk or Hansen-Sengupta methods
#
# In this file we will consider both individual equations and multivariate systems.
#
#
# You can read the theory used to algorithmize the process in the following sources:
# [1] http://www-sbras.nsc.ru/interval/Library/InteBooks/SharyBook.pdf (ru version)
# [2] http://www.nsc.ru/interval/Education/Manuals/Bazhenov-InteAnalBasics.pdf (ru version)


from intvalpy import Interval, asinterval
from intvalpy.nonlinear import Krawczyk, HansenSengupta
import numpy as np

# First, let's look at the one-dimensional equation:
def f(x):
    return np.sin(x)**2 - x/5 - 1

def df(x):
    return 2*np.sin(x)*np.cos(x) - 1/5

x = Interval(-1.5, -1)
print('Krawczyk: ', Krawczyk(f, df, x))
# +-----+-----+-----+-----+-----+-----+-----+ #
# Krawczyk(f, df, x)                          #
# [-1.085983, -1.085983]                      #
# +-----+-----+-----+-----+-----+-----+-----+ #


# For the multidimensional case, consider the system from the textbook [2]:
epsilon = 0
def f(x):
    return asinterval([x[0]**2 + x[1]**2 - 1 - Interval(-epsilon, epsilon),
                       x[0] - x[1]**2])

def J(x):
    result = [[2*x[0], 2*x[1]],
              [1, -2*x[1]]]
    return asinterval(result)

print('Multidimensional Krawczyk: ', Krawczyk(f, J, Interval([0,0],[1,1])))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+ #
# Krawczyk(f, J, Interval([0,0],[1,1]))                                                       #
# Спектральный радиус матрицы ρ(|I - Λ·L|) = 1.31 больше единицы!                             #
# Multidimensional Krawczyk:  interval(['[0.618034, 0.618034]', '[0.786151, 0.786151]'])      #
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+ #


epsilon = 0.1
print('Multidimensional HansenSengupta: ', HansenSengupta(f, J, Interval([0.5,0.5],[1,1])))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+ #
# HansenSengupta(f, J, Interval([0.5,0.5],[1,1]))                                                   #
# Multidimensional HansenSengupta:  interval(['[0.569485, 0.666583]', '[0.755144, 0.817158]'])      #
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+ #
