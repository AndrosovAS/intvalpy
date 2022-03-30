# For global optimization the user can use the function globopt
#
# In this file we will look at just one function and an example of global optimization
#
#
# You can read the theory used to algorithmize the process in the following sources:
# [1] http://www.nsc.ru/interval/Education/Manuals/Bazhenov-InteAnalBasics.pdf (ru version)

from intvalpy import Interval
from intvalpy.nonlinear import globopt
import numpy as np


# Consider the Branin function
def f(x):
    return (x[1] - 5.1/(4*np.pi**2)*x[0]**2 + 5/np.pi*x[0] - 6)**2 + \
           10*(1 - 1/(8*np.pi))*np.cos(x[0]) + 10

print(globopt(f, Interval([-5,0],[10,15]), tol=1e-14))
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+ #
# globopt(f, ip.Interval([-5,0],[10,15]))                                                     #
# (interval(['[9.424778, 9.424778]', '[2.475000, 2.475000]']), [0.397887, 0.397887])          #
# +-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+ #

# As an example of how to solve another problem, you can watch the video on the website below:
# http://www.nsc.ru/interval/?page=Education/Manuals
