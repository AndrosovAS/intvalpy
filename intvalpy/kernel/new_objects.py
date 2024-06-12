import numpy as np

from .real_intervals import Interval, single_type, INTERVAL_CLASSES


def zeros(shape):
    """
    To create an interval array where each element is point and equal to zero.

    Parameters:

        shape: int, tuple
            Shape of the new interval array, e.g., (2, 3) or 4.

    Returns:

        out: Interval
            An interval array of zeros with a given shape
    """
    return Interval(np.zeros(shape, dtype='float64'),
                    np.zeros(shape, dtype='float64'), sortQ=False)


def full(shape, inf, sup):
    return Interval(np.full(shape, inf, dtype='float64'),
                    np.full(shape, sup, dtype='float64'), sortQ=False)


def eye(N, M=None, k=0):
    """
    Return a 2-D interval array with ones on the diagonal and zeros elsewhere.

    Parameters:

        N: int
            Shape of the new interval array, e.g., (2, 3) or 4.

        M: int, optional
            Number of columns in the output. By default, M = N.

        k: int, optional
            Index of the diagonal: 0 refers to the main diagonal, a positive value refers
            to an upper diagonal, and a negative value to a lower diagonal. By default, k = 0.


    Returns:

        out: Interval of shape (N, M)
            An interval array where all elements are equal to zero, except for the k-th diagonal,
            whose values are equal to one.
    """


    if M is None:
        M = N
    return Interval(np.eye(N, M=M, k=k, dtype=np.int64),
                    np.eye(N, M=M, k=k, dtype=np.int64), sortQ=False)


#############################################################################################################
#############################################################################################################
def Neumaier_system(n, theta, infb=None, supb=None):
    """
    This system is a parametric interval linear system, first proposed by K. Reichmann [2],
    and then slightly modified by A. Neumaier. The matrix of the system can be both regular
    and not strongly regular for some values of the diagonal parameter. It is shown that
    n × n matrices are non-singular for theta > n provided that n is even, and, for odd order n,
    the matrices are non-singular for theta > sqrt(n^2 - 1).

    Parameters:

        n: int
            Dimension of the interval system. It may be greater than or equal to two.

        theta: float, optional
            Nonnegative real parameter, which is the number that stands on the main
            diagonal of the matrix А.

        infb: float, optional
            A real parameter that specifies the lower endpoints of the components
            of the right-hand side vector. By default, infb = -1.

        supb: float, optional
            A real parameter that specifies the upper endpoints of the components
            of the right-hand side vector. By default, supb = 1.


    Returns:

        out: Interval, tuple
            The interval matrix and interval vector of the right side are returned, respectively.
    """


    interval = [None, None]
    if infb is None:
        interval[0] = -1
    else:
        interval[0] = infb
    if supb is None:
        interval[1] = 1
    else:
        interval[1] = supb
    b = Interval([interval]*n)

    A = zeros((n,n)) + Interval(0, 2, sortQ=False)
    for k in range(n):
        A[k, k] = Interval(theta, theta, sortQ=False)

    return A, b


def Shary_system(n, N=None, alpha=0.23, beta=0.35):
    """
    One of the popular test systems is the Shary system. Due to its symmetry, it is quite simple
    to determine the structure of its united solution set as well as other solution sets.
    Changing the values of the system parameters, you can get an extensive family of interval linear
    systems for testing the numerical algorithms. As the parameter beta decreases, the matrix
    of the system becomes more and more singular, and the united solution set enlarges indefinitely.

    Parameters:

        n: int
            Dimension of the interval system. It may be greater than or equal to two.

        N: float, optional
            A real number not less than (n − 1). By default, N = n.

        alpha: float, optional
            A parameter used for specifying the lower endpoints of the elements in the interval matrix.
            The parameter is limited to 0 < alpha <= beta <= 1. By default, alpha = 0.23.

        beta: float, optional
            A parameter used for specifying the upper endpoints of the elements in the interval matrix.
            The parameter is limited to 0 < alpha <= beta <= 1. By default, beta = 0.35.

    Returns:

        out: Interval, tuple
            The interval matrix and interval vector of the right side are returned, respectively.
    """


    if N is None:
        N = n
    else:
        assert N>=n-1, "Parameter N must be a real number not less than (n − 1)."

    if alpha < 0 or beta < alpha or beta > 1:
        raise Exception('The parameter is limited to 0 < alpha <= beta <= 1.')

    A = zeros((n,n)) + Interval(alpha-1, 1-beta, sortQ=False)
    b = zeros(n) + Interval(1-n, n-1, sortQ=False)

    for k in range(n):
        A[k, k] = Interval(n-1, N, sortQ=False)

    return A, b


def Toft_system(n, r=0, R=0):
    assert r >= 0, 'The parameter r must be a positive real number.'
    assert R >= 0, 'The parameter R must be a positive real number.'

    A = zeros((n,n))
    b = zeros(n) + Interval(1-R, 1+R, sortQ=False)

    a = zeros(n)
    for k in range(n):
        A[k, k] = Interval(1-r, 1+r, sortQ=False)
        a[k] = Interval(k+1-r, k+1+r)
    A[:, -1], A[-1, :] = a.copy(), a.copy()

    return A, b

#############################################################################################################
#############################################################################################################