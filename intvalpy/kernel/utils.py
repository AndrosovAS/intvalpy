import numpy as np

from .real_intervals import Interval, ARITHMETICS
from .preprocessing import asinterval


infinity = float('inf')
nan = np.nan


#############################################################################################################
#############################################################################################################


def dist(x, y, order=infinity):
    """
    Метрика в интервальных пространствах.

    Parameters:
                a, b: Interval
                    Интервалы между которыми необходимо рассчитать dist.
                    В случае многомерности операндов вычисляется мультиметрика.

                order: int
                    Задаются различные метрики. По умолчанию используется
                    Чебышёвское расстояние.

    Returns:
                out: float
                    Возвращается расстояние между входными операндами.
    """

    def cheb(x, y):
        return np.maximum(abs(x.a - y.a), abs(x.b - y.b))

    if order == infinity:
        return np.amax(cheb(x, y))
    elif isinstance(order, int):
        return pow(np.sum(cheb(x, y) ** order), 1/order)
    else:
        raise Exception('Не верно задан порядок нормы order.')


def diag(v, k=0):
    """
    Extract a diagonal or construct a diagonal interval array.

    Parameters:

        v: Interval
            If v is a 2-D interval array, return a copy of its k-th diagonal.
            If v is a 1-D interval array, return a 2-D interval array with v on the k-th diagonal.

        k: int, optional
            Diagonal in question. Use k>0 for diagonals above the main diagonal, and k<0 for diagonals
            below the main diagonal. By default, k=0.

    Returns:

        out: Interval
            The extracted diagonal or constructed diagonal interval array.
    """

    return asinterval(np.diag(v.data, k=k))


def compmat(A):
    """
    Компарант интервальной матрицы
    """
    Amag = A.mag
    Amig = A.mig
    return np.array([
        [Amig[k, l] if k==l else -Amag[k, l] for l in range(A.shape[1])]
        for k in range(A.shape[0])
    ])


def isnan(x):
    def _isnan(x):
        isnanQ = np.isnan(float(x.a)) or np.isnan(float(x.b))
        return isnanQ
    if isinstance(x, ARITHMETICS):
        return _isnan(x)
    else:
        return np.vectorize(_isnan)(x.data)
    

def wid(x):
    def _wid(x):
        if isinstance(x, ARITHMETICS):
            return x.wid
        else:
            return x
    
    if hasattr(x, '__iter__'):
        return np.vectorize(_wid)(x)
    else:
        return _wid(x)
    

def mid(x):
    def _mid(x):
        if isinstance(x, ARITHMETICS):
            return x.mid
        else:
            return x
    
    if hasattr(x, '__iter__'):
        return np.vectorize(_mid)(x)
    else:
        return _mid(x)
    

def rad(x):
    def _rad(x):
        if isinstance(x, ARITHMETICS):
            return x.rad
        else:
            return x
    
    if hasattr(x, '__iter__'):
        return np.vectorize(_rad)(x)
    else:
        return _rad(x)
    

def inf(x):
    def _inf(x):
        if isinstance(x, ARITHMETICS):
            return x.inf
        else:
            return x
    
    if hasattr(x, '__iter__'):
        return np.vectorize(_inf)(x)
    else:
        return _inf(x)
    

def sup(x):
    def _sup(x):
        if isinstance(x, ARITHMETICS):
            return x.sup
        else:
            return x
    
    if hasattr(x, '__iter__'):
        return np.vectorize(_sup)(x)
    else:
        return _sup(x)
    

def mag(x):
    def _mag(x):
        if isinstance(x, ARITHMETICS):
            return x.mag
        else:
            return x
    
    if hasattr(x, '__iter__'):
        return np.vectorize(_mag)(x)
    else:
        return _mag(x)


subset = lambda a, b: np.array(((a.a >= b.a) & (a.b <= b.b)), dtype=np.bool).all()
superset = lambda a, b: subset(b, a)

proper_subset = lambda a, b: np.array(((a.a > b.a) & (a.b < b.b)), dtype=np.bool).all()
proper_superset = lambda a, b: proper_subset(b, a)

contain = lambda a, b: np.array(((a.a >= b.a) & (a.b <= b.b)), dtype=np.bool)
supercontain = lambda a, b: subset(b, a)