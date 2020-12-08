import numpy as np
import matplotlib.pyplot as plt

from .MyClass import Interval
from .intoper import *


def non_repeat(a, decimals=12):
    """
    Функция возвращает матрицу А с различными строками.
    """

    a = np.ascontiguousarray(a)
    a = np.around(a, decimals = int(decimals))
    _, index = np.unique(a.view([('', a.dtype)]*a.shape[1]), return_index=True)
    index = sorted(index)

    return a[index]


def clear_zero_rows(a, b, decimals=12):
    """
    Функция возвращает матрицу А без строк равных нуль, а также вектор b
    соответствующий этим строкам.


    В случае если система заведома несовместна (строка матрицы нулевая, но
    при этом компонента вектора правой части, соответствующей данной строке,
    не нулевая), то возвращается ошибка.
    """

    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    a, b = np.around(a, decimals = int(decimals)), np.around(b, decimals = int(decimals+4))

    if np.sum((np.sum(a==0, axis=1)==2) & (b!=0))>0:
        raise Exception('Система несовместна!')
    else:
        index = np.where(np.sum(a==0, axis=1)!=2)
        return a[index], b[index]


def BoundaryIntervals(A, b):
    m, n = A.shape
    S = []

    for i in range(m):
        q = [float('-inf'), float('inf')]
        si = True
        dotx = (A[i]*b[i])/np.dot(A[i], A[i])

        p = np.array([-A[i,1], A[i,0]])

        for k in range(m):
            if k==i:
                continue
            Akx = np.dot(A[k], dotx)
            c = np.dot(A[k], p)

            if np.sign(c) == -1:
                q[1] = min(q[1], (b[k]-Akx)/c)
            elif np.sign(c) == 1:
                q[0] = max(q[0], (b[k]-Akx)/c)
            else:
                if Akx < b[k]:
                    if np.dot(A[k], A[i]) > 0:
                        si = False
                        break
                    else:
                        raise Exception('В системе есть пара противоречивых неравенств!')

        if q[0] > q[1]:
            si = False

        # избавление от неопределённости inf * 0
        p = p + 1e-301
        if si:
            S.append(list(dotx+p*q[0]) + list(dotx+p*q[1]) + [i])

    return np.array(S)


def ParticularPoints(S, A, b):
    PP = []
    V = S[:,:2]

    binf = ~((abs(V[:, 0]) < float("inf")) & (abs(V[:, 1]) < float("inf")))

    if len(V[binf]) != 0:
        nV = 1
        for k in S[:, 4]:
            k = int(k)
            PP.append((A[k]*b[k])/np.dot(A[k], A[k]))
    else:
        nV = 0
        PP = V

    return PP, nV, binf


def Intervals2Path(S):
    bp = np.array([S[0, 0], S[0, 1]])
    P = [bp]
    bs = bp

    while len(S) > 0:
        for k in range(len(S)):
            if max(abs(bs - np.array([S[k, 0], S[k, 1]]))) < 1e-8:
                i=k
                break

        es = np.array([S[i, 2], S[i, 3]])

        if max(abs(bs-es)) > 1e-8:
            P.append(es)

            if max(abs(bs-es)) < 1e-8:
                return np.array(P)
            bs = es
        S = np.delete(S, i, axis=0)
    return np.array(P)


__center_rm = []
def lineqs(A, b, show=True, title="Solution Set", color='gray', \
           bounds=None, alpha=0.5, s=10, size=(15,15)):
    """
    Функция визуализирует множество решений системы линейных алгебраических
    неравенств A x >= b с двумя переменными методом граничных интервалов, а
    также выводит вершины множества решений.

    Parameters:
                A: float
                    Матрица системы линейных алгебраических неравенств.

                b: float
                    Вектор правой части системы линейных алгебраических неравенств.

    Optional Parameters:
                show: bool
                    Визуализация множества решений.

                title: str
                    Верхняя легенда графика.

                color: str
                    Каким цветом осуществляется отрисовка графика.

                bounds: array_like
                    Границы отрисовочного окна.

                alpha: float
                    Прозрачность графика.

                s: float
                    Насколько велики точки вершин.

                size: tuple
                    Размер отрисовочного окна.

    Returns:
                out: list
                    Возвращается список упорядоченных вершин.
                    В случае, если show = True, то график отрисовывается.
    """

    A = np.asarray(A)
    b = np.asarray(b)

    n, m = A.shape
    assert m<=2, "В матрице A должно быть два столбца!"
    assert b.shape[0]==n, "Матрица A и правая часть b должны иметь одинаковое число строк!"

    A, b = clear_zero_rows(A, b)

    S = BoundaryIntervals(A, b)
    if len(S)==0:
        return S

    PP, nV, binf = ParticularPoints(S, A, b)

    if (np.asarray([binf])==True).any():
        if bounds is None:
            PP = np.array(PP)
            PPmin, PPmax = np.min(PP, axis=0), np.max(PP, axis=0)
            center = (PPmin + PPmax)/2
            rm = max((PPmax - PPmin)/2)
            __center_rm.append([max(abs(center) + 5*rm)])
            A = np.append(np.append(A, np.eye(2)), -np.eye(2)).reshape((len(A)+4,2))
            b = np.append(np.append(b, center-5*rm), -(center+5*rm))
            S = BoundaryIntervals(A, b)
        else:
            A = np.append(np.append(A, np.eye(2)), -np.eye(2)).reshape((len(A)+4,2))
            b = np.append(np.append(b, [bounds[0][0], bounds[1][0]]), \
                          [-bounds[0][1], -bounds[1][1]])
            S = BoundaryIntervals(A, b)

    vertices = Intervals2Path(S)

    if show:
        fig=plt.figure(figsize=size)
        ax = fig.add_subplot(111, title=title)

        x, y = vertices[:,0], vertices[:,1]
        ax.fill(x, y, linestyle = '-', linewidth = 1, color=color, alpha=alpha)
        ax.scatter(x, y, s=s, color='black', alpha=1)

    return non_repeat(vertices)


def IntLinIncR2(A, b, show=True, title="Solution Set", consistency='uni', \
                bounds=None, color='gray', alpha=0.5, s=10, size=(15,15)):
    """
    Функция визуализирует множество решений интервальной системы линейных
    алгебраических уравнений A x = b с двумя переменными методом граничных
    интервалов, а также выводит вершины множества решений.

    Parameters:
                A: Interval
                    Матрица ИСЛАУ.

                b: Interval
                    Вектор правой части ИСЛАУ.

    Optional Parameters:
                show: bool
                    Визуализация множества решений.

                title: str
                    Верхняя легенда графика.

                consistency: str
                    Параметр указывает какое множество решений (объединённое или
                    допусковое) будет выведено в ответе.

                bounds: array_like
                    Границы отрисовочного окна.

                color: str
                    Каким цветом осуществляется отрисовка графика.

                alpha: float
                    Прозрачность графика.

                s: float
                    Насколько велики точки вершин.

                size: tuple
                    Размер отрисовочного окна.

    Returns:
                out: list
                    Возвращается список упорядоченных вершин в каждом ортанте
                    начиная с первого и совершая обход в положительном направлении.
                    В случае, если show = True, то график отрисовывается.
    """

    if not isinstance(A, Interval):
        return lineqs(A, b, show=show, title=title, color=color, bounds=bounds, \
                      alpha=alpha, s=s, size=size)

    ortant = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    vertices = []
    n, m = A.shape

    assert m<=2, "В матрице A должно быть два столбца!"
    assert b.shape[0]==n, "Матрица A и правая часть b должны иметь одинаковое число строк!"

    def algo(bounds):
        for ort in range(4):
            tmp = A.copy
            WorkListA = np.zeros((2*n+m, m))
            WorkListb = np.zeros(2*n+m)

            for k in range(m):
                if ortant[ort][k] == -1:
                    tmp[:, k] = tmp[:, k].invbar
                WorkListA[2*n+k, k] = -ortant[ort][k]

            if consistency == 'uni':
                WorkListA[:n], WorkListA[n:2*n] = tmp.a, -tmp.b
                WorkListb[:n], WorkListb[n:2*n] = b.b, -b.a
            elif consistency == 'tol':
                WorkListA[:n], WorkListA[n:2*n] = -tmp.a, tmp.b
                WorkListb[:n], WorkListb[n:2*n] = -b.a, b.b
            else:
                msg = "Неверно указан тип согласования системы! Используйте 'uni' или 'tol'."
                raise Exception(msg)

            vertices.append(lineqs(-WorkListA, -WorkListb, show=False, title=title, \
                                   bounds=bounds, color=color, alpha=alpha, s=s, size=size))
    algo(bounds)

    # Если в каком-либо ортанте множество решений неограничено, то создаём
    # новое отрисовочное окно, чтобы срез решений был одинаковым.
    global __center_rm
    if len(__center_rm)>0:
        vertices = []
        _max = max(np.array(__center_rm))
        bounds = np.array([[-_max, _max], [-_max, _max]])
        algo(bounds)
    __center_rm = []

    if show:
        fig=plt.figure(figsize=size)
        ax = fig.add_subplot(111, title=title)

        for k in range(4):
            if len(vertices[k])>0:
                x, y = vertices[k][:,0], vertices[k][:,1]
                ax.fill(x, y, linestyle = '-', linewidth = 1, color=color, alpha=alpha)
                ax.scatter(x, y, s=s, color='black', alpha=1)
    return vertices
