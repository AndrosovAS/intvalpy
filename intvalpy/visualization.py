import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .RealInterval import Interval


def Unique(a, decimals=12):
    a = np.ascontiguousarray(a)
    a = np.around(a, decimals=int(decimals))
    _, index = np.unique(a.view([('', a.dtype)]*a.shape[1]), return_index=True)
    index = sorted(index)

    return a[index]


def non_repeat(a, b):
    a = np.copy(np.ascontiguousarray(a))
    a = np.around(a, decimals=15)
    a1 = (a.T - b).T
    _, index = np.unique(a1.view([('', a1.dtype)]*a1.shape[1]), return_index=True)
    index = sorted(index)

    return a[index], b[index]


def clear_zero_rows(a, b, ndim=2):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    a, b = np.around(a, decimals=15), np.around(b, decimals=15)

    cnmty = True
    if np.sum((np.sum(abs(a) <= 1e-15, axis=1) == ndim) & (b > 0)) > 0:
        cnmty = False

    index = np.where(np.sum(abs(a) <= 1e-15, axis=1) != ndim)
    return a[index], b[index], cnmty


def BoundaryIntervals(A, b):
    m, n = A.shape
    S = []

    for i in range(m):
        q = [float('-inf'), float('inf')]
        si = True
        dotx = (A[i]*b[i])/np.dot(A[i], A[i])

        p = np.array([-A[i, 1], A[i, 0]])

        for k in range(m):
            if k == i:
                continue
            Akx = np.dot(A[k], dotx)
            c = np.dot(A[k], p)

            if np.sign(c) == -1:
                tmp = (b[k] - Akx) / c
                q[1] = q[1] if q[1] <= tmp else tmp
            elif np.sign(c) == 1:
                tmp = (b[k] - Akx) / c
                q[0] = q[0] if tmp < q[0] else tmp
            else:
                if Akx < b[k]:
                    if np.dot(A[k], A[i]) > 0:
                        si = False
                        break
                    else:
                        return []

        if q[0] > q[1]:
            si = False

        # избавление от неопределённости inf * 0
        p = p + 1e-301
        if si:
            S.append(list(dotx+p*q[0]) + list(dotx+p*q[1]) + [i])

    return np.array(S)


def ParticularPoints(S, A, b):
    PP = []
    V = S[:, :2]

    binf = ~((abs(V[:, 0]) < float("inf")) & (abs(V[:, 1]) < float("inf")))

    if len(V[binf]) != 0:
        nV = 1
        for k in S[:, 4]:
            k = int(k)
            PP.append((A[k]*b[k])/np.dot(A[k], A[k]))
    else:
        nV = 0
        PP = V

    return np.array(PP), nV, binf


def Intervals2Path(S):
    bs, bp = S[0, :2], S[0, :2]
    P = [bp]

    while len(S) > 0:
        for k in range(len(S)):
            if np.max(np.abs(bs - S[k, :2])) < 1e-8:
                index = k
                break
        es = S[index, 2:4]

        if np.max(np.abs(bs-es)) > 1e-8:
            P.append(es)

            if np.max(np.abs(bp-es)) < 1e-8:
                return np.array(P)
            bs = es
        S = np.delete(S, index, axis=0)
    return np.array(P)


def ChangeVariable(A, b, k):
    y1 = np.zeros(3)
    xi = (A[k] * b[k]) / (A[k] @ A[k])

    l = np.argmin(np.abs(A[k]))
    l1 = ((l + 1) % 3)
    l2 = ((l1 + 1) % 3)

    y1[l1], y1[l2] = A[k, l2], -A[k, l1]
    y2 = np.cross(A[k], y1)

    index = np.array([l for l in range(len(A)) if l != k])
    A, b = A[index], b[index]

    At = np.array([A @ y1, A @ y2]).T
    bt = b - A @ xi

    return xi, At, bt, y1, y2


__center_rm = []
def lineqs(A, b, show=True, title="Solution Set", color='gray', \
           bounds=None, alpha=0.5, s=10, size=(15, 15), save=False):
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

                save: bool
                    Если значение True, то график сохраняется.

    Returns:
                out: list
                    Возвращается список упорядоченных вершин.
                    В случае, если show = True, то график отрисовывается.
    """
    if isinstance(A, Interval) or isinstance(b, Interval):
        raise Exception('Система интервального типа!')

    A = np.asarray(A)
    b = np.asarray(b)

    n, m = A.shape
    assert m <= 2, "В матрице A должно быть два столбца!"
    assert b.shape[0] == n, "Матрица A и правая часть b должны иметь одинаковое число строк!"

    A, b, cnmty = clear_zero_rows(A, b)

    S = BoundaryIntervals(A, b)
    if len(S) == 0:
        return S

    PP, nV, binf = ParticularPoints(S, A, b)

    if (np.asarray([binf]) == True).any():
        if bounds is None:
            PP = np.array(PP)
            PPmin, PPmax = np.min(PP, axis=0), np.max(PP, axis=0)
            center = (PPmin + PPmax)/2
            rm = max((PPmax - PPmin)/2)
            __center_rm.append([max(abs(center) + 5*rm)])
            A = np.append(np.append(A, np.eye(2)), -np.eye(2)).reshape((len(A)+4, 2))
            b = np.append(np.append(b, center-5*rm), -(center+5*rm))

        else:
            A = np.append(np.append(A, np.eye(2)), -np.eye(2)).reshape((len(A)+4, 2))
            b = np.append(np.append(b, [bounds[0][0], bounds[0][1]]), \
                          [-bounds[1][0], -bounds[1][1]])

        S = BoundaryIntervals(A, b)

    vertices = Intervals2Path(S)

    if show:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, title=title)

        x, y = vertices[:, 0], vertices[:, 1]
        ax.fill(x, y, linestyle='-', linewidth=1, color=color, alpha=alpha)
        ax.scatter(x, y, s=s, color='black', alpha=1)

    if save:
        fig.savefig(title + ".png")
    return Unique(vertices)


def IntLinIncR2(A, b, show=True, title="Solution Set", consistency='uni', \
                bounds=None, color='gray', alpha=0.5, s=10, size=(15, 15), save=False):
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

                save: bool
                    Если значение True, то график сохраняется.

    Returns:
                out: list
                    Возвращается список упорядоченных вершин в каждом ортанте
                    начиная с первого и совершая обход в положительном направлении.
                    В случае, если show = True, то график отрисовывается.
    """

    if not (isinstance(A, Interval) or isinstance(b, Interval)):
        return lineqs(A, b, show=show, title=title, color=color, bounds=bounds, \
                      alpha=alpha, s=s, size=size, save=save)

    ortant = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    vertices = []
    n, m = A.shape

    assert m <= 2, "В матрице A должно быть два столбца!"
    assert b.shape[0] == n, "Матрица A и правая часть b должны иметь одинаковое число строк!"

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

            vertices.append(lineqs(-WorkListA, -WorkListb, show=False, bounds=bounds))
    algo(bounds)

    # Если в каком-либо ортанте множество решений неограничено, то создаём
    # новое отрисовочное окно, чтобы срез решений был одинаковым.
    global __center_rm
    if len(__center_rm) > 0:
        vertices = []
        _max = max(np.array(__center_rm))
        bounds = np.array([[-_max, -_max], [_max, _max]])
        algo(bounds)
    __center_rm = []

    if show:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, title=title)

        for k in range(4):
            if len(vertices[k]) > 0:
                x, y = vertices[k][:, 0], vertices[k][:, 1]
                ax.fill(x, y, linestyle='-', linewidth=1, color=color, alpha=alpha)
                ax.scatter(x, y, s=s, color='black', alpha=1)
    if save:
        fig.savefig(title + ".png")

    return vertices


def lineqs3D(A, b, show=True, color='C0', alpha=0.5, s=10, size=(8,8), \
             bounds=None):

    A = np.asarray(A)
    b = np.asarray(b)

    A, b, cnmty = clear_zero_rows(A, b, ndim=3)
    A, b = non_repeat(A, b)

    n, m = A.shape
    assert m == 3, "В матрице A должно быть три столбца!"
    assert b.shape[0] == n, "Матрица A и правая часть b должны иметь одинаковое число строк!"

    V = []
    PP = []
    cfinite = True

    for k in range(n):
        xi, At, bt, y1, y2 = ChangeVariable(A, b, k)
        At, bt, cnmtyt = clear_zero_rows(At, bt)

        if not cnmtyt:
            continue

        if len(bt) == 0:
            PP.append(xi)
            cfinite = False
            continue

        St = BoundaryIntervals(At, bt)
        if len(St) == 0:
            continue

        PPt, nV, binf = ParticularPoints(St, At, bt)

        PPtt = []
        for l in range(len(PPt)):
            PPtt.append((xi + y1*PPt[l, 0]) + y2*PPt[l, 1])

        PP.append(PPtt)

        if (np.asarray([binf]) == True).any():
            cfinite = False
        if not nV:
            V.append(PPtt)

    if len(PP) == 0:
        return []
        # raise Exception('Система несовместна - множество решений пусто')

    if not cfinite:
        if bounds is None:
            x, y, z = [], [], []
            for pp in PP:
                for el in pp:
                    x.append(el[0])
                    y.append(el[1])
                    z.append(el[2])

            x = np.array(x)
            y = np.array(y)
            z = np.array(z)

            xmin, ymin, zmin = np.min(x), np.min(y), np.min(z)
            xmax, ymax, zmax = np.max(x), np.max(y), np.max(z)

            center = np.array([0, 0, 0]) + (x + y + z).mean()
            rm = (xmax-xmin) + (ymax-ymin) + (zmax-zmin)

            if rm <= 1e-14:
                rm = rm + 1

            A = np.append(np.append(A, np.eye(3)), -np.eye(3)).reshape((len(A)+6, 3))
            b = np.append(np.append(b, center-rm), -(center+rm))

        else:
            A = np.append(np.append(A, np.eye(3)), -np.eye(3)).reshape((len(A)+6, 3))
            b = np.append(np.append(b, [bounds[0][0], bounds[0][1], bounds[0][2]]), \
                          [-bounds[1][0], -bounds[1][1], -bounds[1][2]])

    vertices = []
    mn = len(b)
    for i in range(mn):
        xi, At, bt, y1, y2 = ChangeVariable(A, b, i)
        At, bt, cnmtyt = clear_zero_rows(At, bt)

        if not cnmtyt:
            continue

        St = BoundaryIntervals(At, bt)
        if len(St) > 0:
            Pt = Intervals2Path(St)
            P = []
            for l in range(len(Pt)):
                P.append((xi + y1*Pt[l, 0]) + y2*Pt[l, 1])

            P = np.array(P)
            vertices.append(P)

    if show:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection='3d')
    #     ax.grid(False)

        l = 0
        for v in vertices:
            if l >= n:
                color = 'red'

            x, y, z = v[:, 0], v[:, 1], v[:, 2]

            poly3d = [list(zip(x, y, z))]
            PC = Poly3DCollection(poly3d, linewidths=1)
            PC.set_alpha(alpha)
            PC.set_facecolor(color)
            ax.add_collection3d(PC)

            ax.plot(x, y, z, color='black', alpha=1)
            ax.scatter(x, y, z, s=s, color='black')
            l += 1

    return vertices


def IntLinIncR3(A, b, show=True, consistency='uni', color='C0', \
                alpha=0.5, s=10, size=(8, 8), bounds=None):

    if not (isinstance(A, Interval) or isinstance(b, Interval)):
        return lineqs3D(A, b, show=False, color=color, \
                        alpha=alpha, s=s, size=size, bounds=bounds)

    ortant = [(1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1), \
              (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1)]
    vertices = []
    n, m = A.shape

    assert m <= 3, "В матрице A должно быть два столбца!"
    assert b.shape[0] == n, "Матрица A и правая часть b должны иметь одинаковое число строк!"

    for ort in range(8):
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

        for k in range(2*n):
            if WorkListb[k] >= 0:
                WorkListb[k] -= 1e-15
            else:
                WorkListb[k] += 1e-15

        vertices.append(lineqs3D(-WorkListA, -WorkListb, show=False, bounds=bounds))

    if show:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection='3d')
    #     ax.grid(False)

        color1 = color
        for el in vertices:

            color = color1
            l = 0
            for v in el:
                x, y, z = v[:, 0], v[:, 1], v[:, 2]
                xmin, ymin, zmin = np.min(abs(x)), np.min(abs(y)), np.min(abs(z))

                if n <= l and l < n + 3 and (xmin * ymin * zmin == 0):
                    l += 1
                    continue

                elif l >= n:
                    color = 'red'

                poly3d = [list(zip(x, y, z))]
                PC = Poly3DCollection(poly3d, linewidths=1)
                PC.set_alpha(alpha)
                PC.set_facecolor(color)
                ax.add_collection3d(PC)

                ax.plot(x, y, z, color='black', alpha=1)
                ax.scatter(x, y, z, s=s, color='black')
                l += 1

    return vertices
