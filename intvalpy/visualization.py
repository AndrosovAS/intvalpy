import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .RealInterval import INTERVAL_CLASSES

from .utils import infinity


def scatter_plot(x, y, title="Box", color='gray', alpha=0.5, s=10, size=(15, 15), save=False):
    """
    A mathematical diagram depicting the values of two variables in the form of bars on a Cartesian plane.

    Parameters:

        x: Interval
            Interval vector of the data position on the OX axis.

        y: Interval
            Interval vector of the data position on the OY axis.

        title: str, optional
            The top legend of the graph.

        color: str, optional
            The color of the display of the bars.

        alpha: float, optional
            Transparency of the graph.

        s: float, optional
            How big are the points of the vertices.

        size: tuple, optional
            The size of the drawing window.

        save: bool, optional
            If the value is True, the graph is saved.

    Returns:

        out: None
            A scatter plot is displayed.
    """

    ox = np.array([x.a, x.a, x.b, x.b])
    oy = np.array([y.a, y.b, y.b, y.a])

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, title=title)

    plt.fill(ox, oy, color=color, alpha=alpha)
    index = ((x.a - x.b) == 0) & ((y.a - y.b) == 0)
    ax.scatter(ox[:, index], oy[:, index], s=s, color=color, alpha=alpha)

    if save:
        fig.savefig(title + ".png")


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
        q = [-infinity, infinity]
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
def lineqs(A, b, show=True, title="Solution Set", color='gray',
           bounds=None, alpha=0.5, s=10, size=(15, 15), save=False):
    """
    The function visualizes the set of solutions of a system of linear algebraic
    inequalities A x >= b with two variables by the method of boundary intervals, and
    also outputs the vertices of the set of solutions.

    If the set of solutions is unlimited, then the algorithm independently
    selects the rendering boundaries.

    Parameters:

            A: float, array_like
                Matrix of a system of linear algebraic inequalities.

            b: float, array_like
                The vector of the right part of the system of linear algebraic inequalities.

            show: bool, optional
                This parameter is responsible for whether a set of solutions will be shown.
                By default, the value is set to True, i.e. the graph is being drawn.

            title: str, optional
                The top legend of the graph.

            color: str, optional
                The color of the inner area of the set of solutions.

            bounds: array_like, optional
                Borders of the drawing area. The first element of the array is responsible for the lower faces
                on the OX and OY axes, and the second for the upper ones. Thus, in order to OX
                lay within [-2, 2], and OY within [-3, 4], it is necessary to set bounds as [[-2, -3], [2, 4]].

            alpha: float, optional
                Transparency of the graph.

            s: float, optional
                How big are the points of the vertices.

            size: tuple, optional
                The size of the drawing window.

            save: bool, optional
                If the value is True, the graph is saved.

    Returns:

            out: list
                Returns a list of ordered vertices.
                If show = True, then the graph is drawn.
    """

    if isinstance(A, INTERVAL_CLASSES) or isinstance(b, INTERVAL_CLASSES):
        raise Exception('Interval type system!')

    A = np.asarray(A)
    b = np.asarray(b)

    n, m = A.shape
    assert m <= 2, "There should be two columns in matrix A."
    assert b.shape[0] == n, "The size of the matrix A must be consistent with the size of the vector of the right part of b."

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
            b = np.append(np.append(b, [bounds[0][0], bounds[0][1]]),
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


def OneShotVisual2D(*args, title="Solution Set", grid=True, size=(15, 15), labelsize=None, save=False):

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, title=title)

    if grid:
        ax.grid()

    if labelsize is None:
        ax.xaxis.set_tick_params(labelsize=size[0])
        ax.yaxis.set_tick_params(labelsize=size[0])
    else:
        ax.xaxis.set_tick_params(labelsize=labelsize)
        ax.yaxis.set_tick_params(labelsize=labelsize)

    for v in args:
        alpha = 0.5
        s = 10
        color = 'gray'
        vertices = None
        for key in v.keys():
            if key == "vertices":
                vertices = v["vertices"]
            elif key == "alpha":
                alpha = v["alpha"]
            elif key == "s":
                s = v["s"]
            elif key == "color":
                color = v["color"]

        if not (vertices is None):
            if isinstance(vertices, list):
                for k in range(4):
                    if len(vertices[k]) > 0:
                        x, y = vertices[k][:, 0], vertices[k][:, 1]
                        ax.fill(x, y, linestyle='-', linewidth=1, color=color, alpha=alpha)
                        ax.scatter(x, y, s=s, color='black', alpha=1)
            else:
                if len(vertices) > 0:
                    x, y = vertices[:, 0], vertices[:, 1]
                    ax.fill(x, y, linestyle='-', linewidth=1, color=color, alpha=alpha)
                    ax.scatter(x, y, s=s, color='black', alpha=1)

    if save:
        fig.savefig(title + ".png")


def IntLinIncR2(A, b, show=True, title="Solution Set", consistency='uni',
                bounds=None, color='gray', alpha=0.5, s=10, size=(15, 15), save=False):
    """
    The function visualizes a set of solutions of an interval system of linear
    algebraic equations A x = b with two variables by the boundary value method
    intervals, and also outputs the vertices of the set of solutions.

    If the set of solutions is unlimited, then the algorithm independently
    selects the rendering boundaries.

    Parameters:

            A: Interval
                The input interval matrix of ISLAE, which can be either square or rectangular.

            b: Interval
                The interval vector of the right part of the ISLAE.

            show: bool, optional
                This parameter is responsible for whether a set of solutions will be shown.
                By default, the value is set to True, i.e. the graph is being drawn.

            title: str, optional
                The top legend of the graph.

            consistency: str, optional
                A parameter for selecting the type of a set of solutions.
                If consistency = "uni", then the function returns the united set of the solution.
                If consistency = "tol", then the tolerance set of the solution.

            bounds: array_like, optional
                Borders of the drawing area. The first element of the array is responsible for the lower faces
                on the OX and OY axes, and the second for the upper ones. Thus, in order to OX
                lay within [-2, 2], and OY within [-3, 4], it is necessary to set bounds as [[-2, -3], [2, 4]].

            color: str, optional
                The color of the inner area of the set of solutions.

            alpha: float, optional
                Transparency of the graph.

            s: float, optional
                How big are the points of the vertices.

            size: tuple, optional
                The size of the drawing window.

            save: bool, optional
                If the value is True, the graph is saved.

    Returns:

            out: list
                Returns a list of ordered vertices in each ortant starting from the first
                and making a detour in a positive direction. If show = True, then the graph is drawn.
    """

    if not (isinstance(A, INTERVAL_CLASSES) or isinstance(b, INTERVAL_CLASSES)):
        return lineqs(A, b, show=show, title=title, color=color, bounds=bounds,
                      alpha=alpha, s=s, size=size, save=save)

    ortant = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    vertices = []
    n, m = A.shape

    assert m <= 2, "There should be two columns in matrix A."
    assert b.shape[0] == n, "The size of the matrix A must be consistent with the size of the vector of the right part of b."

    def algo(bounds):
        for ort in range(4):
            tmp = A.copy
            WorkListA = np.zeros((2*n+m, m))
            WorkListb = np.zeros(2*n+m)

            for k in range(m):
                if ortant[ort][k] == -1:
                    tmp[:, k] = tmp[:, k].dual
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


def lineqs3D(A, b, show=True, color='C0', alpha=0.5, s=10, size=(8, 8),
             bounds=None):
    """
    The function visualizes the set of solutions of a system of linear algebraic
    inequalities A x >= b with three variables by the method of boundary intervals, and
    also outputs the vertices of the set of solutions.

    If the set of solutions is unlimited, then the algorithm independently
    selects the rendering boundaries.


    Parameters:

        A: float, array_like
            Matrix of a system of linear algebraic inequalities.

        b: float, array_like
            The vector of the right part of the system of linear algebraic inequalities.

        show: bool, optional
            This parameter is responsible for whether a set of solutions will be shown.
            By default, the value is set to True, i.e. the graph is being drawn.

        color: str, optional
            The color of the inner area of the set of solutions.

        bounds: array_like, optional
            Borders of the drawing area. The first element of the array is responsible
            for the lower faces along the axes OX, OY and OZ, and the second for the upper ones.
            Thus, in order for OX to lie within [-2, 2], OY within [-3, 4], and OZ within [1, 5],
            it is necessary to set bounds as [[-2, -3, 1], [2, 4, 5]].

        alpha: float, optional
            Transparency of the graph.

        s: float, optional
            How big are the points of the vertices.

        size: tuple, optional
            The size of the drawing window.

    Returns:

        out: list
            Returns a list of ordered vertices.
            If show = True, then the graph is drawn.
    """

    A = np.asarray(A)
    b = np.asarray(b)

    A, b, cnmty = clear_zero_rows(A, b, ndim=3)
    A, b = non_repeat(A, b)

    n, m = A.shape
    assert m == 3, "There should be three columns in matrix A."
    assert b.shape[0] == n, "The size of the matrix A must be consistent with the size of the vector of the right part of b."

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
            __center_rm.append([max(abs(center) + 1*rm)])
            A = np.append(np.append(A, np.eye(3)), -np.eye(3)).reshape((len(A)+6, 3))
            b = np.append(np.append(b, center-rm), -(center+rm))
        else:
            A = np.append(np.append(A, np.eye(3)), -np.eye(3)).reshape((len(A)+6, 3))
            b = np.append(np.append(b, [bounds[0][0], bounds[0][1], bounds[0][2]]),
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

        if not (bounds is None):
            ax.set_xlim((bounds[0][0], bounds[1][0]))
            ax.set_ylim((bounds[0][1], bounds[1][1]))
            ax.set_zlim((bounds[0][2], bounds[1][2]))

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


def IntLinIncR3(A, b, show=True, consistency='uni', color='C0',
                alpha=0.5, s=10, size=(8, 8), bounds=None, zero_lvl=True):
    """
    The function visualizes a set of solutions of an interval system of linear
    algebraic equations A x = b with three variables by the boundary value method
    intervals, and also outputs the vertices of the set of solutions.

    If the set of solutions is unlimited, then the algorithm independently
    selects the rendering boundaries.


    Parameters:

        A: Interval
            The input interval matrix of ISLAE, which can be either square or rectangular.

        b: Interval
            The interval vector of the right part of the ISLAE.

        show: bool, optional
            This parameter is responsible for whether a set of solutions will be shown.
            By default, the value is set to True, i.e. the graph is being drawn.

        consistency: str, optional
            A parameter for selecting the type of a set of solutions.
            If consistency = "uni", then the function returns the united set of the solution.
            If consistency = "tol", then the tolerance set of the solution.

        bounds: array_like, optional
            Borders of the drawing area. The first element of the array is responsible
            for the lower faces along the axes OX, OY and OZ, and the second for the upper ones.
            Thus, in order for OX to lie within [-2, 2], OY within [-3, 4], and OZ within [1, 5],
            it is necessary to set bounds as [[-2, -3, 1], [2, 4, 5]].

        color: str, optional
            The color of the inner area of the set of solutions.

        alpha: float, optional
            Transparency of the graph.

        s: float, optional
            How big are the points of the vertices.

        size: tuple, optional
            The size of the drawing window.

    Returns:

        out: list
            Returns a list of ordered vertices in each ortant starting from the first
            and making a detour in a positive direction. If show = True, then the graph is drawn.
    """

    if not (isinstance(A, INTERVAL_CLASSES) or isinstance(b, INTERVAL_CLASSES)):
        return lineqs3D(A, b, show=False, color=color,
                        alpha=alpha, s=s, size=size, bounds=bounds)

    ortant = [(1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1),
              (1, 1, -1), (-1, 1, -1), (-1, -1, -1), (1, -1, -1)]
    vertices = []
    n, m = A.shape

    assert m <= 3, "There should be three columns in matrix A."
    assert b.shape[0] == n, "The size of the matrix A must be consistent with the size of the vector of the right part of b."

    def algo(bounds):
        gxmin, gymin, gzmin = infinity, infinity, infinity
        gxmax, gymax, gzmax = -infinity, -infinity, -infinity
        for ort in range(8):
            tmp = A.copy
            WorkListA = np.zeros((2*n+m, m))
            WorkListb = np.zeros(2*n+m)

            for k in range(m):
                if ortant[ort][k] == -1:
                    tmp[:, k] = tmp[:, k].dual
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
            for v in vertices[-1]:
                gxmin = min(gxmin, np.min(v[:, 0]))
                gymin = min(gymin, np.min(v[:, 1]))
                gzmin = min(gzmin, np.min(v[:, 2]))

                gxmax = max(gxmax, np.max(v[:, 0]))
                gymax = max(gymax, np.max(v[:, 1]))
                gzmax = max(gzmax, np.max(v[:, 2]))
        return gxmin, gymin, gzmin, gxmax, gymax, gzmax

    gxmin, gymin, gzmin, gxmax, gymax, gzmax = algo(bounds)

    # Если в каком-либо ортанте множество решений неограничено, то создаём
    # новое отрисовочное окно, чтобы срез решений был одинаковым.
    global __center_rm
    if len(__center_rm) > 0:
        vertices = []
        _max = max(np.array(__center_rm))
        bounds = np.array([[-_max, -_max, -_max], [_max, _max, _max]])
        gxmin, gymin, gzmin, gxmax, gymax, gzmax = algo(bounds)
    __center_rm = []

    if show:
        fig = plt.figure(figsize=size)
        ax = fig.add_subplot(111, projection='3d')

        if not bounds is None:
            ax.set_xlim((bounds[0][0], bounds[1][0]))
            ax.set_ylim((bounds[0][1], bounds[1][1]))
            ax.set_zlim((bounds[0][2], bounds[1][2]))

        color1 = color
        for el in vertices:

            color = color1
            l = 0
            for v in el:
                x, y, z = v[:, 0], v[:, 1], v[:, 2]
                xmin, ymin, zmin = np.min(abs(x)), np.min(abs(y)), np.min(abs(z))
                xmax, ymax, zmax = np.max(abs(x)), np.max(abs(y)), np.max(abs(z))

                if n <= l and l < n + 3 and (xmin * ymin * zmin == 0):
                    l += 1
                    continue

                elif zero_lvl and ((xmax == xmin and xmax == 0 and abs(gxmin*gxmax) > 1e-14) or
                                   (ymax == ymin and ymax == 0 and abs(gymin*gymax) > 1e-14) or
                                   (zmax == zmin and zmax == 0 and abs(gzmin*gzmax) > 1e-14)):      # demo
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
