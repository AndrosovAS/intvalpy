import numpy as np

from intvalpy.RealInterval import Interval
from intvalpy.utils import asinterval, zeros, dist, intersection, diag, compmat, eye, isnan, full

from bisect import bisect_left


def Gauss(A, b):
    """
    Procedure Gauss.

    Parameters:

        A: Interval
            The input interval matrix of ISLAE, which can be either only square.

        b: Interval
            The interval vector of the right part of the ISLAE.

    Returns:

        out: Interval
            Returns an interval vector, which means an external estimate of the united solution set.

    """

    A, b = A.copy(), b.copy()
    n, _ = A.shape

    r = zeros((n, n))
    x = zeros(n)

    for j in range(n-1):
        mig = A[j:, j].mig
        argmax = np.argmax(mig)
        max_mig = mig[argmax]

        assert max_mig > 0, 'All subdiagonal elements of the column contain zeros.'

        k = argmax + j
        # swap rows, if necessary
        if k != j:
            A[[j, k], j:] = A[[k, j], j:]
            b[[j, k]] = b[[k, j]]

        # transformations of the forward step of the Gauss method
        r[j+1:, j] = A[j+1:, j]/A[j, j]
        A[j+1:, j+1:] = A[j+1:, j+1:] - r[j+1:, j] * A[j, j+1:]
        b[j+1:] = b[j+1:] - r[j+1:, j] * b[j]

    # backward step of the Gauss method
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - sum(A[i, i:] * x[i:])) / A[i, i]
    return x


def Gauss_Seidel(A, b, x0=None, C=None, tol=1e-12, maxiter=2000):
    """
    The iterative Gauss-Seidel method for obtaining external evaluations of the united solution set
    for an interval system of linear algebraic equations (ISLAE).

    Parameters:

        A: Interval
            The input interval matrix of ISLAE, which can be either only square.

        b: Interval
            The interval vector of the right part of the ISLAE.

        X: Interval, optional
            An initial guess within which to search for external evaluation is suggested.
            By default, X is an interval vector consisting of the elements [-1000, 1000].

        C: np.array, Interval
            A matrix for preconditioning the system. By default, C = inv(mid(A)).

        tol: float, optional
            The error that determines when further crushing of the bars is unnecessary,
            i.e. their width is "close enough" to zero, which can be considered exactly zero.

        maxiter: int, optional
            The maximum number of iterations.

    Returns:

        out: Interval
            Returns an interval vector, which means an external estimate of the united solution set.
    """

    n, m = A.shape
    assert n == m, 'Matrix is not square'
    assert n == len(b), 'Inconsistent dimensions of matrix and right-hand side vector'

    A, b = A.copy(), b.copy()
    C = np.linalg.inv(A.to_float().mid) if C is None else C
    A = C @ A
    b = C @ b

    # проверим, что A является H-матрицей
    B = np.linalg.inv(np.array(compmat(A), dtype=np.float64))
    v = abs(B @ np.ones(n))
    u = A @ v
    assert (u > 0).any(), 'Matrix of the system not an H-matrix'


    distance = np.inf
    result = zeros(n)
    pre_result = full(n, -1000, 1000) if x0 is None else x0

    nit = 0
    while distance >= tol and nit <= maxiter:
        for k in range(n):
            new_bar = (b[k] - sum(A[k, :k] * result[:k]) - sum(A[k, k+1:] * pre_result[k+1:])) / A[k, k]
            result[k] = intersection(pre_result[k], new_bar)

            if isnan(result[k]):
                raise Exception("The united solution set does not intersect the bar X.")

        distance = dist(result, pre_result)
        pre_result = result.copy()
        nit += 1
    return result


def HBR(A, b):
    """
    Procedure Hansen-Bliek-Rohn.

    Parameters:

        A: Interval
            The input interval matrix of ISLAE, which can be either only square.

        b: Interval
            The interval vector of the right part of the ISLAE.

    Returns:

        out: Interval
            Returns an interval vector, which means an external estimate of the united solution set.

    """

    n, m = A.shape
    assert n == m, 'Matrix is not square'
    assert n == len(b), 'Inconsistent dimensions of matrix and right-hand side vector'

    # создадим глубокие копии и сделаем предобуславливание
    A, b = A.copy(), b.copy()
    C = np.linalg.inv(A.to_float().mid)
    A = C @ A
    b = C @ b

    # проверим, что A является H-матрицей
    dA = diag(A)
    A = compmat(A)
    B = np.linalg.inv(np.array(A, dtype=np.float64))
    v = abs(B @ np.ones(n))
    u = A @ v
    assert (u > 0).any(), 'Matrix of the system not an H-matrix'

    # проводим процедуру Хансена-Блика-Рона
    dAc = np.diag(A)
    A = A @ B - np.eye(n)
    w = np.max(-A / np.outer(u, np.ones(n)), axis=0)
    dlow = -(v*w - np.diag(B))
    B = B + v @ w
    u = B @ b.mag
    d = np.diag(B)
    alpha = dAc + (-1)/d
    if len(b.shape) == 1:
        beta = u / dlow - b.mag
        return (b + Interval(-beta, beta)) / (dA + Interval(-alpha, alpha))
    else:
        v = np.ones(n)
        beta = u / (d @ v) - b.mag
        return (b + Interval(-beta, beta)) / ( (dA + Interval(-alpha, alpha)) @ v )


def Subdiff(A, b, tol=1e-12, maxiter=500, tau=1, norm_min_val=1e-12):
    """
    Subdifferential Newton method.

    Parameters:

        A: Interval
            The input interval matrix of ISLAE, which can be either square or rectangular.

        b: Interval
            The interval vector of the right part of the ISLAE.

        tol: float, optional
            An error that determines when further iterations of the algorithm are not required,
            i.e. their distance between the solution at iteration k and the solution at iteration k+1
            is "close enough" to zero.


        maxiter: int, optional
            The maximum number of iterations.

        ...

    Returns:

        out: Interval
            Returns an interval vector, which, after substituting into the system of equations
            and performing all operations according to the rules of arithmetic and analysis,
            turns the equations into true equalities.
    """


    def superMatrix(A):
        Amid = A.mid
        index = Amid >= 0
        A_plus = np.zeros(A.shape)
        A_minus = np.zeros(A.shape)
        A_plus[index] = Amid[index]
        A_minus[~index] = Amid[~index]

        result = np.zeros((2*n, 2*m))
        result[:n, :m], result[:n, m:2*m] = A_plus, A_minus
        result[n:2*n, :m], result[n:2*n, m:2*m] = A_minus, A_plus
        return result


    def calcSubgrad(F, i, j, a, b):
        n = int(F.shape[0] / 2)

        if np.sign(a.a) * np.sign(a.b) > 0:
            k = 0 if np.sign(a.a) > 0 else 2
        else:
            k = 1 if a.a < a.b else 3

        if np.sign(b.a) * np.sign(b.b) > 0:
            m = 1 if np.sign(b.a) > 0 else 3
        else:
            m = 2 if b.a <= b.b else 4

        cause = 4*k + m
        if cause == 1:
            F[i, j] = a.a
            F[i + n, j + n] = a.b
        elif cause == 2:
            F[i, j] = a.b
            F[i + n, j + n] = a.b
        elif cause == 3:
            F[i, j] = a.b
            F[i + n, j + n] = a.a
        elif cause == 4:
            F[i, j] = a.a
            F[i + n, j + n] = a.a
        elif cause == 5:
            F[i, j + n] = a.a
            F[i + n, j + n] = a.b
        elif cause == 6:
            if a.a*b.b < a.b*b.a:
                F[i, j + n] = a.a
            else:
                F[i, j] = a.b

            if a.a*b.a > a.b*b.b:
                F[i + n, j] = a.a
            else:
                F[i + n, j + n] = a.b
        elif cause == 7:
            F[i, j] = a.b
            F[i + n, j] = a.a
        elif cause == 9:
            F[i, j + n] = a.a
            F[i + n, j] = a.b
        elif cause == 10:
            F[i, j + n] = a.a
            F[i + n, j] = a.a
        elif cause == 11:
            F[i, j + n] = a.b
            F[i + n, j] = a.a
        elif cause == 12:
            F[i, j + n] = a.b
            F[i + n, j] = a.b
        elif cause == 13:
            F[i, j] = a.a
            F[i + n, j] = a.b
        elif cause == 15:
            F[i, j + n] = a.b
            F[i + n, j + n] = a.a
        elif cause == 16:
            if a.a*b.a > a.b*b.b:
                F[i, j] = a.a
            else:
                F[i, j + n] = -a.b

            if a.a*b.b < a.b*b.a:
                F[i + n, j + n] = a.a
            else:
                F[i + n, j] = a.b

        return F

    n, m = A.shape

    assert n == m, "matrix is not square"
    assert m == b.shape[0], "mismatch of matrix and vector dimensions"

    F = superMatrix(A)
    xx = np.zeros(2*n)
    xx[:n], xx[n:2*n] = b.a, b.b

    xx = np.linalg.solve(F, xx)
    r = float('inf')
    q = 1
    nit = 0
    while nit <= maxiter and r / q > tol:
        r = 0
        x = np.copy(xx)
        F = np.zeros((2*n, 2*n))

        for i in range(n):
            s = Interval(0, 0)

            for j in range(n):
                g = A[i, j]
                h = Interval(x[j], x[j+n], sortQ=False)
                t = g * h
                s = s + t
                F = calcSubgrad(F, i, j, g, h)

            t = s + b[i].opp
            xx[i] = t.a
            xx[i + n] = t.b

            r = r + t.mag

        xx = np.linalg.solve(F, xx)
        xx = x - xx * tau

        q = np.linalg.norm(xx, 1)
        if q <= norm_min_val:
            q = 1

        nit += 1
    return Interval(xx[:n], xx[n:], sortQ=False)


def PPS(A, b, tol=1e-12, maxiter=2000, nu=None):
    """
    PPS - optimal (exact) componentwise estimation of the united solution
    set to interval linear system of equations.

    x = PPS(A, b) computes optimal componentwise lower and upper estimates
    of the solution set to interval linear system of equations Ax = b,
    where A - square interval matrix, b - interval right-hand side vector.


    x = PPS(A, b, tol, maxiter, nu) computes vector x of
    optimal componentwise estimates of the solution set to interval linear
    system Ax = b with accuracy no more than epsilon and after the number of
    iterations no more than numit. Optional input argument ncomp indicates
    a component's number of interval solution in case of computing the estimates
    for this component only. If this argument is omitted, all componentwise
    estimates is computed.


    Parameters:

        A: Interval
            The input interval matrix of ISLAE, which can be either square or rectangular.

        b: Interval
            The interval vector of the right part of the ISLAE.

        tol: float, optional
            The error that determines when further crushing of the bars is unnecessary,
            i.e. their width is "close enough" to zero, which can be considered exactly zero.

        maxiter: int, optional
            The maximum number of iterations.

        nu: int, optional
            Choosing the number of the component along which the set of solutions is evaluated.

    Returns:

        out: Interval
            Returns an optimal interval vector, which means an external estimate of the united solution set.
    """

    class KeyWrapper:
        def __init__(self, iterable, key):
            self.it = iterable
            self.key = key

        def __getitem__(self, i):
            return self.key(self.it[i])

        def __len__(self):
            return len(self.it)

    class PPSLstEl:
        def __init__(self, gamma=None, Q=None, r=None, Y=None, x=None, W=None, s=None, t=None):
            self.gamma = gamma       #lower endpoint of nu-th component of interval estimate
            self.Q = Q               #interval matrix of the system
            self.r = r               #interval right-hand side vector
            self.Y = Y               #inverse interval matrix
            self.x = x               #outer interval estimates of solution set
            self.W = W               #matrix W
            self.s = s               #column vector s
            self.t = t               #row vector t


    def algo(nu):

        def split(ind, fl, L, Wch, sch):
            invQ = L[0].Y
            vx = HBR(Q, r)
            # vx = Gauss(Q, r)
            gamma = vx[nu].a
            Mn = np.linalg.solve(Q.mid, r.mid)

            if gamma < omega and ind:
                tch = False
                Lambda = []
                while Wch or sch or tch:
                    Wc = False
                    sc = False
                    tc = False

                    ##################################################
                    if sch:
                        index = np.where(t != 0)[0]
                        for k in K:
                            for j in index:
                                a = s[k]*t[j]
                                if W[k, j] != a:
                                    Wc = True
                                    Omega.append([k, j])
                                W[k, j] = a

                    if tch:
                        index = np.where(s != 0)[0]
                        for l in Lambda:
                            for j in index:
                                a = s[j] * t[l]
                                if W[j, l] != a:
                                    Wc = True
                                    Omega.append([j, l])
                                W[j, l] = a

                    ##################################################
                    if Wch:
                        for i1, i2 in Omega:
                            if t[i2]:
                                a = W[i1, i2] / t[i2]
                                if s[i1] != a:
                                    sc = True
                                    K.append(i1)
                                s[i1] = a

                    if tch:
                        for l in Lambda:
                            for j in range(n):
                                if (not s[j]) and W[j, l]:
                                    a = W[j, l] / t[l]
                                    if s[j] != a:
                                        sc = True
                                        K.append(j)
                                    s[j] = a

                    ##################################################
                    if Wch:
                        for i1, i2 in Omega:
                            if s[i1]:
                                a = W[i1, i2] / s[i1]
                                if t[i2] != a:
                                    tc = True
                                    Lambda.append(i2)
                                t[i2] = a

                    if sch:
                        for k in K:
                            for j in range(n):
                                if (not t[j]) and W[k, j]:
                                    a = W[k, j] / s[k]
                                    if t[j] != a:
                                        tc = True
                                        Lambda.append(j)
                                    t[j] = a

                    ##################################################
                    Wch = Wc
                    sch = sc
                    tch = tc


            if gamma < omega:
                newcol = PPSLstEl(Q=Q.copy(), r=r.copy(), gamma=gamma, x=vx, W=W, s=s, t=t)
                # newcol.Y = HBR(Q, I) if fl else invQ.copy()
                if fl:
                    invQ_fl = zeros(Q.shape)
                    for k in range(m):
                        e = zeros(n)
                        e[k] = 1
                        invQ_fl[:, k] = Gauss(Q, e)
                    newcol.Y = invQ_fl.copy()
                else:
                    newcol.Y = invQ.copy()

                if gamma < cstL:
                    bslindex = bisect_left(np.array([l.gamma for l in L]), gamma)
                    bslindex = bslindex if bslindex > 0 else 1
                    L.insert(bslindex, newcol)
                else:
                    Lk.append(newcol)

            return Mn


        vx = HBR(A, b)
        # vx = Gauss(A, b)
        omega = float('inf')
        L = []
        L.append(PPSLstEl(vx[nu].a, A.copy(), b.copy(), invA.copy(), vx.copy(),
                          np.zeros((n, n), dtype=int), np.zeros(n, dtype=int), np.zeros(n)))

        Lk = []
        cstL = float('inf')
        nit = 0
        exactQ = True if (L[0].Q.rad == 0).all() else False
        while (not exactQ) and abs(omega - L[0].gamma) > tol and nit <= maxiter:

            #############################################################################################
            # monotony test
            dQ = asinterval( -np.outer(L[0].Y[nu], L[0].x) )

            index_upper_zero, index_lower_zero = dQ >= 0, dQ <= 0
            if index_upper_zero.any():
                L[0].Q[index_upper_zero], L[0].W[index_upper_zero] = L[0].Q[index_upper_zero].a, 1
            if index_lower_zero.any():
                L[0].Q[index_lower_zero], L[0].W[index_lower_zero] = L[0].Q[index_lower_zero].b, -1

            index_upper_zero, index_lower_zero = L[0].Y[nu] >= 0, L[0].Y[nu] <= 0
            if index_upper_zero.any():
                L[0].r[index_upper_zero], L[0].s[index_upper_zero] = L[0].r[index_upper_zero].a, -1
            if index_lower_zero.any():
                L[0].r[index_lower_zero], L[0].s[index_lower_zero] = L[0].r[index_lower_zero].b, 1


            #############################################################################################
            #find the element providing the maximal product of the width by the derivative estimate
            mat = dQ.mag * L[0].Q.wid
            im1, jm1 = np.unravel_index(np.argmax(mat), mat.shape)
            z1 = mat[im1, jm1]

            vector = L[0].Y[nu].mag * L[0].r.wid
            im2 = np.argmax(vector)
            z2 = vector[im2]


            #############################################################################################
            #refining W by searching submatrix
            breakQ = False
            for k in range(n):
                for l in range(n):
                    if k == im1 and l == jm1:
                        continue
                    elif L[0].W[k, l] != 0 and L[0].W[k, jm1] != 0 and L[0].W[im1, l] != 0:
                        L[0].W[im1, jm1] = L[0].W[k, l] * L[0].W[k, jm1] * L[0].W[im1, l]
                        breakQ = True
                        break
                if breakQ: break


            #############################################################################################
            #generating the systems-descendants
            W = np.copy(L[0].W)
            s = np.copy(L[0].s)
            t = np.copy(L[0].t)
#             Q = L[0].Q.copy()
#             r = L[0].r.copy()

            Omega = []
            K = []
            Wch = False
            sch = False

            if z1 > z2:
                Q = L[0].Q.copy()
                r = L[0].r
                if W[im1, jm1] == 0:
                    Q[im1, jm1] = Q[im1, jm1].a
                    W[im1, jm1] = 1
                    Omega.append([im1, jm1])
                    Wch = True
                    m1 = split(1, 1, L, Wch, sch)
                    W = L[0].W
                    s = L[0].s
                    t = L[0].t
                    Omega = []
                    K = []

                    Q[im1, jm1] = L[0].Q[im1, jm1].b
                    W[im1, jm1] = -1
                    Omega.append([im1, jm1])
                    Wch = True
                    m2 = split(1, 1, L, Wch, sch)
                    mu = min(m1[nu], m2[nu])
                elif W[im1, jm1] == 1:
                    Q[im1, jm1] = Q[im1, jm1].a
                    m1 = split(0, 1, L, Wch, sch)
                    mu = m1[nu]

                else:
                    Q[im1, jm1] = Q[im1, jm1].b
                    m1 = split(0, 1, L, Wch, sch)
                    mu = m1[nu]

            else:
                Q = L[0].Q
                r = L[0].r.copy()
                if s[im2] == 0:
                    r[im2] = r[im2].a
                    s[im2] = -1
                    K.append(im2)
                    sch = True
                    m1 = split(1, 0, L, Wch, sch)
                    s = L[0].s
                    W = L[0].W
                    t = L[0].t
                    Omega = []
                    K = []

                    r[im2] = L[0].r[im2].b
                    s[im2] = 1
                    K.append(im2)
                    sch = True
                    m2 = split(1, 0, L, Wch, sch)
                    mu = min(m1[nu], m2[nu])

                elif s[im2] == 1:
                    r[im2] = r[im2].b
                    m1 = split(0, 0, L, Wch, sch)
                    mu = m1[nu]

                else:
                    r[im2] = r[im2].a
                    m1 = split(0, 0, L, Wch, sch)
                    mu = m1[nu]


            #############################################################################################
            L = L[1:]
            if omega > mu:
                omega = mu

            tL = len(L)
            tLk = len(Lk)
            if not L and Lk:
                Lk = [lk for lk in Lk if lk.gamma <= omega]
                tLk = len(Lk)

                if Lk:
                    _gamma = [lk.gamma for lk in Lk]
                    index_sort = np.argsort(_gamma)
                    H = _gamma[index_sort]

                    cstL = (rt*omega + Lk[index_sort[0]].gamma) / (1 + rt)
                    T = 0
                    while H[T] < cstL:
                        L.append(Lk[index_sort[T]])
                        T += 1
                        if T >= tLk: break
                    Lk = [Lk[k] for k in range(len(Lk)) if not k in index_sort[:T]]


            tL = len(L)
            if L and not Lk:
                L = [l for l in L if l.gamma <= omega]

            if not L and not Lk:
                ocenka = omega
                break
            ocenka = L[0].gamma
            exactQ = True if (L[0].Q.rad == 0).all() else False

            nit += 1
        return ocenka


    n, m = A.shape

    assert n == m, 'Matrix is not square'
    assert n == len(b), 'Inconsistent dimensions of matrix and right-hand side vector'

    assert np.linalg.det(A.mid) != 0, 'Matrix is singular'
    # assert max(abs(np.linalg.svd(abs(np.linalg.inv(A.mid)) @ A.rad, compute_uv=False))) < 1, 'Matrix is singular'

    A, b = A.copy(), b.copy()
    I = eye(n)
    # invA = HBR(A, I)
    invA = zeros(A.shape)
    for k in range(m):
        e = zeros(n)
        e[k] = 1
        invA[:, k] = Gauss(A, e)


    inf = []
    sup = []
    if nu is None:
        for endint in [1, -1]:
            b = endint * b.copy()
            for _nu in range(n):
                if endint == -1:
                    sup.append(endint * algo(_nu))
                else:
                    inf.append(algo(_nu))
    else:
        _nu = nu
        for endint in [1, -1]:
            b = endint * b.copy()
            if endint == -1:
                sup.append(endint * algo(_nu))
            else:
                inf.append(algo(_nu))

    return Interval(inf, sup, sortQ=False)
