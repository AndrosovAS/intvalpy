import numpy as np
from bisect import bisect_left

from ..kernel.real_intervals import Interval
from ..kernel.new_objects import zeros, eye
from ..kernel.preprocessing import asinterval
from .HBR import HBR
from .Gauss import Gauss



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
                    # TODO, что такое rt? 
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
    A = asinterval(np.vectorize(lambda el: Interval(float(el.a), float(el.b), sortQ=False))(A._data))
    b = asinterval(np.vectorize(lambda el: Interval(float(el.a), float(el.b), sortQ=False))(b._data))
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
