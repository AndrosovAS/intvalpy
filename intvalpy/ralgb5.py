import numpy as np

def ralgb5(calcfg, x0, tol=1e-12, maxiter=2000, alpha=2.3, nsims=30, h0=1, nh=3, q1=0.9, q2=1.1, tolx=1e-12, tolg=1e-12, tolf=1e-12):

    m = len(x0)
    hs = h0
    B = np.eye(m)
    vf = np.zeros(nsims) + float('inf')
    w = 1./alpha - 1

    x = np.copy(x0)
    xr = np.copy(x0)

    nit = 0
    ncalls = 1
    fr, g0 = calcfg(xr)

    if np.linalg.norm(g0) < tolg:
        ccode = 2
        return xr, fr, nit, ncalls, ccode

    while nit <= maxiter:
        vf[nsims-1] = fr

        g1 = B.T @ g0
        dx = B @ (g1 / np.linalg.norm(g1))
        normdx = np.linalg.norm(dx)

        d = 1
        cal = 0
        deltax = 0

        while d > 0 and cal <= 500:
            x = x - hs*dx
            deltax = deltax + hs * normdx

            ncalls += 1
            f, g1 = calcfg(x)

            if f < fr:
                fr = f
                xr = x

            if np.linalg.norm(g1) < tolg:
                ccode = 2
                return xr, fr, nit, ncalls, ccode

            if np.mod(cal, nh) == 0:
                hs = hs * q2
            d = dx @ g1

            cal += 1

        if cal > 500:
            ccode = 5
            return xr, fr, nit, ncalls, ccode

        if cal == 1:
            hs = hs * q1

        if deltax < tolx:
            ccode = 3
            return xr, fr, nit, ncalls, ccode

        dg = B.T @ (g1 - g0)
        xi = dg / np.linalg.norm(dg)

        B = B + w * np.outer((B @ xi), xi)
        g0 = g1

        vf = np.roll(vf, 1)
        vf[0] = abs(fr - vf[0])

        if abs(fr) > 1:
            deltaf = np.sum(vf)/abs(fr)
        else:
            deltaf = np.sum(vf)

        if deltaf < tolf:
            ccode = 1
            return xr, fr, nit, ncalls, ccode

        nit += 1

    ccode=4
    return xr, fr, nit, ncalls, ccode
