import numpy as np

def ralgb5(calcfg, x0, tolx=1e-12, tolg=1e-12, tolf=1e-12, maxiter=2000, alpha=2.3, nsims=30, h0=1, nh=3, q1=0.9, q2=1.1):
    """
    Subgradient method ralgb5 for minimizing convex functions with ravine-like structure.

    This function implements the ralgb5 algorithm, which is a subgradient method with adaptive stepsize control
    and space dilation. It is particularly effective for minimizing non-smooth convex functions or smooth convex
    functions with ravine-like level surfaces.

    Parameters:
    -----------
    calcfg : function
        A function that computes the value of the objective function and its subgradient at a given point x.
        The function should return a tuple (f, g), where f is the function value and g is the subgradient.

    x0 : numpy.ndarray
        The initial starting point for the optimization algorithm. It should be a 1D array of length n, where n
        is the number of variables.

    tolx : float, optional
        Tolerance for stopping based on the change in the argument (default is 1e-12). The algorithm stops if
        the norm of the change in x between iterations is less than tolx.

    tolg : float, optional
        Tolerance for stopping based on the norm of the subgradient (default is 1e-12). The algorithm stops if
        the norm of the subgradient is less than tolg.

    tolf : float, optional
        Tolerance for stopping based on the change in the function value (default is 1e-12). The algorithm stops
        if the relative change in the function value over the last nsims iterations is less than tolf.

    maxiter : int, optional
        Maximum number of iterations allowed (default is 2000). The algorithm stops if this number of iterations
        is reached.

    alpha : float, optional
        Space dilation coefficient (default is 2.3). This parameter controls the rate at which the space is
        stretched in the direction of the subgradient difference.

    nsims : int, optional
        Number of iterations to consider for the stopping criterion based on the change in the function value
        (default is 30).

    h0 : float, optional
        Initial step size (default is 1). This is the starting step size for the line search along the
        subgradient direction.

    nh : int, optional
        Number of steps after which the step size is increased (default is 3). The step size is increased by a
        factor of q2 every nh steps.

    q1 : float, optional
        Coefficient for decreasing the step size (default is 0.9). If the line search completes in one step, the
        step size is multiplied by q1.

    q2 : float, optional
        Coefficient for increasing the step size (default is 1.1). The step size is multiplied by q2 every nh
        steps.

    Returns:
    --------
    xr : numpy.ndarray
        The best approximation to the minimum point found by the algorithm.

    fr : float
        The value of the objective function at the point xr.

    nit : int
        The number of iterations performed.

    ncalls : int
        The number of calls to the calcfg function.

    ccode : int
        A code indicating the reason for stopping:
        - 1: Stopped based on the change in the function value (tolf).
        - 2: Stopped based on the norm of the subgradient (tolg).
        - 3: Stopped based on the change in the argument (tolx).
        - 4: Maximum number of iterations reached (maxiter).
        - 5: Emergency stop due to too many steps in the line search.

    Notes:
    ------
    The algorithm is based on the r-algorithm proposed by N.Z. Shor, with modifications for adaptive stepsize
    control and space dilation. It is particularly effective for minimizing non-smooth convex functions or
    smooth convex functions with ravine-like level surfaces.

    References:
    -----------
    [1] Stetsyuk, P.I. (2017). Subgradient methods ralgb5 and ralgb4 for minimization of ravine-like convex functions.
    [2] Shor, N.Z. (1979). Minimization Methods for Non-Differentiable Functions. Kiev: Naukova Dumka.
    """

    # Initialize variables
    m = len(x0) # Number of variables
    hs = h0 # Current step size
    B = np.eye(m) # Transformation matrix (initially identity)
    vf = np.zeros(nsims) + float('inf') # Array to store function values for stopping criterion
    w = 1./alpha - 1 # Coefficient for space dilation

    x = np.copy(x0) # Current point
    xr = np.copy(x0) # Best point found so far

    nit = 0 # Iteration counter
    ncalls = 1 # Counter for function evaluations
    fr, g0 = calcfg(xr) # Initial function value and subgradient

    # Check if the initial subgradient is already small enough
    if np.linalg.norm(g0) < tolg:
        ccode = 2 # Stopping code: subgradient norm is below tolerance
        return xr, fr, nit, ncalls, ccode

    # Main optimization loop
    while nit <= maxiter:
        vf[nsims-1] = fr # Store the current function value

        # Compute the direction of movement in the transformed space
        g1 = B.T @ g0
        dx = B @ (g1 / np.linalg.norm(g1))
        normdx = np.linalg.norm(dx)

        d = 1 # Initialize the inner loop condition
        cal = 0 # Counter for steps in the line search
        deltax = 0 # Accumulated change in x during the line search

        # Line search along the subgradient direction
        while d > 0 and cal <= 500:
            x = x - hs*dx # Update the current point
            deltax = deltax + hs * normdx # Accumulate the change in x

            ncalls += 1 # Increment the function evaluation counter
            f, g1 = calcfg(x) # Evaluate the function and subgradient at the new point

            # Update the best point found so far
            if f < fr:
                fr = f
                xr = x

            # Check if the subgradient norm is below tolerance
            if np.linalg.norm(g1) < tolg:
                ccode = 2 # Stopping code: subgradient norm is below tolerance
                return xr, fr, nit, ncalls, ccode

            # Increase the step size every nh steps
            if np.mod(cal, nh) == 0:
                hs = hs * q2
            
            # Check the inner loop condition
            d = dx @ g1

            cal += 1 # Increment the line search step counter

        # Emergency stop if too many steps are taken in the line search
        if cal > 500:
            ccode = 5 # Stopping code: emergency stop
            return xr, fr, nit, ncalls, ccode

        # Decrease the step size if the line search completes in one step
        if cal == 1:
            hs = hs * q1

        # Check if the change in x is below tolerance
        if deltax < tolx:
            ccode = 3 # Stopping code: change in x is below tolerance
            return xr, fr, nit, ncalls, ccode

        # Update the transformation matrix B and the subgradient
        dg = B.T @ (g1 - g0)
        xi = dg / np.linalg.norm(dg)
        B = B + w * np.outer((B @ xi), xi)
        g0 = g1

        # Update the function value history for the stopping criterion
        vf = np.roll(vf, 1)
        vf[0] = abs(fr - vf[0])

        # Compute the relative change in the function value
        if abs(fr) > 1:
            deltaf = np.sum(vf)/abs(fr)
        else:
            deltaf = np.sum(vf)

        # Check if the relative change in the function value is below tolerance
        if deltaf < tolf:
            ccode = 1 # Stopping code: change in function value is below tolerance
            return xr, fr, nit, ncalls, ccode

        nit += 1 # Increment the iteration counter

    # If the maximum number of iterations is reached
    ccode = 4  # Stopping code: maximum number of iterations reached
    return xr, fr, nit, ncalls, ccode
