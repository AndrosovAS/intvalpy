import numpy as np
import pandas as pd

from ..kernel.utils import inf, sup, mid, rad
from ..kernel.ralgb5 import ralgb5
from ..kernel.real_intervals import Interval
from ..kernel.preprocessing import asinterval


class ISPAE(object):
    """
    Interval System of Polynomial Algebraic Equations (ISPAE).
    This class implements a linear interval additive model for regression tasks.
    It supports various polynomial bases and interval arithmetic for handling uncertainty in data.
    """

    def __init__(self, alpha=0, beta=1, basis='monom'):
        """
        Initialize the ISPAE model.

        Parameters:
        -----------
        alpha : float, optional
            Smoothing parameter for weights (default: 0).
        beta : float, optional
            Scaling parameter for weights (default: 1).
        basis : str, optional
            Type of polynomial basis to use. 
            Options: 'monom', 'chebyshev', 'hermite', 'laguerre', 'legendre' (default: 'monom').
        """
        self.alpha = alpha
        self.beta = beta

        # Define available polynomial bases
        self.__bases = {
            'monom': np.polynomial.polynomial.Polynomial.basis,
            'chebyshev': np.polynomial.chebyshev.Chebyshev.basis,
            'hermite': np.polynomial.hermite.Hermite.basis,
            'laguerre': np.polynomial.laguerre.Laguerre.basis,
            'legendre': np.polynomial.legendre.Legendre.basis,
        }
        self.basis = self.__bases[basis]

    
    def _wrap_norm(self, residual, norm):
        """
        Compute the objective function value and indices based on the specified norm.

        Parameters:
        -----------
        residual : np.ndarray
            Residual values.
        norm : str
            Norm type. Options: 'inf' (infinity norm), 'l1' (L1 norm).

        Returns:
        --------
        obj : float
            Objective function value.
        mcs : list or range
            Indices of residuals contributing to the objective.
        """
        if norm == 'inf':
            mcs = [np.argmin(residual)]
            obj = residual[mcs[0]]
        elif norm == 'l1':
            mcs = list(range(len(residual)))
            obj = np.sum(residual)
        return obj, mcs

        
    def value_of_one_col(self, beta, x, deg=None):
        """
        Compute the interval value of a polynomial for a single feature column.

        Parameters:
        -----------
        beta : np.ndarray
            Coefficients of the polynomial.
        x : Interval or np.ndarray
            Input data (interval or point values).
        deg : np.ndarray, optional
            Degrees of the polynomial basis (default: None).

        Returns:
        --------
        res : Interval
            Interval value of the polynomial.
        """
        # Extract interval bounds
        infs, sups = np.array([inf(x)]).flatten(), np.array([sup(x)]).flatten()
        beta = np.array([beta]).flatten()

        # Initialize polynomial basis
        if deg is None:
            basis = np.array([self.basis(k) for k in range(1, len(beta) + 1)])
        else:
            deg = np.array([deg]).flatten()
            assert len(deg) == len(beta), "Degree array length must match beta coefficients."
            basis = np.array([self.basis(d) for d in deg])

        # Compute polynomial and its roots
        poly = beta @ basis
        roots = poly.deriv(1).roots().real

        # Evaluate polynomial at interval bounds and roots
        vals = np.zeros((len(infs), 2 + len(roots)), dtype=float)
        vals[:, 0] = poly(infs)
        vals[:, 1] = poly(sups)
        vals[:, 2:] = poly(roots)

        # Determine valid points within the interval
        dots = np.zeros((len(infs), 2 + len(roots)), dtype=float)
        dots[:, 0] = infs
        dots[:, 1] = sups
        dots[:, 2:] = roots
        masks = (infs[:, np.newaxis] <= dots) & (dots <= sups[:, np.newaxis])

        # Compute interval result
        res = asinterval([
            Interval(min(val[mask]), max(val[mask]), sortQ=False)
            for val, mask in zip(vals, masks)
        ])
        return res

    
    def value(self, beta, dataframe):
        """
        Compute the interval value of the polynomial for all feature columns.

        Parameters:
        -----------
        beta : np.ndarray
            Coefficients of the polynomial.
        dataframe : pd.DataFrame
            Input data.

        Returns:
        --------
        sum(vals_by_cols) : Interval
            Sum of interval values for all columns.
        """
        vals_by_cols = [
            self.value_of_one_col(beta[sum(self.order[:k]): sum(self.order[:k + 1])], dataframe[col])
            for k, col in enumerate(self.columns)
        ]
        return sum(vals_by_cols)

    
    def _uni_calcfg_extend(self, beta, X_train, infX, supX, Xm, Xr, ym, yr, weight, norm):
        """
        Compute objective and subgradient for the 'Uni' objective function.

        Parameters:
        -----------
        beta : np.ndarray
            Coefficients of the polynomial.
        X_train : pd.DataFrame
            Training data.
        infX, supX : np.ndarray
            Lower and upper bounds of the expanded data.
        Xm, Xr : np.ndarray
            Midpoints and radiuses of the expanded data.
        ym, yr : np.ndarray
            Midpoints and radiuses of the target intervals.
        weight : np.ndarray
            Weights for residuals.
        norm : str
            Norm type.

        Returns:
        --------
        obj : float
            Objective function value.
        sub_diff : np.ndarray
            Subgradient of the objective.
        """
        index = beta >= 0
        centered_mval = ym - self.value(beta, X_train)
        infs, sups = centered_mval.inf, centered_mval.sup
        mig = np.array([
            0.0 if inf * sup <= 0 else min(abs(inf), abs(sup))
            for inf, sup in zip(infs, sups)
        ])
        residual = weight * (yr - mig)

        obj, mcs = self._wrap_norm(residual, norm)
        sub_diff = np.zeros(len(beta), dtype=float)
        for mc in mcs:
            if -infs[mc] <= sups[mc]:
                diff = weight[mc] * (supX[mc] * index + infX[mc] * (~index))
            else:
                diff = -weight[mc] * (infX[mc] * index + supX[mc] * (~index))
            sub_diff = sub_diff + diff
        return -obj, -sub_diff

        
    def _tol_calcfg_extend(self, beta, X_train, infX, supX, Xm, Xr, ym, yr, weight, norm):
        """
        Compute objective and subgradient for the 'Tol' objective function.

        Parameters:
        -----------
        beta : np.ndarray
            Coefficients of the polynomial.
        X_train : pd.DataFrame
            Training data.
        infX, supX : np.ndarray
            Lower and upper bounds of the expanded data.
        Xm, Xr : np.ndarray
            Midpoints and radiuses of the expanded data.
        ym, yr : np.ndarray
            Midpoints and radiuses of the target intervals.
        weight : np.ndarray
            Weights for residuals.
        norm : str
            Norm type.

        Returns:
        --------
        obj : float
            Objective function value.
        sub_diff : np.ndarray
            Subgradient of the objective.
        """
        index = beta >= 0
        centered_mval = ym - self.value(beta, X_train)
        infs, sups = centered_mval.inf, centered_mval.sup
        residual = weight * (yr - np.maximum(abs(infs), abs(sups)))

        obj, mcs = self._wrap_norm(residual, norm)
        sub_diff = np.zeros(len(beta), dtype=float)
        for mc in mcs:
            if -infs[mc] <= sups[mc]:
                diff = weight[mc] * (infX[mc] * index + supX[mc] * (~index))
            else:
                diff = -weight[mc] * (supX[mc] * index + infX[mc] * (~index))
            sub_diff = sub_diff + diff
        return -obj, -sub_diff

    
    def _expand_dataframe(self, dataframe):
        """
        Expand the input dataframe using the polynomial basis.

        Parameters:
        -----------
        dataframe : pd.DataFrame
            Input data.

        Returns:
        --------
        X_expand : pd.DataFrame
            Expanded dataframe with polynomial features.
        """
        X_expand = pd.DataFrame()
        nit = 0
        for k, col in enumerate(self.columns):
            for deg in range(1, self.order[k] + 1):
                X_expand[nit] = self.value_of_one_col(1, dataframe[col], deg=deg)
                nit = nit + 1
        return X_expand

    
    def fit(
            self,
            X_train,
            y_train,
            order=3,
            x0=None,
            weight=None,
            objective='Tol',
            norm='inf',
            **kwargs
    ):
        """
        Fit the ISPAE model to the training data.

        Parameters:
        -----------
        X_train : pd.DataFrame or np.ndarray
            Training data (features).
        y_train : Interval or np.ndarray
            Training data (target intervals).
        order : int or list, optional
            Order of the polynomial for each feature (default: 3).
        x0 : np.ndarray, optional
            Initial guess for coefficients (default: None).
        weight : np.ndarray, optional
            Weights for residuals (default: None).
        objective : str, optional
            Objective function type. Options: 'Tol', 'Uni' (default: 'Tol').
        norm : str, optional
            Norm type for the objective function. Options: 'inf', 'l1' (default: 'inf').
        **kwargs : dict
            Additional arguments for the optimizer ralgb5.
        """
        # Initialize training data
        X_train = pd.DataFrame(X_train)
        self.columns = X_train.columns
        n, m = X_train.shape

        # Validate and set polynomial order
        if hasattr(order, '__iter__'):
            assert len(order) == m, "Order array size must match the number of features."
        else:
            order = np.full(m, order)
        self.order = np.array(order)

        # Expand the dataframe with polynomial features
        X_expand = self._expand_dataframe(X_train)
        infX, supX = inf(X_expand), sup(X_expand)
        Xm, Xr = mid(X_expand), rad(X_expand)
        ym, yr = mid(y_train), rad(y_train)
        del X_expand

        # Validate dimensions
        assert n == len(ym), "Matrix and target vector dimensions do not match."
        assert (self.order >= 1).all(), "Order must be greater than or equal to 1."

        # Initialize coefficients
        if x0 is None:
            Xm = np.array(Xm, dtype=np.float64)
            ym = np.array(ym, dtype=np.float64)

            sv = np.linalg.svd(Xm, compute_uv=False)
            minsv, maxsv = np.min(sv), np.max(sv)

            if minsv != 0 and maxsv / minsv < 1e15:
                x0 = np.linalg.lstsq(Xm, ym, rcond=-1)[0]
            else:
                x0 = np.zeros(infX.shape[1])
        else:
            assert sum(self.order) == len(x0), "Initial guess x0 must match the order."
            x0 = np.copy(x0)
        beta0 = x0

        # Initialize weights
        if weight is None:
            weight = np.ones(n)
        weight[0] = self.beta * weight[0] * (1 - self.alpha)
        for k in range(1, n):
            weight[k] = self.beta * (weight[k] * (1 - self.alpha) + self.alpha * weight[k - 1])

        # Define the objective function
        def calcfg(beta):
            if objective == 'Tol':
                obj, sub_diff = self._tol_calcfg_extend(beta, X_train, infX, supX, Xm, Xr, ym, yr, weight, norm)
            elif objective == 'Uni':
                obj, sub_diff = self._uni_calcfg_extend(beta, X_train, infX, supX, Xm, Xr, ym, yr, weight, norm)
            else:
                raise ValueError("Invalid objective function.")
            return obj, sub_diff

        # Optimize using ralgb5
        xr, fr, nit, ncalls, ccode = ralgb5(calcfg, beta0, **kwargs)
        self.estimator = {
            'xr': xr,
            'fr': -fr,
            'nit': nit,
            'ncalls': ncalls,
            'ccode': ccode
        }

    
    def predict(self, X_test):
        """
        Predict target intervals for the test data.

        Parameters:
        -----------
        X_test : pd.DataFrame or np.ndarray
            Test data (features).

        Returns:
        --------
        predictions : Interval
            Predicted target intervals.
        """
        return self._expand_dataframe(X_test) @ self.estimator['xr']