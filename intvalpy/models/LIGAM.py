import numpy as np
import pandas as pd

from ..kernel.utils import inf, sup, mid, rad
from ..kernel.ralgb5 import ralgb5
from ..kernel.real_intervals import Interval
from ..kernel.preprocessing import asinterval


class LIGAM(object):
    
    def __init__(self, alpha=0, beta=1):
        """
        Linear Interval Generalized Additive Model
        """
        self.alpha = alpha
        self.beta = beta
        

    def _wrap_norm(self, residual, norm):
        # print('residual: ', residual)
        if norm == 'inf':
            mcs = [np.argmin(residual)]
            obj = residual[ mcs[0] ]
            
        elif norm == 'l1':
            mcs = range(len(residual))
            obj = np.sum(residual)

        return obj, mcs
        
    
    def value_of_one_col(self, beta, x):
        """
        y = beta[0]*x + ... + beta[N-1]*x**(N-2) + beta[N]*x**(N-1)
        """
        def _value_of_one_element(inf, sup):
            roots = np.roots( np.polyder(beta, m=1) ) # roots of the first deriv
            roots = roots[np.isreal(roots)].real # only real roots
            roots = roots[(inf <= roots) & (roots <= sup)] # only in interval (bounded)
            roots = [inf, sup] + list(roots)
            vals = np.array([np.polyval(beta, root) for root in roots])
            return Interval(min(vals), max(vals), sortQ=False)

        infs, sups = inf(x), sup(x)
        # need reverse for np.polyder, np.polyval
        # the fictitious zero coefficient is needed for the free term
        beta = np.array([0] + list(beta))[::-1]
        if hasattr(x, '__iter__'):
            res = asinterval([_value_of_one_element(inf, sup) for inf, sup in zip(infs, sups)])
        else:
            res = _value_of_one_element(infs, sups)
        return res

    
    def value(self, betas, dataframe):
        return sum([self.value_of_one_col(betas[col], dataframe[col]) for col in betas.keys()])


    def _uni_calcfg_extend(self, beta, X_train, infX, supX, Xm, Xr, ym, yr, weight, norm):
        index = beta>=0
        betas = {}
        for k, col in enumerate(self.columns):
            betas[col] = beta[ sum(self.order[:k]): sum(self.order[:k+1]) ]
        centered_mval = ym - self.value(betas, X_train)
        infs, sups = centered_mval.inf, centered_mval.sup
        mig = np.array([
            0.0 if inf*sup <= 0 else min(abs(inf), abs(sup))
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
        index = beta>=0
        betas = {}
        for k, col in enumerate(self.columns):
            betas[col] = beta[ sum(self.order[:k]): sum(self.order[:k+1]) ]
        centered_mval = ym - self.value(betas, X_train)
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
        X_expand = pd.DataFrame()
        nit = 0
        for k, col in enumerate(self.columns):
            for l in range(1, self.order[k]+1):
                X_expand[nit] = dataframe[col]**l
                nit = nit + 1
        return X_expand

    
    def fit(
            self, 
            X_train, 
            y_train, 
            order=3,
            x0 = None, 
            weight=None, 
            objective='Tol', 
            norm='inf', 
            constraint=None, 
            **kwargs
        ):
        #+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
        X_train = pd.DataFrame(X_train)
        self.columns = X_train.columns
        n, m = X_train.shape
        if hasattr(order, '__iter__'):
            assert len(order)==m, 'Inconsistent order array sizes and the number of parameters.'
        else:
            order = np.full(m, order)
        self.order = order
        
        X_expand = self._expand_dataframe(X_train)
        infX, supX = inf(X_expand), sup(X_expand)
        Xm, Xr = mid(X_expand), rad(X_expand)
        ym, yr = mid(y_train), rad(y_train)
        del X_expand

        #+-----+-----+-----+-----+-----+-----+
        assert n == len(ym), 'Inconsistent dimensions of matrix and right-hand side vector.'
        assert (self.order >= 1).all(), 'The order must be greater than or equal to 1.'

        if x0 is None:
            Xm = np.array(Xm, dtype=np.float64)
            ym = np.array(ym, dtype=np.float64)

            sv = np.linalg.svd(Xm, compute_uv=False)
            minsv, maxsv = np.min(sv), np.max(sv)

            if (minsv != 0 and maxsv/minsv < 1e15):
                x0 = np.linalg.lstsq(Xm, ym, rcond=-1)[0]
            else:
                x0 = np.zeros(infX.shape[1])
        else:
            assert sum(self.order) == len(x0), 'Inconsistent dimensions of initial guess x0 and order.'
            x0 = np.copy(x0)
        beta0 = x0
        
        #+-----+-----+-----+-----+-----+-----+
        if weight is None:
            weight = np.ones(n)
        # делаем сглаживание
        weight[0] = self.beta * weight[0]*(1 - self.alpha)
        for k in range(1, n):
            weight[k] = self.beta * (weight[k]*(1 - self.alpha) + self.alpha*weight[k-1])

        #+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
        def calcfg(beta):
            if objective == 'Tol':
                obj, sub_diff = self._tol_calcfg_extend(beta, X_train, infX, supX, Xm, Xr, ym, yr, weight, norm)
            elif objective == 'Uni':
                obj, sub_diff = self._uni_calcfg_extend(beta, X_train, infX, supX, Xm, Xr, ym, yr, weight, norm)
            # elif objective == 'Uss':
            #     obj, sub_diff = self._uss_calcfg_extend(beta, infX, supX, Xm, Xr, ym, yr, weight, norm)
            else:
                raise Exception('Such a objective function is not provided.')

            # if constraint is not None:
            #     p, dp = BaseRecFun.linear_penalty(beta, constraint)
            # else:
            #     p, dp = 0.0, 0.0
                
            # return obj + p, sub_diff + dp
            return obj, sub_diff

        #+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
        xr, fr, nit, ncalls, ccode = ralgb5(calcfg, beta0, **kwargs)
        self.estimator = {
            'xr': xr, 
            'fr': -fr, 
            'nit': nit, 
            'ncalls': ncalls, 
            'ccode': ccode
        }


    def predict(self, X_test):
        X_expand = self._expand_dataframe(X_test)
        return X_expand @ self.estimator['xr']