import numpy as np
from ..kernel.utils import inf, sup, mid, rad
from ..kernel.abstract import BaseRecFun
from ..kernel.ralgb5 import ralgb5


class WILS(object):
    
    def __init__(self, alpha=0, beta=1):
        """
        Weighted Interval Linear System of several variables.
        
        """
        self.alpha = alpha
        self.beta = beta


    def _wrap_norm(self, residual, norm):
        if norm == 'inf':
            mcs = [np.argmin(residual)]
            obj = residual[ mcs[0] ]
            
        elif norm == 'l1':
            mcs = range(len(residual))
            obj = np.sum(residual)
            
        return obj, mcs

    # TODO
    def _uni_calcfg_extend(self, beta, infX, supX, Xm, Xr, ym, yr, weight, norm):
        index = beta>=0
        Xm_beta = Xm @ beta
        Xr_absbeta = Xr @ np.abs(beta)
        infs = ym - (Xm_beta + Xr_absbeta)
        sups = ym - (Xm_beta - Xr_absbeta)
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


    def _uss_calcfg_extend(self, beta, infX, supX, Xm, Xr, ym, yr, weight, norm):
        sign_beta = np.sign(beta)
        ym_Xmbeta = ym - Xm @ beta
        residual = weight * (yr + Xr @ np.abs(beta) - np.abs(ym_Xmbeta))

        obj, mcs = self._wrap_norm(residual, norm)
        sub_diff = np.zeros(len(beta), dtype=float)
        for mc in mcs:
            diff = weight[mc] * (sign_beta*Xr[mc] + np.sign(ym_Xmbeta[mc]) * Xm[mc])
            sub_diff = sub_diff + diff
        return -obj, -sub_diff
        

    def _tol_calcfg_extend(self, beta, infX, supX, Xm, Xr, ym, yr, weight, norm):
        index = beta>=0
        Xm_beta = Xm @ beta
        Xr_absbeta = Xr @ np.abs(beta)
        infs = ym - (Xm_beta + Xr_absbeta)
        sups = ym - (Xm_beta - Xr_absbeta)
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

    
    def fit(self, X_train, y_train, x0=None, weight=None, objective='Tol', norm='inf', constraint=None, **kwargs):

        #+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
        infX, supX = inf(X_train), sup(X_train)
        Xm, Xr = mid(X_train), rad(X_train)
        ym, yr = mid(y_train), rad(y_train)
        
        n, m = infX.shape
        assert n == len(ym), 'Inconsistent dimensions of matrix and right-hand side vector'

        if x0 is None:
            Xm = np.array(Xm, dtype=np.float64)
            ym = np.array(ym, dtype=np.float64)

            sv = np.linalg.svd(Xm, compute_uv=False)
            minsv, maxsv = np.min(sv), np.max(sv)

            if (minsv != 0 and maxsv/minsv < 1e15):
                x0 = np.linalg.lstsq(Xm, ym, rcond=-1)[0]
            else:
                x0 = np.zeros(m)
        else:
            x0 = np.copy(x0)
        
        if weight is None:
            weight = np.ones(n)
        # делаем сглаживание
        weight[0] = self.beta * weight[0]*(1 - self.alpha)
        for k in range(1, n):
            weight[k] = self.beta * (weight[k]*(1 - self.alpha) + self.alpha*weight[k-1])


        #+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
        def calcfg(beta):
            if objective == 'Tol':
                obj, sub_diff = self._tol_calcfg_extend(beta, infX, supX, Xm, Xr, ym, yr, weight, norm)
            elif objective == 'Uni':
                obj, sub_diff = self._uni_calcfg_extend(beta, infX, supX, Xm, Xr, ym, yr, weight, norm)
            elif objective == 'Uss':
                obj, sub_diff = self._uss_calcfg_extend(beta, infX, supX, Xm, Xr, ym, yr, weight, norm)
            else:
                raise Exception('Such a objective function is not provided.')

            if constraint is not None:
                p, dp = BaseRecFun.linear_penalty(beta, constraint)
            else:
                p, dp = 0.0, 0.0
                
            return obj + p, sub_diff + dp

        #+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
        xr, fr, nit, ncalls, ccode = ralgb5(calcfg, x0, **kwargs)
        self.estimator = {
            'xr': xr, 
            'fr': -fr, 
            'nit': nit, 
            'ncalls': ncalls, 
            'ccode': ccode
        }

    
    def predict(self, X_test):
        return X_test @ self.estimator['xr']
