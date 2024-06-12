import numpy as np

from ..linear.Tol import Tol
from ..linear.Uss import Uss
from ..linear.Uni import Uni


class WILS(object):

    def init(self, alpha=0, beta=1):
        self.alpha = alpha
        self.beta = beta

    def fit(self, X_train, y_train, weight=None, rec_func='Tol', **kwargs):
        if weight is None:
            weight = np.ones(len(y_train))

        # делаем сглаживание
        weight[0] = self.beta * weight[0]*(1 - self.alpha)
        for k in range(1, len(weight)):
            weight[k] = self.beta * (weight[k]*(1 - self.alpha) + self.alpha*weight[k-1])

        if rec_func == 'Tol':
            RecFunc = Tol
        elif rec_func == 'Uni':
            RecFunc = Uni
        elif rec_func == 'Uss':
            RecFunc = Uss

        self.estimator = RecFunc.maximize(
            X_train, 
            y_train, 
            weight=weight,
            **kwargs
        )

    def predict(self, X_test):
        return X_test @ self.estimator[0]