import unittest
import numpy as np
from intvalpy.linear.overdetermined import TolSolSetEstimation
from intvalpy import Interval

class TolSolSetEstimationTests(unittest.TestCase):
    def test_Neumaier_method_for_straight_box(self):
        A = Interval([[1, 0], [0, 1]], [[1, 0], [0, 1]])
        b = Interval([-2, -3], [2, 3])
        w = np.array([2, 3])
        y_0 = np.array([0.0, 0.0])
        x = TolSolSetEstimation.Neumaier(A, b, y_0, w)
        assert (x == Interval([-2, -3], [2, 3])).all()

    def test_Neumaier_method_for_one_dim(self):
        A = Interval([[-1]], [[3]])
        b = Interval([-2], [4])
        y_0 = np.array([1.0])
        x = TolSolSetEstimation.Neumaier(A, b, y_0)
        y = A @ x
        assert (y in b)
        assert (abs(y.b - 4) < 0.0001) and (abs(y.a - (-1.33333)) < 0.0001)

