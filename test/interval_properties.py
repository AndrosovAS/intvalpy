import unittest
import numpy as np

from intvalpy import Interval, precision
precision.extendedPrecisionQ = False


class TestsOfIntervalClassMethods(unittest.TestCase):

    def test_a(self):
        assert Interval(-2, 3).a                                                 == -2
        assert Interval(3, -2, sortQ=False).a                                    == 3
        assert (Interval([ [-1, 2], [0.5, 0] ], sortQ=False).a                   == np.array([-1, 0.5])).all()

        assert Interval(-2, 3).inf                                               == -2
        assert Interval(3, -2, sortQ=False).inf                                  == 3
        assert (Interval([ [-1, 2], [0.5, 0] ], sortQ=False).inf                 == np.array([-1, 0.5])).all()


    def test_b(self):
        assert Interval(-2, 3).b                                                 == 3
        assert Interval(3, -2, sortQ=False).b                                    == -2
        assert (Interval([ [-1, 2], [0.5, 0] ], sortQ=False).b                   == np.array([2, 0])).all()

        assert Interval(-2, 3).sup                                               == 3
        assert Interval(3, -2, sortQ=False).sup                                  == -2
        assert (Interval([ [-1, 2], [0.5, 0]] , sortQ=False).sup                 == np.array([2, 0])).all()


    def test_copy(self):
        f = Interval([ [-1, 2], [0.5, 0] ], sortQ=False)

        s = f.copy
        s[0] = Interval(0, 0)
        assert f[0]                                                              != s[0]

        s = f
        s[0] = Interval(0, 0)
        assert f[0]                                                              == s[0]


    def test_wid(self):
        assert Interval(2, 4).wid                                                == 2
        assert Interval(1, -2, sortQ=False).wid                                  == -3
        assert (Interval([ [-1, 2], [0.5, 0] ], sortQ=False).wid                 == np.array([3, -0.5])).all()


    def test_rad(self):
        assert Interval(2, 4).rad                                                == 1
        assert Interval(1, -2, sortQ=False).rad                                  == -1.5
        assert (Interval([ [-1, 2], [0.5, 0] ], sortQ=False).rad                 == np.array([1.5, -0.25])).all()


    def test_mid(self):
        assert Interval(2, 4).mid                                                == 3
        assert Interval(1, -2, sortQ=False).mid                                  == -0.5
        assert (Interval([ [-1, 2], [0.5, 0] ], sortQ=False).mid                 == np.array([0.5, 0.25])).all()


    def test_mig(self):
        assert Interval(2, 4).mig                                                == 2
        assert Interval(1, -2, sortQ=False).mig                                  == 1
        assert (Interval([ [-1, 2], [0.5, 0] ], sortQ=False).mig                 == np.array([0, 0])).all()


    def test_mag(self):
        assert Interval(2, 4).mag                                                == 4
        assert Interval(1, -2, sortQ=False).mag                                  == 2
        assert (Interval([ [-1, 2], [0.5, 0] ], sortQ=False).mag                 == np.array([2, 0.5])).all()


    def test_dual(self):
        assert Interval(-2, 3).dual                                              == Interval(3, -2, sortQ=False)
        assert Interval(3, -2, sortQ=False).dual                                 == Interval(-2, 3)
        assert (Interval([[-2, 3], [3, -2]], sortQ=False).dual                   == Interval([[3, -2], [-2, 3]], sortQ=False)).all()


    def test_pro(self):
        assert Interval(-2, 3).pro                                               == Interval(-2, 3)
        assert Interval(3, -2, sortQ=False).pro                                  == Interval(-2, 3)
        assert (Interval([[-2, 3], [3, -2]], sortQ=False).pro                    == Interval([[-2, 3], [-2, 3]])).all()


    def test_opp(self):
        assert Interval(-2, 3).opp                                               == Interval(2, -3, sortQ=False)
        assert Interval(3, -2, sortQ=False).opp                                  == Interval(-3, 2)
        assert (Interval([[-2, 3], [3, -2]], sortQ=False).opp                    == Interval([[2, -3], [-3, 2]], sortQ=False)).all()


    def test_inv(self):
        assert Interval(1, 4).inv                                                == Interval(1, 0.25, sortQ=False)
        assert Interval(4, -2, sortQ=False).inv                                  == Interval(0.25, -0.5, sortQ=False)
        assert (Interval([[1, 4], [4, -2]], sortQ=False).inv                     == Interval([[1, 0.25], [0.25, -0.5]], sortQ=False)).all()


if __name__ == '__main__':
    unittest.main()
