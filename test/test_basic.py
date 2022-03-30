import unittest

import numpy as np
from intvalpy import Interval, precision

precision.increasedPrecisionQ = False

class TestArithmeticsOperation(unittest.TestCase):

    def test_add(self):
        assert Interval(1, 2) + 2                                                == Interval(3, 4)
        assert -1 + Interval(-2, 1)                                              == Interval(-3, 0)
        assert Interval(1, 2) + Interval(-2, 3)                                  == Interval(-1, 5)
        assert Interval(1, 2) + Interval(3, -2, sortQ=False)                     == Interval(4, 0, sortQ=False)

        assert Interval(2, 1, sortQ=False) + 2                                   == Interval(4, 3, sortQ=False)
        assert -1 + Interval(1, -2, sortQ=False)                                 == Interval(0, -3, sortQ=False)
        assert Interval(3, -2, sortQ=False) + Interval(-1, 2)                    == Interval(2, 0, sortQ=False)
        assert Interval(3, -2, sortQ=False) + Interval(2, -1, sortQ=False)       == Interval(5, -3, sortQ=False)

    def test_sub(self):
        assert Interval(1, 2) - 2                                                == Interval(-1, 0)
        assert -1 - Interval(-2, 1)                                              == Interval(-2, 1)
        assert Interval(1, 2) - Interval(-2, 3)                                  == Interval(-2, 4)
        assert Interval(1, 2) - Interval(3, -2, sortQ=False)                     == Interval(3, -1, sortQ=False)

        assert Interval(2, 1, sortQ=False) - 2                                   == Interval(0, -1, sortQ=False)
        assert -1 - Interval(1, -2, sortQ=False)                                 == Interval(1, -2, sortQ=False)
        assert Interval(3, -2, sortQ=False) - Interval(-1, 2)                    == Interval(1, -1, sortQ=False)
        assert Interval(3, -2, sortQ=False) - Interval(2, -1, sortQ=False)       == Interval(4, -4, sortQ=False)

    def test_mul(self):
        assert Interval(1, 2) * 2                                                == Interval(2, 4)
        assert -1 * Interval(-2, 1)                                              == Interval(-1, 2)
        assert Interval(1, 2) * Interval(-2, 3)                                  == Interval(-4, 6)
        assert Interval(1, 2) * Interval(3, -2, sortQ=False)                     == Interval(3, -2, sortQ=False)

        assert Interval(2, 1, sortQ=False) * 2                                   == Interval(4, 2, sortQ=False)
        assert -1 * Interval(1, -2, sortQ=False)                                 == Interval(2, -1, sortQ=False)
        assert Interval(3, -2, sortQ=False) * Interval(-1, 2)                    == Interval(0, 0, sortQ=False)
        assert Interval(3, -2, sortQ=False) * Interval(2, -1, sortQ=False)       == Interval(6, -4, sortQ=False)

    def test_div(self):
        assert Interval(1, 2) / 2                                                == Interval(0.5, 1)
        assert -1 / Interval(-2, -1)                                             == Interval(0.5, 1)
        assert Interval(1, 2) / Interval(2, 5)                                   == Interval(0.2, 1)
        assert Interval(1, 2) / Interval(4, 2, sortQ=False)                      == Interval(0.5, 0.5, sortQ=False)

        assert Interval(2, 1, sortQ=False) / 2                                   == Interval(1, 0.5, sortQ=False)
        assert -1 / Interval(-1, -2, sortQ=False)                                == Interval(1, 0.5, sortQ=False)
        assert Interval(3, -2, sortQ=False) / Interval(-1, -2)                   == Interval(1, -1.5, sortQ=False)
        assert Interval(3, -2, sortQ=False) / Interval(2, 1, sortQ=False)        == Interval(3, -2, sortQ=False)

    # def test_pow(self):
    #     assert Interval(-2, 3) ** 2                                              == Interval(0, 9)
    #     assert Interval(2, 3) ** 2                                               == Interval(4, 9)
    #     assert Interval(-3, -2) ** 2                                             == Interval(4, 9)
    #
    #     assert Interval(-2, 3) ** 3                                              == Interval(-8, 27)
    #     assert Interval(2, 3) ** 3                                               == Interval(8, 27)
    #     assert Interval(-3, -2) ** 3                                             == Interval(-27, -8)
    #
    #     assert Interval(3, -2, sortQ=False) ** 2                                 == Interval(9, 4, sortQ=False)     # уточнить
    #     assert Interval(3, 2, sortQ=False) ** 2                                  == Interval(9, 4, sortQ=False)
    #     assert Interval(-2, -3, sortQ=False) ** 2                                == Interval(4, 9, sortQ=False)
    #
    #     assert Interval(3, -2, sortQ=False) ** 3                                 == Interval(27, -8, sortQ=False)
    #     assert Interval(3, 2, sortQ=False) ** 3                                  == Interval(27, 8, sortQ=False)
    #     assert Interval(-2, -3, sortQ=False) ** 3                                == Interval(-8, -27, sortQ=False)

    def test_matmul(self):
        A = Interval([
            [[2, 4], [-2, 1]],
            [[-1, 2], [2, 4]]
        ])

        B = Interval([
            [[1, 2], [-7, 6]],
            [[-7, 6], [1, 2]]
        ])
        x = Interval([[-2, 2], [1, 2]])

        assert ((x @ A) @ B) @ x                                                 == Interval(-384, 400)

if __name__ == '__main__':
    unittest.main()
