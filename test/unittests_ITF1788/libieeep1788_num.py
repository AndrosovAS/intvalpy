#
# 
# Unit tests from libieeep1788 for interval numeric operations
# (Original author: Marco Nehmeier)
# converted into portable ITL format by Oliver Heimlich.
# 
# Copyright 2013-2015 Marco Nehmeier (nehmeier@informatik.uni-wuerzburg.de)
# Copyright 2015-2017 Oliver Heimlich (oheim@posteo.de)
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
#
#Language imports
import math
inf = float('Inf')
nan = float('NaN')

#Test library imports
import unittest
import warnings

#Arithmetic library imports
import intvalpy as ip
from intvalpy import Interval
import math

#Preamble
from intvalpy import Interval

warnings.simplefilter("always")
class NotTightest(UserWarning):
    """Happens when a test does not produce the most accurate result"""

suite = unittest.TestSuite()
class TestCase_minimal_inf_test(unittest.TestCase):
    """minimal_inf_test"""
    def test_1(self):
        self.assertEqual(Interval(ip.nan, ip.nan).a, inf)
    def test_2(self):
        self.assertEqual(Interval(-inf, inf).a, -inf)
    def test_3(self):
        self.assertEqual(Interval(1.0, 2.0).a, 1.0)
    def test_4(self):
        self.assertEqual(Interval(-3.0, -2.0).a, -3.0)
    def test_5(self):
        self.assertEqual(Interval(-inf, 2.0).a, -inf)
    def test_6(self):
        self.assertEqual(Interval(-inf, 0.0).a, -inf)
    def test_7(self):
        self.assertEqual(Interval(-inf, -0.0).a, -inf)
    def test_8(self):
        self.assertEqual(Interval(-2.0, inf).a, -2.0)
    def test_9(self):
        self.assertEqual(Interval(0.0, inf).a, -0.0)
    def test_10(self):
        self.assertEqual(Interval(-0.0, inf).a, -0.0)
    def test_11(self):
        self.assertEqual(Interval(-0.0, 0.0).a, -0.0)
    def test_12(self):
        self.assertEqual(Interval(0.0, -0.0).a, -0.0)
    def test_13(self):
        self.assertEqual(Interval(0.0, 0.0).a, -0.0)
    def test_14(self):
        self.assertEqual(Interval(-0.0, -0.0).a, -0.0)
suite.addTest(TestCase_minimal_inf_test())

class TestCase_minimal_inf_dec_test(unittest.TestCase):
    """minimal_inf_dec_test"""
    def test_15(self):
        self.assertEqual(None.a, nan)
suite.addTest(TestCase_minimal_inf_dec_test())

class TestCase_minimal_sup_test(unittest.TestCase):
    """minimal_sup_test"""
    def test_30(self):
        self.assertEqual(Interval(ip.nan, ip.nan).b, -inf)
    def test_31(self):
        self.assertEqual(Interval(-inf, inf).b, inf)
    def test_32(self):
        self.assertEqual(Interval(1.0, 2.0).b, 2.0)
    def test_33(self):
        self.assertEqual(Interval(-3.0, -2.0).b, -2.0)
    def test_34(self):
        self.assertEqual(Interval(-inf, 2.0).b, 2.0)
    def test_35(self):
        self.assertEqual(Interval(-inf, 0.0).b, 0.0)
    def test_36(self):
        self.assertEqual(Interval(-inf, -0.0).b, 0.0)
    def test_37(self):
        self.assertEqual(Interval(-2.0, inf).b, inf)
    def test_38(self):
        self.assertEqual(Interval(0.0, inf).b, inf)
    def test_39(self):
        self.assertEqual(Interval(-0.0, inf).b, inf)
    def test_40(self):
        self.assertEqual(Interval(-0.0, 0.0).b, 0.0)
    def test_41(self):
        self.assertEqual(Interval(0.0, -0.0).b, 0.0)
    def test_42(self):
        self.assertEqual(Interval(0.0, 0.0).b, 0.0)
    def test_43(self):
        self.assertEqual(Interval(-0.0, -0.0).b, 0.0)
suite.addTest(TestCase_minimal_sup_test())

class TestCase_minimal_sup_dec_test(unittest.TestCase):
    """minimal_sup_dec_test"""
    def test_44(self):
        self.assertEqual(None.b, nan)
suite.addTest(TestCase_minimal_sup_dec_test())

class TestCase_minimal_mid_test(unittest.TestCase):
    """minimal_mid_test"""
    def test_59(self):
        self.assertEqual(Interval(ip.nan, ip.nan).mid, nan)
    def test_60(self):
        self.assertEqual(Interval(-inf, inf).mid, 0.0)
    def test_61(self):
        self.assertEqual(Interval(float.fromhex('-0x1.FFFFFFFFFFFFFp1023'), float.fromhex('+0x1.FFFFFFFFFFFFFp1023')).mid, 0.0)
    def test_62(self):
        self.assertEqual(Interval(0.0, 2.0).mid, 1.0)
    def test_63(self):
        self.assertEqual(Interval(2.0, 2.0).mid, 2.0)
    def test_64(self):
        self.assertEqual(Interval(-2.0, 2.0).mid, 0.0)
    def test_65(self):
        self.assertEqual(Interval(0.0, inf).mid, float.fromhex('0x1.FFFFFFFFFFFFFp1023'))
    def test_66(self):
        self.assertEqual(Interval(-inf, 1.2).mid, float.fromhex('-0x1.FFFFFFFFFFFFFp1023'))
    def test_67(self):
        self.assertEqual(Interval(float.fromhex('-0X0.0000000000002P-1022'), float.fromhex('0X0.0000000000001P-1022')).mid, 0.0)
    def test_68(self):
        self.assertEqual(Interval(float.fromhex('-0X0.0000000000001P-1022'), float.fromhex('0X0.0000000000002P-1022')).mid, 0.0)
    def test_69(self):
        self.assertEqual(Interval(float.fromhex('0X1.FFFFFFFFFFFFFP+1022'), float.fromhex('0X1.FFFFFFFFFFFFFP+1023')).mid, float.fromhex('0X1.7FFFFFFFFFFFFP+1023'))
    def test_70(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000001P-1022'), float.fromhex('0X0.0000000000003P-1022')).mid, float.fromhex('0X0.0000000000002P-1022'))
suite.addTest(TestCase_minimal_mid_test())

class TestCase_minimal_mid_dec_test(unittest.TestCase):
    """minimal_mid_dec_test"""
    def test_72(self):
        self.assertEqual(None.mid, nan)
suite.addTest(TestCase_minimal_mid_dec_test())

class TestCase_minimal_rad_test(unittest.TestCase):
    """minimal_rad_test"""
    def test_0191_minimal_rad_test(self):
        self.assertEqual(Interval(0.0, 2.0).rad, 1.0)
    def test_0192_minimal_rad_test(self):
        self.assertEqual(Interval(2.0, 2.0).rad, 0.0)
    def test_0193_minimal_rad_test(self):
        self.assertEqual(Interval(math.nan, math.nan).rad, math.nan)
    def test_0194_minimal_rad_test(self):
        self.assertEqual(Interval(-math.inf, math.inf).rad, math.inf)
    def test_0195_minimal_rad_test(self):
        self.assertEqual(Interval(0.0, math.inf).rad, math.inf)
    def test_0196_minimal_rad_test(self):
        self.assertEqual(Interval(-math.inf, 1.2).rad, math.inf)
    def test_0197_minimal_rad_test(self):
        self.assertEqual(Interval(float.fromhex('-0X0.0000000000002P-1022'), float.fromhex('0X0.0000000000001P-1022')).rad, float.fromhex('0X0.0000000000002P-1022'))
    def test_0198_minimal_rad_test(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000001P-1022'), float.fromhex('0X0.0000000000002P-1022')).rad, float.fromhex('0X0.0000000000001P-1022'))
    def test_0199_minimal_rad_test(self):
        self.assertEqual(Interval(float.fromhex('0X1P+0'), float.fromhex('0X1.0000000000003P+0')).rad, float.fromhex('0X1P-51'))

suite.addTest(TestCase_minimal_rad_test())

class TestCase_minimal_rad_dec_test(unittest.TestCase):
    """minimal_rad_dec_test"""

suite.addTest(TestCase_minimal_rad_dec_test())

class TestCase_minimal_mid_rad_test(unittest.TestCase):
    """minimal_mid_rad_test"""

suite.addTest(TestCase_minimal_mid_rad_test())

class TestCase_minimal_mid_rad_dec_test(unittest.TestCase):
    """minimal_mid_rad_dec_test"""

suite.addTest(TestCase_minimal_mid_rad_dec_test())

class TestCase_minimal_wid_test(unittest.TestCase):
    """minimal_wid_test"""
    def test_128(self):
        self.assertEqual(Interval(2.0, 2.0).wid, 0.0)
    def test_129(self):
        self.assertEqual(Interval(1.0, 2.0).wid, 1.0)
    def test_130(self):
        self.assertEqual(Interval(1.0, inf).wid, inf)
    def test_131(self):
        self.assertEqual(Interval(-inf, 2.0).wid, inf)
    def test_132(self):
        self.assertEqual(Interval(-inf, inf).wid, inf)
    def test_133(self):
        self.assertEqual(Interval(ip.nan, ip.nan).wid, nan)
    def test_134(self):
        self.assertEqual(Interval(float.fromhex('0X1P+0'), float.fromhex('0X1.0000000000001P+0')).wid, float.fromhex('0X1P-52'))
    def test_135(self):
        self.assertEqual(Interval(float.fromhex('0X1P-1022'), float.fromhex('0X1.0000000000001P-1022')).wid, float.fromhex('0X0.0000000000001P-1022'))
suite.addTest(TestCase_minimal_wid_test())

class TestCase_minimal_wid_dec_test(unittest.TestCase):
    """minimal_wid_dec_test"""
    def test_142(self):
        self.assertEqual(None.wid, nan)
suite.addTest(TestCase_minimal_wid_dec_test())

class TestCase_minimal_mag_test(unittest.TestCase):
    """minimal_mag_test"""
    def test_0172_minimal_mag_test(self):
        self.assertEqual(Interval(1.0, 2.0).mag, 2.0)
    def test_0173_minimal_mag_test(self):
        self.assertEqual(Interval(-4.0, 2.0).mag, 4.0)
    def test_0174_minimal_mag_test(self):
        self.assertEqual(Interval(-math.inf, 2.0).mag, math.inf)
    def test_0175_minimal_mag_test(self):
        self.assertEqual(Interval(1.0, math.inf).mag, math.inf)
    def test_0176_minimal_mag_test(self):
        self.assertEqual(Interval(-math.inf, math.inf).mag, math.inf)
    def test_0177_minimal_mag_test(self):
        self.assertEqual(Interval(math.nan, math.nan).mag, math.nan)
    def test_0178_minimal_mag_test(self):
        self.assertEqual(Interval(-0.0, 0.0).mag, 0.0)
    def test_0179_minimal_mag_test(self):
        self.assertEqual(Interval(-0.0, -0.0).mag, 0.0)

suite.addTest(TestCase_minimal_mag_test())

class TestCase_minimal_mag_dec_test(unittest.TestCase):
    """minimal_mag_dec_test"""

suite.addTest(TestCase_minimal_mag_dec_test())

class TestCase_minimal_mig_test(unittest.TestCase):
    """minimal_mig_test"""
    def test_0180_minimal_mig_test(self):
        self.assertEqual(Interval(1.0, 2.0).mig, 1.0)
    def test_0181_minimal_mig_test(self):
        self.assertEqual(Interval(-4.0, 2.0).mig, 0.0)
    def test_0182_minimal_mig_test(self):
        self.assertEqual(Interval(-4.0, -2.0).mig, 2.0)
    def test_0183_minimal_mig_test(self):
        self.assertEqual(Interval(-math.inf, 2.0).mig, 0.0)
    def test_0184_minimal_mig_test(self):
        self.assertEqual(Interval(-math.inf, -2.0).mig, 2.0)
    def test_0185_minimal_mig_test(self):
        self.assertEqual(Interval(-1.0, math.inf).mig, 0.0)
    def test_0186_minimal_mig_test(self):
        self.assertEqual(Interval(1.0, math.inf).mig, 1.0)
    def test_0187_minimal_mig_test(self):
        self.assertEqual(Interval(-math.inf, math.inf).mig, 0.0)
    def test_0188_minimal_mig_test(self):
        self.assertEqual(Interval(math.nan, math.nan).mig, math.nan)
    def test_0189_minimal_mig_test(self):
        self.assertEqual(Interval(-0.0, 0.0).mig, 0.0)
    def test_0190_minimal_mig_test(self):
        self.assertEqual(Interval(-0.0, -0.0).mig, 0.0)

suite.addTest(TestCase_minimal_mig_test())

class TestCase_minimal_mig_dec_test(unittest.TestCase):
    """minimal_mig_dec_test"""

suite.addTest(TestCase_minimal_mig_dec_test())

if __name__ == '__main__':
    unittest.main()
