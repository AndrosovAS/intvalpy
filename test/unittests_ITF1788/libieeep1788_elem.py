#
# 
# Unit tests from libieeep1788 for elementary interval functions
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
class TestCase_minimal_pos_test(unittest.TestCase):
    """minimal_pos_test"""

suite.addTest(TestCase_minimal_pos_test())

class TestCase_minimal_pos_dec_test(unittest.TestCase):
    """minimal_pos_dec_test"""

suite.addTest(TestCase_minimal_pos_dec_test())

class TestCase_minimal_neg_test(unittest.TestCase):
    """minimal_neg_test"""
    def test_16(self):
        self.assertEqual(-(Interval(1.0, 2.0)), Interval(-2.0, -1.0))
    def test_17(self):
        self.assertEqual(-(Interval(ip.nan, ip.nan)), Interval(ip.nan, ip.nan))
    def test_18(self):
        self.assertEqual(-(Interval(-math.inf, math.inf)), Interval(-math.inf, math.inf))
    def test_19(self):
        self.assertEqual(-(Interval(1.0, inf)), Interval(-inf, -1.0))
    def test_20(self):
        self.assertEqual(-(Interval(-inf, 1.0)), Interval(-1.0, inf))
    def test_21(self):
        self.assertEqual(-(Interval(0.0, 2.0)), Interval(-2.0, 0.0))
    def test_22(self):
        self.assertEqual(-(Interval(-0.0, 2.0)), Interval(-2.0, 0.0))
    def test_23(self):
        self.assertEqual(-(Interval(-2.0, 0.0)), Interval(0.0, 2.0))
    def test_24(self):
        self.assertEqual(-(Interval(-2.0, -0.0)), Interval(0.0, 2.0))
    def test_25(self):
        self.assertEqual(-(Interval(0.0, -0.0)), Interval(0.0, 0.0))
    def test_26(self):
        self.assertEqual(-(Interval(-0.0, -0.0)), Interval(0.0, 0.0))
suite.addTest(TestCase_minimal_neg_test())

class TestCase_minimal_neg_dec_test(unittest.TestCase):
    """minimal_neg_dec_test"""
    def test_27(self):
        self.assertEqual(-(None), None)
suite.addTest(TestCase_minimal_neg_dec_test())

class TestCase_minimal_add_test(unittest.TestCase):
    """minimal_add_test"""
    def test_31(self):
        self.assertEqual(Interval(ip.nan, ip.nan) + Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_32(self):
        self.assertEqual(Interval(-1.0, 1.0) + Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_33(self):
        self.assertEqual(Interval(ip.nan, ip.nan) + Interval(-1.0, 1.0), Interval(ip.nan, ip.nan))
    def test_34(self):
        self.assertEqual(Interval(ip.nan, ip.nan) + Interval(-math.inf, math.inf), Interval(ip.nan, ip.nan))
    def test_35(self):
        self.assertEqual(Interval(-math.inf, math.inf) + Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_36(self):
        self.assertEqual(Interval(-math.inf, math.inf) + Interval(-inf, 1.0), Interval(-math.inf, math.inf))
    def test_37(self):
        self.assertEqual(Interval(-math.inf, math.inf) + Interval(-1.0, 1.0), Interval(-math.inf, math.inf))
    def test_38(self):
        self.assertEqual(Interval(-math.inf, math.inf) + Interval(-1.0, inf), Interval(-math.inf, math.inf))
    def test_39(self):
        self.assertEqual(Interval(-math.inf, math.inf) + Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_40(self):
        self.assertEqual(Interval(-inf, 1.0) + Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_41(self):
        self.assertEqual(Interval(-1.0, 1.0) + Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_42(self):
        self.assertEqual(Interval(-1.0, inf) + Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_43(self):
        self.assertEqual(Interval(-inf, 2.0) + Interval(-inf, 4.0), Interval(-inf, 6.0))
    def test_44(self):
        self.assertEqual(Interval(-inf, 2.0) + Interval(3.0, 4.0), Interval(-inf, 6.0))
    def test_45(self):
        self.assertEqual(Interval(-inf, 2.0) + Interval(3.0, inf), Interval(-math.inf, math.inf))
    def test_46(self):
        self.assertEqual(Interval(1.0, 2.0) + Interval(-inf, 4.0), Interval(-inf, 6.0))
    def test_47(self):
        self.assertEqual(Interval(1.0, 2.0) + Interval(3.0, 4.0), Interval(4.0, 6.0))
    def test_48(self):
        self.assertEqual(Interval(1.0, 2.0) + Interval(3.0, inf), Interval(4.0, inf))
    def test_49(self):
        self.assertEqual(Interval(1.0, inf) + Interval(-inf, 4.0), Interval(-math.inf, math.inf))
    def test_50(self):
        self.assertEqual(Interval(1.0, inf) + Interval(3.0, 4.0), Interval(4.0, inf))
    def test_51(self):
        self.assertEqual(Interval(1.0, inf) + Interval(3.0, inf), Interval(4.0, inf))
    def test_52(self):
        self.assertEqual(Interval(1.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')) + Interval(3.0, 4.0), Interval(4.0, inf))
    def test_53(self):
        self.assertEqual(Interval(float.fromhex('-0x1.FFFFFFFFFFFFFp1023'), 2.0) + Interval(-3.0, 4.0), Interval(-inf, 6.0))
    def test_54(self):
        self.assertEqual(Interval(float.fromhex('-0x1.FFFFFFFFFFFFFp1023'), 2.0) + Interval(-3.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')), Interval(-math.inf, math.inf))
    def test_55(self):
        self.assertEqual(Interval(1.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')) + Interval(0.0, 0.0), Interval(1.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')))
    def test_56(self):
        self.assertEqual(Interval(1.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')) + Interval(-0.0, -0.0), Interval(1.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')))
    def test_57(self):
        self.assertEqual(Interval(0.0, 0.0) + Interval(-3.0, 4.0), Interval(-3.0, 4.0))
    def test_58(self):
        self.assertEqual(Interval(-0.0, -0.0) + Interval(-3.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')), Interval(-3.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')))
    def test_59(self):
        self.assertEqual(Interval(float.fromhex('0X1.FFFFFFFFFFFFP+0'), float.fromhex('0X1.FFFFFFFFFFFFP+0')) + Interval(float.fromhex('0X1.999999999999AP-4'), float.fromhex('0X1.999999999999AP-4')), Interval(float.fromhex('0X1.0CCCCCCCCCCC4P+1'), float.fromhex('0X1.0CCCCCCCCCCC5P+1')))
    def test_60(self):
        self.assertEqual(Interval(float.fromhex('0X1.FFFFFFFFFFFFP+0'), float.fromhex('0X1.FFFFFFFFFFFFP+0')) + Interval(float.fromhex('-0X1.999999999999AP-4'), float.fromhex('-0X1.999999999999AP-4')), Interval(float.fromhex('0X1.E666666666656P+0'), float.fromhex('0X1.E666666666657P+0')))
    def test_61(self):
        self.assertEqual(Interval(float.fromhex('-0X1.FFFFFFFFFFFFP+0'), float.fromhex('0X1.FFFFFFFFFFFFP+0')) + Interval(float.fromhex('0X1.999999999999AP-4'), float.fromhex('0X1.999999999999AP-4')), Interval(float.fromhex('-0X1.E666666666657P+0'), float.fromhex('0X1.0CCCCCCCCCCC5P+1')))
suite.addTest(TestCase_minimal_add_test())

class TestCase_minimal_add_dec_test(unittest.TestCase):
    """minimal_add_dec_test"""

suite.addTest(TestCase_minimal_add_dec_test())

class TestCase_minimal_sub_test(unittest.TestCase):
    """minimal_sub_test"""
    def test_68(self):
        self.assertEqual(Interval(ip.nan, ip.nan) - Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_69(self):
        self.assertEqual(Interval(-1.0, 1.0) - Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_70(self):
        self.assertEqual(Interval(ip.nan, ip.nan) - Interval(-1.0, 1.0), Interval(ip.nan, ip.nan))
    def test_71(self):
        self.assertEqual(Interval(ip.nan, ip.nan) - Interval(-math.inf, math.inf), Interval(ip.nan, ip.nan))
    def test_72(self):
        self.assertEqual(Interval(-math.inf, math.inf) - Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_73(self):
        self.assertEqual(Interval(-math.inf, math.inf) - Interval(-inf, 1.0), Interval(-math.inf, math.inf))
    def test_74(self):
        self.assertEqual(Interval(-math.inf, math.inf) - Interval(-1.0, 1.0), Interval(-math.inf, math.inf))
    def test_75(self):
        self.assertEqual(Interval(-math.inf, math.inf) - Interval(-1.0, inf), Interval(-math.inf, math.inf))
    def test_76(self):
        self.assertEqual(Interval(-math.inf, math.inf) - Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_77(self):
        self.assertEqual(Interval(-inf, 1.0) - Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_78(self):
        self.assertEqual(Interval(-1.0, 1.0) - Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_79(self):
        self.assertEqual(Interval(-1.0, inf) - Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_80(self):
        self.assertEqual(Interval(-inf, 2.0) - Interval(-inf, 4.0), Interval(-math.inf, math.inf))
    def test_81(self):
        self.assertEqual(Interval(-inf, 2.0) - Interval(3.0, 4.0), Interval(-inf, -1.0))
    def test_82(self):
        self.assertEqual(Interval(-inf, 2.0) - Interval(3.0, inf), Interval(-inf, -1.0))
    def test_83(self):
        self.assertEqual(Interval(1.0, 2.0) - Interval(-inf, 4.0), Interval(-3.0, inf))
    def test_84(self):
        self.assertEqual(Interval(1.0, 2.0) - Interval(3.0, 4.0), Interval(-3.0, -1.0))
    def test_85(self):
        self.assertEqual(Interval(1.0, 2.0) - Interval(3.0, inf), Interval(-inf, -1.0))
    def test_86(self):
        self.assertEqual(Interval(1.0, inf) - Interval(-inf, 4.0), Interval(-3.0, inf))
    def test_87(self):
        self.assertEqual(Interval(1.0, inf) - Interval(3.0, 4.0), Interval(-3.0, inf))
    def test_88(self):
        self.assertEqual(Interval(1.0, inf) - Interval(3.0, inf), Interval(-math.inf, math.inf))
    def test_89(self):
        self.assertEqual(Interval(1.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')) - Interval(-3.0, 4.0), Interval(-3.0, inf))
    def test_90(self):
        self.assertEqual(Interval(float.fromhex('-0x1.FFFFFFFFFFFFFp1023'), 2.0) - Interval(3.0, 4.0), Interval(-inf, -1.0))
    def test_91(self):
        self.assertEqual(Interval(float.fromhex('-0x1.FFFFFFFFFFFFFp1023'), 2.0) - Interval(float.fromhex('-0x1.FFFFFFFFFFFFFp1023'), 4.0), Interval(-math.inf, math.inf))
    def test_92(self):
        self.assertEqual(Interval(1.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')) - Interval(0.0, 0.0), Interval(1.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')))
    def test_93(self):
        self.assertEqual(Interval(1.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')) - Interval(-0.0, -0.0), Interval(1.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')))
    def test_94(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(-3.0, 4.0), Interval(-4.0, 3.0))
    def test_95(self):
        self.assertEqual(Interval(-0.0, -0.0) - Interval(-3.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023')), Interval(float.fromhex('-0x1.FFFFFFFFFFFFFp1023'), 3.0))
    def test_96(self):
        self.assertEqual(Interval(float.fromhex('0X1.FFFFFFFFFFFFP+0'), float.fromhex('0X1.FFFFFFFFFFFFP+0')) - Interval(float.fromhex('0X1.999999999999AP-4'), float.fromhex('0X1.999999999999AP-4')), Interval(float.fromhex('0X1.E666666666656P+0'), float.fromhex('0X1.E666666666657P+0')))
    def test_97(self):
        self.assertEqual(Interval(float.fromhex('0X1.FFFFFFFFFFFFP+0'), float.fromhex('0X1.FFFFFFFFFFFFP+0')) - Interval(float.fromhex('-0X1.999999999999AP-4'), float.fromhex('-0X1.999999999999AP-4')), Interval(float.fromhex('0X1.0CCCCCCCCCCC4P+1'), float.fromhex('0X1.0CCCCCCCCCCC5P+1')))
    def test_98(self):
        self.assertEqual(Interval(float.fromhex('-0X1.FFFFFFFFFFFFP+0'), float.fromhex('0X1.FFFFFFFFFFFFP+0')) - Interval(float.fromhex('0X1.999999999999AP-4'), float.fromhex('0X1.999999999999AP-4')), Interval(float.fromhex('-0X1.0CCCCCCCCCCC5P+1'), float.fromhex('0X1.E666666666657P+0')))
suite.addTest(TestCase_minimal_sub_test())

class TestCase_minimal_sub_dec_test(unittest.TestCase):
    """minimal_sub_dec_test"""

suite.addTest(TestCase_minimal_sub_dec_test())

class TestCase_minimal_mul_test(unittest.TestCase):
    """minimal_mul_test"""
    def test_105(self):
        self.assertEqual(Interval(ip.nan, ip.nan) * Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_106(self):
        self.assertEqual(Interval(-1.0, 1.0) * Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_107(self):
        self.assertEqual(Interval(ip.nan, ip.nan) * Interval(-1.0, 1.0), Interval(ip.nan, ip.nan))
    def test_108(self):
        self.assertEqual(Interval(ip.nan, ip.nan) * Interval(-math.inf, math.inf), Interval(ip.nan, ip.nan))
    def test_109(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_110(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_111(self):
        self.assertEqual(Interval(ip.nan, ip.nan) * Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_112(self):
        self.assertEqual(Interval(-0.0, -0.0) * Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_113(self):
        self.assertEqual(Interval(ip.nan, ip.nan) * Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_114(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_115(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(-0.0, -0.0), Interval(0.0, 0.0))
    def test_116(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(-5.0, -1.0), Interval(-math.inf, math.inf))
    def test_117(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(-5.0, 3.0), Interval(-math.inf, math.inf))
    def test_118(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(1.0, 3.0), Interval(-math.inf, math.inf))
    def test_119(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(-inf, -1.0), Interval(-math.inf, math.inf))
    def test_120(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_121(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(-5.0, inf), Interval(-math.inf, math.inf))
    def test_122(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(1.0, inf), Interval(-math.inf, math.inf))
    def test_123(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_124(self):
        self.assertEqual(Interval(1.0, inf) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_125(self):
        self.assertEqual(Interval(1.0, inf) * Interval(-0.0, -0.0), Interval(0.0, 0.0))
    def test_126(self):
        self.assertEqual(Interval(1.0, inf) * Interval(-5.0, -1.0), Interval(-inf, -1.0))
    def test_127(self):
        self.assertEqual(Interval(1.0, inf) * Interval(-5.0, 3.0), Interval(-math.inf, math.inf))
    def test_128(self):
        self.assertEqual(Interval(1.0, inf) * Interval(1.0, 3.0), Interval(1.0, inf))
    def test_129(self):
        self.assertEqual(Interval(1.0, inf) * Interval(-inf, -1.0), Interval(-inf, -1.0))
    def test_130(self):
        self.assertEqual(Interval(1.0, inf) * Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_131(self):
        self.assertEqual(Interval(1.0, inf) * Interval(-5.0, inf), Interval(-math.inf, math.inf))
    def test_132(self):
        self.assertEqual(Interval(1.0, inf) * Interval(1.0, inf), Interval(1.0, inf))
    def test_133(self):
        self.assertEqual(Interval(1.0, inf) * Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_134(self):
        self.assertEqual(Interval(-1.0, inf) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_135(self):
        self.assertEqual(Interval(-1.0, inf) * Interval(-0.0, -0.0), Interval(0.0, 0.0))
    def test_136(self):
        self.assertEqual(Interval(-1.0, inf) * Interval(-5.0, -1.0), Interval(-inf, 5.0))
    def test_137(self):
        self.assertEqual(Interval(-1.0, inf) * Interval(-5.0, 3.0), Interval(-math.inf, math.inf))
    def test_138(self):
        self.assertEqual(Interval(-1.0, inf) * Interval(1.0, 3.0), Interval(-3.0, inf))
    def test_139(self):
        self.assertEqual(Interval(-1.0, inf) * Interval(-inf, -1.0), Interval(-math.inf, math.inf))
    def test_140(self):
        self.assertEqual(Interval(-1.0, inf) * Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_141(self):
        self.assertEqual(Interval(-1.0, inf) * Interval(-5.0, inf), Interval(-math.inf, math.inf))
    def test_142(self):
        self.assertEqual(Interval(-1.0, inf) * Interval(1.0, inf), Interval(-math.inf, math.inf))
    def test_143(self):
        self.assertEqual(Interval(-1.0, inf) * Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_144(self):
        self.assertEqual(Interval(-inf, 3.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_145(self):
        self.assertEqual(Interval(-inf, 3.0) * Interval(-0.0, -0.0), Interval(0.0, 0.0))
    def test_146(self):
        self.assertEqual(Interval(-inf, 3.0) * Interval(-5.0, -1.0), Interval(-15.0, inf))
    def test_147(self):
        self.assertEqual(Interval(-inf, 3.0) * Interval(-5.0, 3.0), Interval(-math.inf, math.inf))
    def test_148(self):
        self.assertEqual(Interval(-inf, 3.0) * Interval(1.0, 3.0), Interval(-inf, 9.0))
    def test_149(self):
        self.assertEqual(Interval(-inf, 3.0) * Interval(-inf, -1.0), Interval(-math.inf, math.inf))
    def test_150(self):
        self.assertEqual(Interval(-inf, 3.0) * Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_151(self):
        self.assertEqual(Interval(-inf, 3.0) * Interval(-5.0, inf), Interval(-math.inf, math.inf))
    def test_152(self):
        self.assertEqual(Interval(-inf, 3.0) * Interval(1.0, inf), Interval(-math.inf, math.inf))
    def test_153(self):
        self.assertEqual(Interval(-inf, 3.0) * Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_154(self):
        self.assertEqual(Interval(-inf, -3.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_155(self):
        self.assertEqual(Interval(-inf, -3.0) * Interval(-0.0, -0.0), Interval(0.0, 0.0))
    def test_156(self):
        self.assertEqual(Interval(-inf, -3.0) * Interval(-5.0, -1.0), Interval(3.0, inf))
    def test_157(self):
        self.assertEqual(Interval(-inf, -3.0) * Interval(-5.0, 3.0), Interval(-math.inf, math.inf))
    def test_158(self):
        self.assertEqual(Interval(-inf, -3.0) * Interval(1.0, 3.0), Interval(-inf, -3.0))
    def test_159(self):
        self.assertEqual(Interval(-inf, -3.0) * Interval(-inf, -1.0), Interval(3.0, inf))
    def test_160(self):
        self.assertEqual(Interval(-inf, -3.0) * Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_161(self):
        self.assertEqual(Interval(-inf, -3.0) * Interval(-5.0, inf), Interval(-math.inf, math.inf))
    def test_162(self):
        self.assertEqual(Interval(-inf, -3.0) * Interval(1.0, inf), Interval(-inf, -3.0))
    def test_163(self):
        self.assertEqual(Interval(-inf, -3.0) * Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_164(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_165(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(-0.0, -0.0), Interval(0.0, 0.0))
    def test_166(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(-5.0, -1.0), Interval(0.0, 0.0))
    def test_167(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(-5.0, 3.0), Interval(0.0, 0.0))
    def test_168(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(1.0, 3.0), Interval(0.0, 0.0))
    def test_169(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(-inf, -1.0), Interval(0.0, 0.0))
    def test_170(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(-inf, 3.0), Interval(0.0, 0.0))
    def test_171(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(-5.0, inf), Interval(0.0, 0.0))
    def test_172(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(1.0, inf), Interval(0.0, 0.0))
    def test_173(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(-math.inf, math.inf), Interval(0.0, 0.0))
    def test_174(self):
        self.assertEqual(Interval(-0.0, -0.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_175(self):
        self.assertEqual(Interval(-0.0, -0.0) * Interval(-0.0, -0.0), Interval(0.0, 0.0))
    def test_176(self):
        self.assertEqual(Interval(-0.0, -0.0) * Interval(-5.0, -1.0), Interval(0.0, 0.0))
    def test_177(self):
        self.assertEqual(Interval(-0.0, -0.0) * Interval(-5.0, 3.0), Interval(0.0, 0.0))
    def test_178(self):
        self.assertEqual(Interval(-0.0, -0.0) * Interval(1.0, 3.0), Interval(0.0, 0.0))
    def test_179(self):
        self.assertEqual(Interval(-0.0, -0.0) * Interval(-inf, -1.0), Interval(0.0, 0.0))
    def test_180(self):
        self.assertEqual(Interval(-0.0, -0.0) * Interval(-inf, 3.0), Interval(0.0, 0.0))
    def test_181(self):
        self.assertEqual(Interval(-0.0, -0.0) * Interval(-5.0, inf), Interval(0.0, 0.0))
    def test_182(self):
        self.assertEqual(Interval(-0.0, -0.0) * Interval(1.0, inf), Interval(0.0, 0.0))
    def test_183(self):
        self.assertEqual(Interval(-0.0, -0.0) * Interval(-math.inf, math.inf), Interval(0.0, 0.0))
    def test_184(self):
        self.assertEqual(Interval(1.0, 5.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_185(self):
        self.assertEqual(Interval(1.0, 5.0) * Interval(-0.0, -0.0), Interval(0.0, 0.0))
    def test_186(self):
        self.assertEqual(Interval(1.0, 5.0) * Interval(-5.0, -1.0), Interval(-25.0, -1.0))
    def test_187(self):
        self.assertEqual(Interval(1.0, 5.0) * Interval(-5.0, 3.0), Interval(-25.0, 15.0))
    def test_188(self):
        self.assertEqual(Interval(1.0, 5.0) * Interval(1.0, 3.0), Interval(1.0, 15.0))
    def test_189(self):
        self.assertEqual(Interval(1.0, 5.0) * Interval(-inf, -1.0), Interval(-inf, -1.0))
    def test_190(self):
        self.assertEqual(Interval(1.0, 5.0) * Interval(-inf, 3.0), Interval(-inf, 15.0))
    def test_191(self):
        self.assertEqual(Interval(1.0, 5.0) * Interval(-5.0, inf), Interval(-25.0, inf))
    def test_192(self):
        self.assertEqual(Interval(1.0, 5.0) * Interval(1.0, inf), Interval(1.0, inf))
    def test_193(self):
        self.assertEqual(Interval(1.0, 5.0) * Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_194(self):
        self.assertEqual(Interval(-1.0, 5.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_195(self):
        self.assertEqual(Interval(-1.0, 5.0) * Interval(-0.0, -0.0), Interval(0.0, 0.0))
    def test_196(self):
        self.assertEqual(Interval(-1.0, 5.0) * Interval(-5.0, -1.0), Interval(-25.0, 5.0))
    #min max
    def test_197(self):
        self.assertEqual(Interval(-1.0, 5.0) * Interval(-5.0, 3.0), Interval(-25.0, 15.0))
    def test_198(self):
        self.assertEqual(Interval(-10.0, 2.0) * Interval(-5.0, 3.0), Interval(-30.0, 50.0))
    def test_199(self):
        self.assertEqual(Interval(-1.0, 5.0) * Interval(-1.0, 10.0), Interval(-10.0, 50.0))
    def test_200(self):
        self.assertEqual(Interval(-2.0, 2.0) * Interval(-5.0, 3.0), Interval(-10.0, 10.0))
    #end min max
    def test_201(self):
        self.assertEqual(Interval(-1.0, 5.0) * Interval(1.0, 3.0), Interval(-3.0, 15.0))
    def test_202(self):
        self.assertEqual(Interval(-1.0, 5.0) * Interval(-inf, -1.0), Interval(-math.inf, math.inf))
    def test_203(self):
        self.assertEqual(Interval(-1.0, 5.0) * Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_204(self):
        self.assertEqual(Interval(-1.0, 5.0) * Interval(-5.0, inf), Interval(-math.inf, math.inf))
    def test_205(self):
        self.assertEqual(Interval(-1.0, 5.0) * Interval(1.0, inf), Interval(-math.inf, math.inf))
    def test_206(self):
        self.assertEqual(Interval(-1.0, 5.0) * Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_207(self):
        self.assertEqual(Interval(-10.0, -5.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_208(self):
        self.assertEqual(Interval(-10.0, -5.0) * Interval(-0.0, -0.0), Interval(0.0, 0.0))
    def test_209(self):
        self.assertEqual(Interval(-10.0, -5.0) * Interval(-5.0, -1.0), Interval(5.0, 50.0))
    def test_210(self):
        self.assertEqual(Interval(-10.0, -5.0) * Interval(-5.0, 3.0), Interval(-30.0, 50.0))
    def test_211(self):
        self.assertEqual(Interval(-10.0, -5.0) * Interval(1.0, 3.0), Interval(-30.0, -5.0))
    def test_212(self):
        self.assertEqual(Interval(-10.0, -5.0) * Interval(-inf, -1.0), Interval(5.0, inf))
    def test_213(self):
        self.assertEqual(Interval(-10.0, -5.0) * Interval(-inf, 3.0), Interval(-30.0, inf))
    def test_214(self):
        self.assertEqual(Interval(-10.0, -5.0) * Interval(-5.0, inf), Interval(-inf, 50.0))
    def test_215(self):
        self.assertEqual(Interval(-10.0, -5.0) * Interval(1.0, inf), Interval(-inf, -5.0))
    def test_216(self):
        self.assertEqual(Interval(-10.0, -5.0) * Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_217(self):
        self.assertEqual(Interval(float.fromhex('0X1.999999999999AP-4'), float.fromhex('0X1.FFFFFFFFFFFFP+0')) * Interval(float.fromhex('-0X1.FFFFFFFFFFFFP+0'), inf), Interval(float.fromhex('-0X1.FFFFFFFFFFFE1P+1'), inf))
    def test_218(self):
        self.assertEqual(Interval(float.fromhex('-0X1.999999999999AP-4'), float.fromhex('0X1.FFFFFFFFFFFFP+0')) * Interval(float.fromhex('-0X1.FFFFFFFFFFFFP+0'), float.fromhex('-0X1.999999999999AP-4')), Interval(float.fromhex('-0X1.FFFFFFFFFFFE1P+1'), float.fromhex('0X1.999999999998EP-3')))
    def test_219(self):
        self.assertEqual(Interval(float.fromhex('-0X1.999999999999AP-4'), float.fromhex('0X1.999999999999AP-4')) * Interval(float.fromhex('-0X1.FFFFFFFFFFFFP+0'), float.fromhex('0X1.999999999999AP-4')), Interval(float.fromhex('-0X1.999999999998EP-3'), float.fromhex('0X1.999999999998EP-3')))
    def test_220(self):
        self.assertEqual(Interval(float.fromhex('-0X1.FFFFFFFFFFFFP+0'), float.fromhex('-0X1.999999999999AP-4')) * Interval(float.fromhex('0X1.999999999999AP-4'), float.fromhex('0X1.FFFFFFFFFFFFP+0')), Interval(float.fromhex('-0X1.FFFFFFFFFFFE1P+1'), float.fromhex('-0X1.47AE147AE147BP-7')))
suite.addTest(TestCase_minimal_mul_test())

class TestCase_minimal_mul_dec_test(unittest.TestCase):
    """minimal_mul_dec_test"""

suite.addTest(TestCase_minimal_mul_dec_test())

class TestCase_minimal_div_test(unittest.TestCase):
    """minimal_div_test"""
    def test_227(self):
        self.assertEqual(Interval(ip.nan, ip.nan) / Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_228(self):
        self.assertEqual(Interval(-1.0, 1.0) / Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_229(self):
        self.assertEqual(Interval(ip.nan, ip.nan) / Interval(-1.0, 1.0), Interval(ip.nan, ip.nan))
    def test_230(self):
        self.assertEqual(Interval(ip.nan, ip.nan) / Interval(0.1, 1.0), Interval(ip.nan, ip.nan))
    def test_231(self):
        self.assertEqual(Interval(ip.nan, ip.nan) / Interval(-1.0, -0.1), Interval(ip.nan, ip.nan))
    def test_232(self):
        self.assertEqual(Interval(ip.nan, ip.nan) / Interval(-math.inf, math.inf), Interval(ip.nan, ip.nan))
    def test_233(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_234(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_235(self):
        self.assertEqual(Interval(ip.nan, ip.nan) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_236(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(ip.nan, ip.nan), Interval(ip.nan, ip.nan))
    def test_237(self):
        self.assertEqual(Interval(ip.nan, ip.nan) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_238(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(-5.0, -3.0), Interval(-math.inf, math.inf))
    def test_239(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(3.0, 5.0), Interval(-math.inf, math.inf))
    def test_240(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(-inf, -3.0), Interval(-math.inf, math.inf))
    def test_241(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(3.0, inf), Interval(-math.inf, math.inf))
    def test_242(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_243(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_244(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(-3.0, 0.0), Interval(-math.inf, math.inf))
    def test_245(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(-3.0, -0.0), Interval(-math.inf, math.inf))
    def test_246(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_247(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(0.0, 3.0), Interval(-math.inf, math.inf))
    def test_248(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(-inf, 0.0), Interval(-math.inf, math.inf))
    def test_249(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(-0.0, 3.0), Interval(-math.inf, math.inf))
    def test_250(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(-inf, -0.0), Interval(-math.inf, math.inf))
    def test_251(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_252(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_253(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(0.0, inf), Interval(-math.inf, math.inf))
    def test_254(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(-0.0, inf), Interval(-math.inf, math.inf))
    def test_255(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_256(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(-5.0, -3.0), Interval(3.0, 10.0))
    def test_257(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(3.0, 5.0), Interval(-10.0, -3.0))
    def test_258(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(-inf, -3.0), Interval(0.0, 10.0))
    def test_259(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(3.0, inf), Interval(-10.0, 0.0))
    def test_260(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_261(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(-3.0, 0.0), Interval(5.0, inf))
    def test_262(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_263(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(-3.0, -0.0), Interval(5.0, inf))
    def test_264(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_265(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(0.0, 3.0), Interval(-inf, -5.0))
    def test_266(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(-inf, 0.0), Interval(0.0, inf))
    def test_267(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(-0.0, 3.0), Interval(-inf, -5.0))
    def test_268(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(-inf, -0.0), Interval(0.0, inf))
    def test_269(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_270(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_271(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(0.0, inf), Interval(-inf, 0.0))
    def test_272(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(-0.0, inf), Interval(-inf, 0.0))
    def test_273(self):
        self.assertEqual(Interval(-30.0, -15.0) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_274(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(-5.0, -3.0), Interval(-5.0, 10.0))
    def test_275(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(3.0, 5.0), Interval(-10.0, 5.0))
    def test_276(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(-inf, -3.0), Interval(-5.0, 10.0))
    def test_277(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(3.0, inf), Interval(-10.0, 5.0))
    def test_278(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_279(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_280(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(-3.0, 0.0), Interval(-math.inf, math.inf))
    def test_281(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(-3.0, -0.0), Interval(-math.inf, math.inf))
    def test_282(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_283(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(0.0, 3.0), Interval(-math.inf, math.inf))
    def test_284(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(-inf, 0.0), Interval(-math.inf, math.inf))
    def test_285(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(-0.0, 3.0), Interval(-math.inf, math.inf))
    def test_286(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(-inf, -0.0), Interval(-math.inf, math.inf))
    def test_287(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_288(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_289(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(0.0, inf), Interval(-math.inf, math.inf))
    def test_290(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(-0.0, inf), Interval(-math.inf, math.inf))
    def test_291(self):
        self.assertEqual(Interval(-30.0, 15.0) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_292(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(-5.0, -3.0), Interval(-10.0, -3.0))
    def test_293(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(3.0, 5.0), Interval(3.0, 10.0))
    def test_294(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(-inf, -3.0), Interval(-10.0, 0.0))
    def test_295(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(3.0, inf), Interval(0.0, 10.0))
    def test_296(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_297(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(-3.0, 0.0), Interval(-inf, -5.0))
    def test_298(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_299(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(-3.0, -0.0), Interval(-inf, -5.0))
    def test_300(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_301(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(0.0, 3.0), Interval(5.0, inf))
    def test_302(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(-inf, 0.0), Interval(-inf, 0.0))
    def test_303(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(-0.0, 3.0), Interval(5.0, inf))
    def test_304(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(-inf, -0.0), Interval(-inf, 0.0))
    def test_305(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_306(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_307(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(0.0, inf), Interval(0.0, inf))
    def test_308(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(-0.0, inf), Interval(0.0, inf))
    def test_309(self):
        self.assertEqual(Interval(15.0, 30.0) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_310(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-5.0, -3.0), Interval(0.0, 0.0))
    def test_311(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(3.0, 5.0), Interval(0.0, 0.0))
    def test_312(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-inf, -3.0), Interval(0.0, 0.0))
    def test_313(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(3.0, inf), Interval(0.0, 0.0))
    def test_314(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_315(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-3.0, 0.0), Interval(0.0, 0.0))
    def test_316(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_317(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-3.0, -0.0), Interval(0.0, 0.0))
    def test_318(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-3.0, 3.0), Interval(0.0, 0.0))
    def test_319(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(0.0, 3.0), Interval(0.0, 0.0))
    def test_320(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-inf, 0.0), Interval(0.0, 0.0))
    def test_321(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-0.0, 3.0), Interval(0.0, 0.0))
    def test_322(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-inf, -0.0), Interval(0.0, 0.0))
    def test_323(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-inf, 3.0), Interval(0.0, 0.0))
    def test_324(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-3.0, inf), Interval(0.0, 0.0))
    def test_325(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(0.0, inf), Interval(0.0, 0.0))
    def test_326(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-0.0, inf), Interval(0.0, 0.0))
    def test_327(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-math.inf, math.inf), Interval(0.0, 0.0))
    def test_328(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(-5.0, -3.0), Interval(0.0, 0.0))
    def test_329(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(3.0, 5.0), Interval(0.0, 0.0))
    def test_330(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(-inf, -3.0), Interval(0.0, 0.0))
    def test_331(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(3.0, inf), Interval(0.0, 0.0))
    def test_332(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_333(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(-3.0, 0.0), Interval(0.0, 0.0))
    def test_334(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_335(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(-3.0, -0.0), Interval(0.0, 0.0))
    def test_336(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(-3.0, 3.0), Interval(0.0, 0.0))
    def test_337(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(0.0, 3.0), Interval(0.0, 0.0))
    def test_338(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(-inf, 0.0), Interval(0.0, 0.0))
    def test_339(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(-0.0, 3.0), Interval(0.0, 0.0))
    def test_340(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(-inf, -0.0), Interval(0.0, 0.0))
    def test_341(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(-inf, 3.0), Interval(0.0, 0.0))
    def test_342(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(-3.0, inf), Interval(0.0, 0.0))
    def test_343(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(0.0, inf), Interval(0.0, 0.0))
    def test_344(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(-0.0, inf), Interval(0.0, 0.0))
    def test_345(self):
        self.assertEqual(Interval(-0.0, -0.0) / Interval(-math.inf, math.inf), Interval(0.0, 0.0))
    def test_346(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(-5.0, -3.0), Interval(3.0, inf))
    def test_347(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(3.0, 5.0), Interval(-inf, -3.0))
    def test_348(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(-inf, -3.0), Interval(0.0, inf))
    def test_349(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(3.0, inf), Interval(-inf, 0.0))
    def test_350(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_351(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(-3.0, 0.0), Interval(5.0, inf))
    def test_352(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_353(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(-3.0, -0.0), Interval(5.0, inf))
    def test_354(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_355(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(0.0, 3.0), Interval(-inf, -5.0))
    def test_356(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(-inf, 0.0), Interval(0.0, inf))
    def test_357(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(-0.0, 3.0), Interval(-inf, -5.0))
    def test_358(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(-inf, -0.0), Interval(0.0, inf))
    def test_359(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_360(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_361(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(0.0, inf), Interval(-inf, 0.0))
    def test_362(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(-0.0, inf), Interval(-inf, 0.0))
    def test_363(self):
        self.assertEqual(Interval(-inf, -15.0) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_364(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(-5.0, -3.0), Interval(-5.0, inf))
    def test_365(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(3.0, 5.0), Interval(-inf, 5.0))
    def test_366(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(-inf, -3.0), Interval(-5.0, inf))
    def test_367(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(3.0, inf), Interval(-inf, 5.0))
    def test_368(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_369(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(-3.0, 0.0), Interval(-math.inf, math.inf))
    def test_370(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_371(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(-3.0, -0.0), Interval(-math.inf, math.inf))
    def test_372(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_373(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(0.0, 3.0), Interval(-math.inf, math.inf))
    def test_374(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(-inf, 0.0), Interval(-math.inf, math.inf))
    def test_375(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(-0.0, 3.0), Interval(-math.inf, math.inf))
    def test_376(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(-inf, -0.0), Interval(-math.inf, math.inf))
    def test_377(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_378(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_379(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(0.0, inf), Interval(-math.inf, math.inf))
    def test_380(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(-0.0, inf), Interval(-math.inf, math.inf))
    def test_381(self):
        self.assertEqual(Interval(-inf, 15.0) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_382(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(-5.0, -3.0), Interval(-inf, 5.0))
    def test_383(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(3.0, 5.0), Interval(-5.0, inf))
    def test_384(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(-inf, -3.0), Interval(-inf, 5.0))
    def test_385(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(3.0, inf), Interval(-5.0, inf))
    def test_386(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_387(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(-3.0, 0.0), Interval(-math.inf, math.inf))
    def test_388(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_389(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(-3.0, -0.0), Interval(-math.inf, math.inf))
    def test_390(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_391(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(0.0, 3.0), Interval(-math.inf, math.inf))
    def test_392(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(-inf, 0.0), Interval(-math.inf, math.inf))
    def test_393(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(-0.0, 3.0), Interval(-math.inf, math.inf))
    def test_394(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(-inf, -0.0), Interval(-math.inf, math.inf))
    def test_395(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_396(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_397(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(0.0, inf), Interval(-math.inf, math.inf))
    def test_398(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(-0.0, inf), Interval(-math.inf, math.inf))
    def test_399(self):
        self.assertEqual(Interval(-15.0, inf) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_400(self):
        self.assertEqual(Interval(15.0, inf) / Interval(-5.0, -3.0), Interval(-inf, -3.0))
    def test_401(self):
        self.assertEqual(Interval(15.0, inf) / Interval(3.0, 5.0), Interval(3.0, inf))
    def test_402(self):
        self.assertEqual(Interval(15.0, inf) / Interval(-inf, -3.0), Interval(-inf, 0.0))
    def test_403(self):
        self.assertEqual(Interval(15.0, inf) / Interval(3.0, inf), Interval(0.0, inf))
    def test_404(self):
        self.assertEqual(Interval(15.0, inf) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_405(self):
        self.assertEqual(Interval(15.0, inf) / Interval(-3.0, 0.0), Interval(-inf, -5.0))
    def test_406(self):
        self.assertEqual(Interval(15.0, inf) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_407(self):
        self.assertEqual(Interval(15.0, inf) / Interval(-3.0, -0.0), Interval(-inf, -5.0))
    def test_408(self):
        self.assertEqual(Interval(15.0, inf) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_409(self):
        self.assertEqual(Interval(15.0, inf) / Interval(0.0, 3.0), Interval(5.0, inf))
    def test_410(self):
        self.assertEqual(Interval(15.0, inf) / Interval(-inf, 0.0), Interval(-inf, 0.0))
    def test_411(self):
        self.assertEqual(Interval(15.0, inf) / Interval(-0.0, 3.0), Interval(5.0, inf))
    def test_412(self):
        self.assertEqual(Interval(15.0, inf) / Interval(-inf, -0.0), Interval(-inf, 0.0))
    def test_413(self):
        self.assertEqual(Interval(15.0, inf) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_414(self):
        self.assertEqual(Interval(15.0, inf) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_415(self):
        self.assertEqual(Interval(15.0, inf) / Interval(0.0, inf), Interval(0.0, inf))
    def test_416(self):
        self.assertEqual(Interval(15.0, inf) / Interval(-0.0, inf), Interval(0.0, inf))
    def test_417(self):
        self.assertEqual(Interval(15.0, inf) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_418(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(-5.0, -3.0), Interval(0.0, 10.0))
    def test_419(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(3.0, 5.0), Interval(-10.0, 0.0))
    def test_420(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(-inf, -3.0), Interval(0.0, 10.0))
    def test_421(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(3.0, inf), Interval(-10.0, 0.0))
    def test_422(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_423(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(-3.0, 0.0), Interval(0.0, inf))
    def test_424(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_425(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(-3.0, -0.0), Interval(0.0, inf))
    def test_426(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_427(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(0.0, 3.0), Interval(-inf, 0.0))
    def test_428(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(-inf, 0.0), Interval(0.0, inf))
    def test_429(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(-0.0, 3.0), Interval(-inf, 0.0))
    def test_430(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(-inf, -0.0), Interval(0.0, inf))
    def test_431(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_432(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_433(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(0.0, inf), Interval(-inf, 0.0))
    def test_434(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(-0.0, inf), Interval(-inf, 0.0))
    def test_435(self):
        self.assertEqual(Interval(-30.0, 0.0) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_436(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(-5.0, -3.0), Interval(0.0, 10.0))
    def test_437(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(3.0, 5.0), Interval(-10.0, 0.0))
    def test_438(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(-inf, -3.0), Interval(0.0, 10.0))
    def test_439(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(3.0, inf), Interval(-10.0, 0.0))
    def test_440(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_441(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(-3.0, 0.0), Interval(0.0, inf))
    def test_442(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_443(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(-3.0, -0.0), Interval(0.0, inf))
    def test_444(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_445(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(0.0, 3.0), Interval(-inf, 0.0))
    def test_446(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(-inf, 0.0), Interval(0.0, inf))
    def test_447(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(-0.0, 3.0), Interval(-inf, 0.0))
    def test_448(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(-inf, -0.0), Interval(0.0, inf))
    def test_449(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_450(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_451(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(0.0, inf), Interval(-inf, 0.0))
    def test_452(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(-0.0, inf), Interval(-inf, 0.0))
    def test_453(self):
        self.assertEqual(Interval(-30.0, -0.0) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_454(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(-5.0, -3.0), Interval(-10.0, 0.0))
    def test_455(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(3.0, 5.0), Interval(0.0, 10.0))
    def test_456(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(-inf, -3.0), Interval(-10.0, 0.0))
    def test_457(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(3.0, inf), Interval(0.0, 10.0))
    def test_458(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_459(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(-3.0, 0.0), Interval(-inf, 0.0))
    def test_460(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_461(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(-3.0, -0.0), Interval(-inf, 0.0))
    def test_462(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_463(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(0.0, 3.0), Interval(0.0, inf))
    def test_464(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(-inf, 0.0), Interval(-inf, 0.0))
    def test_465(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(-0.0, 3.0), Interval(0.0, inf))
    def test_466(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(-inf, -0.0), Interval(-inf, 0.0))
    def test_467(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_468(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_469(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(0.0, inf), Interval(0.0, inf))
    def test_470(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(-0.0, inf), Interval(0.0, inf))
    def test_471(self):
        self.assertEqual(Interval(0.0, 30.0) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_472(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(-5.0, -3.0), Interval(-10.0, 0.0))
    def test_473(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(3.0, 5.0), Interval(0.0, 10.0))
    def test_474(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(-inf, -3.0), Interval(-10.0, 0.0))
    def test_475(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(3.0, inf), Interval(0.0, 10.0))
    def test_476(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_477(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(-3.0, 0.0), Interval(-inf, 0.0))
    def test_478(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_479(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(-3.0, -0.0), Interval(-inf, 0.0))
    def test_480(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_481(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(0.0, 3.0), Interval(0.0, inf))
    def test_482(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(-inf, 0.0), Interval(-inf, 0.0))
    def test_483(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(-0.0, 3.0), Interval(0.0, inf))
    def test_484(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(-inf, -0.0), Interval(-inf, 0.0))
    def test_485(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_486(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_487(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(0.0, inf), Interval(0.0, inf))
    def test_488(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(-0.0, inf), Interval(0.0, inf))
    def test_489(self):
        self.assertEqual(Interval(-0.0, 30.0) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_490(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(-5.0, -3.0), Interval(0.0, inf))
    def test_491(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(3.0, 5.0), Interval(-inf, 0.0))
    def test_492(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(-inf, -3.0), Interval(0.0, inf))
    def test_493(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(3.0, inf), Interval(-inf, 0.0))
    def test_494(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_495(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(-3.0, 0.0), Interval(0.0, inf))
    def test_496(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_497(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(-3.0, -0.0), Interval(0.0, inf))
    def test_498(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_499(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(0.0, 3.0), Interval(-inf, 0.0))
    def test_500(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(-inf, 0.0), Interval(0.0, inf))
    def test_501(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(-0.0, 3.0), Interval(-inf, 0.0))
    def test_502(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(-inf, -0.0), Interval(0.0, inf))
    def test_503(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_504(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_505(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(0.0, inf), Interval(-inf, 0.0))
    def test_506(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(-0.0, inf), Interval(-inf, 0.0))
    def test_507(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_508(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(-5.0, -3.0), Interval(0.0, inf))
    def test_509(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(3.0, 5.0), Interval(-inf, 0.0))
    def test_510(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(-inf, -3.0), Interval(0.0, inf))
    def test_511(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(3.0, inf), Interval(-inf, 0.0))
    def test_512(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_513(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(-3.0, 0.0), Interval(0.0, inf))
    def test_514(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_515(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(-3.0, -0.0), Interval(0.0, inf))
    def test_516(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_517(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(0.0, 3.0), Interval(-inf, 0.0))
    def test_518(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(-inf, 0.0), Interval(0.0, inf))
    def test_519(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(-0.0, 3.0), Interval(-inf, 0.0))
    def test_520(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(-inf, -0.0), Interval(0.0, inf))
    def test_521(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_522(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_523(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(0.0, inf), Interval(-inf, 0.0))
    def test_524(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(-0.0, inf), Interval(-inf, 0.0))
    def test_525(self):
        self.assertEqual(Interval(-inf, -0.0) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_526(self):
        self.assertEqual(Interval(0.0, inf) / Interval(-5.0, -3.0), Interval(-inf, 0.0))
    def test_527(self):
        self.assertEqual(Interval(0.0, inf) / Interval(3.0, 5.0), Interval(0.0, inf))
    def test_528(self):
        self.assertEqual(Interval(0.0, inf) / Interval(-inf, -3.0), Interval(-inf, 0.0))
    def test_529(self):
        self.assertEqual(Interval(0.0, inf) / Interval(3.0, inf), Interval(0.0, inf))
    def test_530(self):
        self.assertEqual(Interval(0.0, inf) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_531(self):
        self.assertEqual(Interval(0.0, inf) / Interval(-3.0, 0.0), Interval(-inf, 0.0))
    def test_532(self):
        self.assertEqual(Interval(0.0, inf) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_533(self):
        self.assertEqual(Interval(0.0, inf) / Interval(-3.0, -0.0), Interval(-inf, 0.0))
    def test_534(self):
        self.assertEqual(Interval(0.0, inf) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_535(self):
        self.assertEqual(Interval(0.0, inf) / Interval(0.0, 3.0), Interval(0.0, inf))
    def test_536(self):
        self.assertEqual(Interval(0.0, inf) / Interval(-inf, 0.0), Interval(-inf, 0.0))
    def test_537(self):
        self.assertEqual(Interval(0.0, inf) / Interval(-0.0, 3.0), Interval(0.0, inf))
    def test_538(self):
        self.assertEqual(Interval(0.0, inf) / Interval(-inf, -0.0), Interval(-inf, 0.0))
    def test_539(self):
        self.assertEqual(Interval(0.0, inf) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_540(self):
        self.assertEqual(Interval(0.0, inf) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_541(self):
        self.assertEqual(Interval(0.0, inf) / Interval(0.0, inf), Interval(0.0, inf))
    def test_542(self):
        self.assertEqual(Interval(0.0, inf) / Interval(-0.0, inf), Interval(0.0, inf))
    def test_543(self):
        self.assertEqual(Interval(0.0, inf) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_544(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(-5.0, -3.0), Interval(-inf, 0.0))
    def test_545(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(3.0, 5.0), Interval(0.0, inf))
    def test_546(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(-inf, -3.0), Interval(-inf, 0.0))
    def test_547(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(3.0, inf), Interval(0.0, inf))
    def test_548(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_549(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(-3.0, 0.0), Interval(-inf, 0.0))
    def test_550(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(-0.0, -0.0), Interval(ip.nan, ip.nan))
    def test_551(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(-3.0, -0.0), Interval(-inf, 0.0))
    def test_552(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(-3.0, 3.0), Interval(-math.inf, math.inf))
    def test_553(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(0.0, 3.0), Interval(0.0, inf))
    def test_554(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(-inf, 0.0), Interval(-inf, 0.0))
    def test_555(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(-0.0, 3.0), Interval(0.0, inf))
    def test_556(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(-inf, -0.0), Interval(-inf, 0.0))
    def test_557(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(-inf, 3.0), Interval(-math.inf, math.inf))
    def test_558(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(-3.0, inf), Interval(-math.inf, math.inf))
    def test_559(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(0.0, inf), Interval(0.0, inf))
    def test_560(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(-0.0, inf), Interval(0.0, inf))
    def test_561(self):
        self.assertEqual(Interval(-0.0, inf) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_562(self):
        self.assertEqual(Interval(-2.0, -1.0) / Interval(-10.0, -3.0), Interval(float.fromhex('0X1.9999999999999P-4'), float.fromhex('0X1.5555555555556P-1')))
    def test_563(self):
        self.assertEqual(Interval(-2.0, -1.0) / Interval(0.0, 10.0), Interval(-inf, float.fromhex('-0X1.9999999999999P-4')))
    def test_564(self):
        self.assertEqual(Interval(-2.0, -1.0) / Interval(-0.0, 10.0), Interval(-inf, float.fromhex('-0X1.9999999999999P-4')))
    def test_565(self):
        self.assertEqual(Interval(-1.0, 2.0) / Interval(10.0, inf), Interval(float.fromhex('-0X1.999999999999AP-4'), float.fromhex('0X1.999999999999AP-3')))
    def test_566(self):
        self.assertEqual(Interval(1.0, 3.0) / Interval(-inf, -10.0), Interval(float.fromhex('-0X1.3333333333334P-2'), 0.0))
    def test_567(self):
        self.assertEqual(Interval(-inf, -1.0) / Interval(1.0, 3.0), Interval(-inf, float.fromhex('-0X1.5555555555555P-2')))
suite.addTest(TestCase_minimal_div_test())

class TestCase_minimal_div_dec_test(unittest.TestCase):
    """minimal_div_dec_test"""

suite.addTest(TestCase_minimal_div_dec_test())

class TestCase_minimal_recip_test(unittest.TestCase):
    """minimal_recip_test"""

suite.addTest(TestCase_minimal_recip_test())

class TestCase_minimal_recip_dec_test(unittest.TestCase):
    """minimal_recip_dec_test"""

suite.addTest(TestCase_minimal_recip_dec_test())

class TestCase_minimal_sqr_test(unittest.TestCase):
    """minimal_sqr_test"""

suite.addTest(TestCase_minimal_sqr_test())

class TestCase_minimal_sqr_dec_test(unittest.TestCase):
    """minimal_sqr_dec_test"""

suite.addTest(TestCase_minimal_sqr_dec_test())

class TestCase_minimal_sqrt_test(unittest.TestCase):
    """minimal_sqrt_test"""
    def test_0158_minimal_sqrt_test(self):
        self.assertEqual(ip.sqrt(Interval(math.nan, math.nan)), Interval(math.nan, math.nan))
    def test_0159_minimal_sqrt_test(self):
        self.assertEqual(ip.sqrt(Interval(-math.inf, math.inf)), Interval(0.0, math.inf))
    def test_0160_minimal_sqrt_test(self):
        self.assertEqual(ip.sqrt(Interval(-math.inf, float.fromhex('-0x0.0000000000001p-1022'))), Interval(math.nan, math.nan))
    def test_0161_minimal_sqrt_test(self):
        self.assertEqual(ip.sqrt(Interval(-1.0, 1.0)), Interval(0.0, 1.0))
    def test_0162_minimal_sqrt_test(self):
        self.assertEqual(ip.sqrt(Interval(0.0, 1.0)), Interval(0.0, 1.0))
    def test_0163_minimal_sqrt_test(self):
        self.assertEqual(ip.sqrt(Interval(-0.0, 1.0)), Interval(0.0, 1.0))
    def test_0164_minimal_sqrt_test(self):
        self.assertEqual(ip.sqrt(Interval(-5.0, 25.0)), Interval(0.0, 5.0))
    def test_0165_minimal_sqrt_test(self):
        self.assertEqual(ip.sqrt(Interval(0.0, 25.0)), Interval(0.0, 5.0))
    def test_0166_minimal_sqrt_test(self):
        self.assertEqual(ip.sqrt(Interval(-0.0, 25.0)), Interval(0.0, 5.0))
    def test_0167_minimal_sqrt_test(self):
        self.assertEqual(ip.sqrt(Interval(-5.0, math.inf)), Interval(0.0, math.inf))
    def test_0168_minimal_sqrt_test(self):
        self.assertEqual(ip.sqrt(Interval(float.fromhex('0X1.999999999999AP-4'), float.fromhex('0X1.999999999999AP-4'))), Interval(float.fromhex('0X1.43D136248490FP-2'), float.fromhex('0X1.43D136248491P-2')))
    def test_0169_minimal_sqrt_test(self):
        self.assertEqual(ip.sqrt(Interval(float.fromhex('-0X1.FFFFFFFFFFFFP+0'), float.fromhex('0X1.999999999999AP-4'))), Interval(0.0, float.fromhex('0X1.43D136248491P-2')))
    def test_0170_minimal_sqrt_test(self):
        self.assertEqual(ip.sqrt(Interval(float.fromhex('0X1.999999999999AP-4'), float.fromhex('0X1.FFFFFFFFFFFFP+0'))), Interval(float.fromhex('0X1.43D136248490FP-2'), float.fromhex('0X1.6A09E667F3BC7P+0')))
    def test_0171_minimal_sqrt_dec_test(self):
        self.assertEqual(ip.sqrt(Interval(1.0, 4.0)), Interval(1.0, 2.0))

suite.addTest(TestCase_minimal_sqrt_test())

class TestCase_minimal_sqrt_dec_test(unittest.TestCase):
    """minimal_sqrt_dec_test"""

suite.addTest(TestCase_minimal_sqrt_dec_test())

class TestCase_minimal_fma_test(unittest.TestCase):
    """minimal_fma_test"""

suite.addTest(TestCase_minimal_fma_test())

class TestCase_minimal_fma_dec_test(unittest.TestCase):
    """minimal_fma_dec_test"""

suite.addTest(TestCase_minimal_fma_dec_test())

class TestCase_minimal_pown_test(unittest.TestCase):
    """minimal_pown_test"""

suite.addTest(TestCase_minimal_pown_test())

class TestCase_minimal_pown_dec_test(unittest.TestCase):
    """minimal_pown_dec_test"""

suite.addTest(TestCase_minimal_pown_dec_test())

class TestCase_minimal_pow_test(unittest.TestCase):
    """minimal_pow_test"""

suite.addTest(TestCase_minimal_pow_test())

class TestCase_minimal_pow_dec_test(unittest.TestCase):
    """minimal_pow_dec_test"""

suite.addTest(TestCase_minimal_pow_dec_test())

class TestCase_minimal_exp_test(unittest.TestCase):
    """minimal_exp_test"""
    def test_0065_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(math.nan, math.nan)), Interval(math.nan, math.nan))
    def test_0066_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(-math.inf, 0.0)), Interval(0.0, 1.0))
    def test_0067_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(-math.inf, -0.0)), Interval(0.0, 1.0))
    def test_0068_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(0.0, math.inf)), Interval(1.0, math.inf))
    def test_0069_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(-0.0, math.inf)), Interval(1.0, math.inf))
    def test_0070_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(-math.inf, math.inf)), Interval(0.0, math.inf))
    def test_0071_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(-math.inf, float.fromhex('0X1.62E42FEFA39FP+9'))), Interval(0.0, math.inf))
    def test_0072_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(float.fromhex('0X1.62E42FEFA39FP+9'), float.fromhex('0X1.62E42FEFA39FP+9'))), Interval(float.fromhex('0X1.FFFFFFFFFFFFFP+1023'), math.inf))
    def test_0073_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(0.0, float.fromhex('0X1.62E42FEFA39EP+9'))), Interval(1.0, float.fromhex('0X1.FFFFFFFFFC32BP+1023')))
    def test_0074_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(-0.0, float.fromhex('0X1.62E42FEFA39EP+9'))), Interval(1.0, float.fromhex('0X1.FFFFFFFFFC32BP+1023')))
    def test_0075_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(float.fromhex('-0X1.6232BDD7ABCD3P+9'), float.fromhex('0X1.62E42FEFA39EP+9'))), Interval(float.fromhex('0X0.FFFFFFFFFFE7BP-1022'), float.fromhex('0X1.FFFFFFFFFC32BP+1023')))
    def test_0076_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(float.fromhex('-0X1.6232BDD7ABCD3P+8'), float.fromhex('0X1.62E42FEFA39EP+9'))), Interval(float.fromhex('0X1.FFFFFFFFFFE7BP-512'), float.fromhex('0X1.FFFFFFFFFC32BP+1023')))
    def test_0077_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(float.fromhex('-0X1.6232BDD7ABCD3P+8'), 0.0)), Interval(float.fromhex('0X1.FFFFFFFFFFE7BP-512'), 1.0))
    def test_0078_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(float.fromhex('-0X1.6232BDD7ABCD3P+8'), -0.0)), Interval(float.fromhex('0X1.FFFFFFFFFFE7BP-512'), 1.0))
    def test_0079_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(float.fromhex('-0X1.6232BDD7ABCD3P+8'), 1.0)), Interval(float.fromhex('0X1.FFFFFFFFFFE7BP-512'), float.fromhex('0X1.5BF0A8B14576AP+1')))
    def test_0080_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(1.0, 5.0)), Interval(float.fromhex('0X1.5BF0A8B145769P+1'), float.fromhex('0X1.28D389970339P+7')))
    def test_0081_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(float.fromhex('-0X1.A934F0979A372P+1'), float.fromhex('0X1.CEAECFEA8085AP+0'))), Interval(float.fromhex('0X1.2797F0A337A5FP-5'), float.fromhex('0X1.86091CC9095C5P+2')))
    def test_0082_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(float.fromhex('0X1.87F42B972949CP-1'), float.fromhex('0X1.8B55484710029P+6'))), Interval(float.fromhex('0X1.1337E9E45812AP+1'), float.fromhex('0X1.805A5C88021B6P+142')))
    def test_0083_minimal_exp_test(self):
        self.assertEqual(ip.exp(Interval(float.fromhex('0X1.78025C8B3FD39P+3'), float.fromhex('0X1.9FD8EEF3FA79BP+4'))), Interval(float.fromhex('0X1.EF461A783114CP+16'), float.fromhex('0X1.691D36C6B008CP+37')))

suite.addTest(TestCase_minimal_exp_test())

class TestCase_minimal_exp_dec_test(unittest.TestCase):
    """minimal_exp_dec_test"""

suite.addTest(TestCase_minimal_exp_dec_test())

class TestCase_minimal_exp2_test(unittest.TestCase):
    """minimal_exp2_test"""

suite.addTest(TestCase_minimal_exp2_test())

class TestCase_minimal_exp2_dec_test(unittest.TestCase):
    """minimal_exp2_dec_test"""

suite.addTest(TestCase_minimal_exp2_dec_test())

class TestCase_minimal_exp10_test(unittest.TestCase):
    """minimal_exp10_test"""

suite.addTest(TestCase_minimal_exp10_test())

class TestCase_minimal_exp10_dec_test(unittest.TestCase):
    """minimal_exp10_dec_test"""

suite.addTest(TestCase_minimal_exp10_dec_test())

class TestCase_minimal_log_test(unittest.TestCase):
    """minimal_log_test"""
    def test_0084_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(math.nan, math.nan)), Interval(math.nan, math.nan))
    def test_0085_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(-math.inf, 0.0)), Interval(math.nan, math.nan))
    def test_0086_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(-math.inf, -0.0)), Interval(math.nan, math.nan))
    def test_0087_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(0.0, 1.0)), Interval(-math.inf, 0.0))
    def test_0088_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(-0.0, 1.0)), Interval(-math.inf, 0.0))
    def test_0089_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(1.0, math.inf)), Interval(0.0, math.inf))
    def test_0090_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(0.0, math.inf)), Interval(-math.inf, math.inf))
    def test_0091_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(-0.0, math.inf)), Interval(-math.inf, math.inf))
    def test_0092_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(-math.inf, math.inf)), Interval(-math.inf, math.inf))
    def test_0093_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(0.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023'))), Interval(-math.inf, float.fromhex('0X1.62E42FEFA39FP+9')))
    def test_0094_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(-0.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023'))), Interval(-math.inf, float.fromhex('0X1.62E42FEFA39FP+9')))
    def test_0095_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(1.0, float.fromhex('0x1.FFFFFFFFFFFFFp1023'))), Interval(0.0, float.fromhex('0X1.62E42FEFA39FP+9')))
    def test_0096_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(float.fromhex('0x0.0000000000001p-1022'), float.fromhex('0x1.FFFFFFFFFFFFFp1023'))), Interval(float.fromhex('-0x1.74385446D71C4p9'), float.fromhex('+0x1.62E42FEFA39Fp9')))
    def test_0097_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(float.fromhex('0x0.0000000000001p-1022'), 1.0)), Interval(float.fromhex('-0x1.74385446D71C4p9'), 0.0))
    def test_0098_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(float.fromhex('0X1.5BF0A8B145769P+1'), float.fromhex('0X1.5BF0A8B145769P+1'))), Interval(float.fromhex('0X1.FFFFFFFFFFFFFP-1'), float.fromhex('0X1P+0')))
    def test_0099_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(float.fromhex('0X1.5BF0A8B14576AP+1'), float.fromhex('0X1.5BF0A8B14576AP+1'))), Interval(float.fromhex('0X1P+0'), float.fromhex('0X1.0000000000001P+0')))
    def test_0100_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(float.fromhex('0x0.0000000000001p-1022'), float.fromhex('0X1.5BF0A8B14576AP+1'))), Interval(float.fromhex('-0x1.74385446D71C4p9'), float.fromhex('0X1.0000000000001P+0')))
    def test_0101_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(float.fromhex('0X1.5BF0A8B145769P+1'), 32.0)), Interval(float.fromhex('0X1.FFFFFFFFFFFFFP-1'), float.fromhex('0X1.BB9D3BEB8C86CP+1')))
    def test_0102_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(float.fromhex('0X1.999999999999AP-4'), float.fromhex('0X1.CP+1'))), Interval(float.fromhex('-0X1.26BB1BBB55516P+1'), float.fromhex('0X1.40B512EB53D6P+0')))
    def test_0103_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(float.fromhex('0X1.B333333333333P+0'), float.fromhex('0X1.C81FD88228B2FP+98'))), Interval(float.fromhex('0X1.0FAE81914A99P-1'), float.fromhex('0X1.120627F6AE7F1P+6')))
    def test_0104_minimal_log_test(self):
        self.assertEqual(ip.log(Interval(float.fromhex('0X1.AEA0000721861P+11'), float.fromhex('0X1.FCA055555554CP+25'))), Interval(float.fromhex('0X1.04A1363DB1E63P+3'), float.fromhex('0X1.203E52C0256B5P+4')))
    def test_0105_minimal_log_dec_test(self):
        self.assertEqual(ip.log(Interval(float.fromhex('0x0.0000000000001p-1022'), float.fromhex('0x1.FFFFFFFFFFFFFp1023'))), Interval(float.fromhex('-0x1.74385446D71C4p9'), float.fromhex('0X1.62E42FEFA39FP+9')))

suite.addTest(TestCase_minimal_log_test())

class TestCase_minimal_log_dec_test(unittest.TestCase):
    """minimal_log_dec_test"""

suite.addTest(TestCase_minimal_log_dec_test())

class TestCase_minimal_log2_test(unittest.TestCase):
    """minimal_log2_test"""

suite.addTest(TestCase_minimal_log2_test())

class TestCase_minimal_log2_dec_test(unittest.TestCase):
    """minimal_log2_dec_test"""

suite.addTest(TestCase_minimal_log2_dec_test())

class TestCase_minimal_log10_test(unittest.TestCase):
    """minimal_log10_test"""

suite.addTest(TestCase_minimal_log10_test())

class TestCase_minimal_log10_dec_test(unittest.TestCase):
    """minimal_log10_dec_test"""

suite.addTest(TestCase_minimal_log10_dec_test())

class TestCase_minimal_sin_test(unittest.TestCase):
    """minimal_sin_test"""
    def test_0106_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(math.nan, math.nan)), Interval(math.nan, math.nan))
    def test_0107_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(0.0, math.inf)), Interval(-1.0, 1.0))
    def test_0108_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(-0.0, math.inf)), Interval(-1.0, 1.0))
    def test_0109_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(-math.inf, 0.0)), Interval(-1.0, 1.0))
    def test_0110_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(-math.inf, -0.0)), Interval(-1.0, 1.0))
    def test_0111_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(-math.inf, math.inf)), Interval(-1.0, 1.0))
    def test_0112_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(0.0, 0.0)), Interval(0.0, 0.0))
    def test_0113_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(-0.0, -0.0)), Interval(0.0, 0.0))
    def test_0114_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('0X1.921FB54442D18P+0'), float.fromhex('0X1.921FB54442D18P+0'))), Interval(float.fromhex('0X1.FFFFFFFFFFFFFP-1'), float.fromhex('0X1P+0')))
    def test_0115_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('0X1.921FB54442D19P+0'), float.fromhex('0X1.921FB54442D19P+0'))), Interval(float.fromhex('0X1.FFFFFFFFFFFFFP-1'), float.fromhex('0X1P+0')))
    def test_0116_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('0X1.921FB54442D18P+0'), float.fromhex('0X1.921FB54442D19P+0'))), Interval(float.fromhex('0X1.FFFFFFFFFFFFFP-1'), float.fromhex('0X1P+0')))
    def test_0117_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(0.0, float.fromhex('0X1.921FB54442D18P+0'))), Interval(0.0, float.fromhex('0X1P+0')))
    def test_0118_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(-0.0, float.fromhex('0X1.921FB54442D18P+0'))), Interval(0.0, float.fromhex('0X1P+0')))
    def test_0119_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(0.0, float.fromhex('0X1.921FB54442D19P+0'))), Interval(0.0, float.fromhex('0X1P+0')))
    def test_0120_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(-0.0, float.fromhex('0X1.921FB54442D19P+0'))), Interval(0.0, float.fromhex('0X1P+0')))
    def test_0121_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('0X1.921FB54442D18P+1'), float.fromhex('0X1.921FB54442D18P+1'))), Interval(float.fromhex('0X1.1A62633145C06P-53'), float.fromhex('0X1.1A62633145C07P-53')))
    def test_0122_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('0X1.921FB54442D19P+1'), float.fromhex('0X1.921FB54442D19P+1'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-52'), float.fromhex('-0X1.72CECE675D1FCP-52')))
    def test_0123_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('0X1.921FB54442D18P+1'), float.fromhex('0X1.921FB54442D19P+1'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-52'), float.fromhex('0X1.1A62633145C07P-53')))
    def test_0124_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(0.0, float.fromhex('0X1.921FB54442D18P+1'))), Interval(0.0, 1.0))
    def test_0125_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(-0.0, float.fromhex('0X1.921FB54442D18P+1'))), Interval(0.0, 1.0))
    def test_0126_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(0.0, float.fromhex('0X1.921FB54442D19P+1'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-52'), 1.0))
    def test_0127_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(-0.0, float.fromhex('0X1.921FB54442D19P+1'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-52'), 1.0))
    def test_0128_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('0X1.921FB54442D18P+0'), float.fromhex('0X1.921FB54442D18P+1'))), Interval(float.fromhex('0X1.1A62633145C06P-53'), float.fromhex('0X1P+0')))
    def test_0129_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('0X1.921FB54442D18P+0'), float.fromhex('0X1.921FB54442D19P+1'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-52'), float.fromhex('0X1P+0')))
    def test_0130_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('0X1.921FB54442D19P+0'), float.fromhex('0X1.921FB54442D18P+1'))), Interval(float.fromhex('0X1.1A62633145C06P-53'), float.fromhex('0X1P+0')))
    def test_0131_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('0X1.921FB54442D19P+0'), float.fromhex('0X1.921FB54442D19P+1'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-52'), float.fromhex('0X1P+0')))
    def test_0132_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D18P+0'), float.fromhex('-0X1.921FB54442D18P+0'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('-0X1.FFFFFFFFFFFFFP-1')))
    def test_0133_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D19P+0'), float.fromhex('-0X1.921FB54442D19P+0'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('-0X1.FFFFFFFFFFFFFP-1')))
    def test_0134_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D19P+0'), float.fromhex('-0X1.921FB54442D18P+0'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('-0X1.FFFFFFFFFFFFFP-1')))
    def test_0135_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D18P+0'), 0.0)), Interval(float.fromhex('-0X1P+0'), 0.0))
    def test_0136_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D18P+0'), -0.0)), Interval(float.fromhex('-0X1P+0'), 0.0))
    def test_0137_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D19P+0'), 0.0)), Interval(float.fromhex('-0X1P+0'), 0.0))
    def test_0138_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D19P+0'), -0.0)), Interval(float.fromhex('-0X1P+0'), 0.0))
    def test_0139_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D18P+1'), float.fromhex('-0X1.921FB54442D18P+1'))), Interval(float.fromhex('-0X1.1A62633145C07P-53'), float.fromhex('-0X1.1A62633145C06P-53')))
    def test_0140_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D19P+1'), float.fromhex('-0X1.921FB54442D19P+1'))), Interval(float.fromhex('0X1.72CECE675D1FCP-52'), float.fromhex('0X1.72CECE675D1FDP-52')))
    def test_0141_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D19P+1'), float.fromhex('-0X1.921FB54442D18P+1'))), Interval(float.fromhex('-0X1.1A62633145C07P-53'), float.fromhex('0X1.72CECE675D1FDP-52')))
    def test_0142_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D18P+1'), 0.0)), Interval(-1.0, 0.0))
    def test_0143_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D18P+1'), -0.0)), Interval(-1.0, 0.0))
    def test_0144_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D19P+1'), 0.0)), Interval(-1.0, float.fromhex('0X1.72CECE675D1FDP-52')))
    def test_0145_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D19P+1'), -0.0)), Interval(-1.0, float.fromhex('0X1.72CECE675D1FDP-52')))
    def test_0146_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D18P+1'), float.fromhex('-0X1.921FB54442D18P+0'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('-0X1.1A62633145C06P-53')))
    def test_0147_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D19P+1'), float.fromhex('-0X1.921FB54442D18P+0'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('0X1.72CECE675D1FDP-52')))
    def test_0148_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D18P+1'), float.fromhex('-0X1.921FB54442D19P+0'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('-0X1.1A62633145C06P-53')))
    def test_0149_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D19P+1'), float.fromhex('-0X1.921FB54442D19P+0'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('0X1.72CECE675D1FDP-52')))
    def test_0150_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D18P+0'), float.fromhex('0X1.921FB54442D18P+0'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('0X1P+0')))
    def test_0151_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D18P+0'), float.fromhex('0X1.921FB54442D19P+0'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('0X1P+0')))
    def test_0152_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D19P+0'), float.fromhex('0X1.921FB54442D18P+0'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('0X1P+0')))
    def test_0153_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(float.fromhex('-0X1.921FB54442D19P+0'), float.fromhex('0X1.921FB54442D19P+0'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('0X1P+0')))
    def test_0154_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(-0.7, 0.1)), Interval(float.fromhex('-0X1.49D6E694619B9P-1'), float.fromhex('0X1.98EAECB8BCB2DP-4')))
    def test_0155_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(1.0, 2.0)), Interval(float.fromhex('0X1.AED548F090CEEP-1'), 1.0))
    def test_0156_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(-3.2, -2.9)), Interval(float.fromhex('-0X1.E9FB8D64830E3P-3'), float.fromhex('0X1.DE33739E82D33P-5')))
    def test_0157_minimal_sin_test(self):
        self.assertEqual(ip.sin(Interval(2.0, 3.0)), Interval(float.fromhex('0X1.210386DB6D55BP-3'), float.fromhex('0X1.D18F6EAD1B446P-1')))

suite.addTest(TestCase_minimal_sin_test())

class TestCase_minimal_sin_dec_test(unittest.TestCase):
    """minimal_sin_dec_test"""

suite.addTest(TestCase_minimal_sin_dec_test())

class TestCase_minimal_cos_test(unittest.TestCase):
    """minimal_cos_test"""
    def test_0013_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(math.nan, math.nan)), Interval(math.nan, math.nan))
    def test_0014_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(0.0, math.inf)), Interval(-1.0, 1.0))
    def test_0015_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(-0.0, math.inf)), Interval(-1.0, 1.0))
    def test_0016_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(-math.inf, 0.0)), Interval(-1.0, 1.0))
    def test_0017_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(-math.inf, -0.0)), Interval(-1.0, 1.0))
    def test_0018_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(-math.inf, math.inf)), Interval(-1.0, 1.0))
    def test_0019_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(0.0, 0.0)), Interval(1.0, 1.0))
    def test_0020_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(-0.0, -0.0)), Interval(1.0, 1.0))
    def test_0021_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('0X1.921FB54442D18P+0'), float.fromhex('0X1.921FB54442D18P+0'))), Interval(float.fromhex('0X1.1A62633145C06P-54'), float.fromhex('0X1.1A62633145C07P-54')))
    def test_0022_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('0X1.921FB54442D19P+0'), float.fromhex('0X1.921FB54442D19P+0'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-53'), float.fromhex('-0X1.72CECE675D1FCP-53')))
    def test_0023_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('0X1.921FB54442D18P+0'), float.fromhex('0X1.921FB54442D19P+0'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-53'), float.fromhex('0X1.1A62633145C07P-54')))
    def test_0024_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(0.0, float.fromhex('0X1.921FB54442D18P+0'))), Interval(float.fromhex('0X1.1A62633145C06P-54'), 1.0))
    def test_0025_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(-0.0, float.fromhex('0X1.921FB54442D18P+0'))), Interval(float.fromhex('0X1.1A62633145C06P-54'), 1.0))
    def test_0026_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(0.0, float.fromhex('0X1.921FB54442D19P+0'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-53'), 1.0))
    def test_0027_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(-0.0, float.fromhex('0X1.921FB54442D19P+0'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-53'), 1.0))
    def test_0028_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('0X1.921FB54442D18P+1'), float.fromhex('0X1.921FB54442D18P+1'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('-0X1.FFFFFFFFFFFFFP-1')))
    def test_0029_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('0X1.921FB54442D19P+1'), float.fromhex('0X1.921FB54442D19P+1'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('-0X1.FFFFFFFFFFFFFP-1')))
    def test_0030_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('0X1.921FB54442D18P+1'), float.fromhex('0X1.921FB54442D19P+1'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('-0X1.FFFFFFFFFFFFFP-1')))
    def test_0031_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(0.0, float.fromhex('0X1.921FB54442D18P+1'))), Interval(-1.0, 1.0))
    def test_0032_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(-0.0, float.fromhex('0X1.921FB54442D18P+1'))), Interval(-1.0, 1.0))
    def test_0033_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(0.0, float.fromhex('0X1.921FB54442D19P+1'))), Interval(-1.0, 1.0))
    def test_0034_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(-0.0, float.fromhex('0X1.921FB54442D19P+1'))), Interval(-1.0, 1.0))
    def test_0035_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('0X1.921FB54442D18P+0'), float.fromhex('0X1.921FB54442D18P+1'))), Interval(-1.0, float.fromhex('0X1.1A62633145C07P-54')))
    def test_0036_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('0X1.921FB54442D18P+0'), float.fromhex('0X1.921FB54442D19P+1'))), Interval(-1.0, float.fromhex('0X1.1A62633145C07P-54')))
    def test_0037_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('0X1.921FB54442D19P+0'), float.fromhex('0X1.921FB54442D18P+1'))), Interval(-1.0, float.fromhex('-0X1.72CECE675D1FCP-53')))
    def test_0038_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('0X1.921FB54442D19P+0'), float.fromhex('0X1.921FB54442D19P+1'))), Interval(-1.0, float.fromhex('-0X1.72CECE675D1FCP-53')))
    def test_0039_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D18P+0'), float.fromhex('-0X1.921FB54442D18P+0'))), Interval(float.fromhex('0X1.1A62633145C06P-54'), float.fromhex('0X1.1A62633145C07P-54')))
    def test_0040_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D19P+0'), float.fromhex('-0X1.921FB54442D19P+0'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-53'), float.fromhex('-0X1.72CECE675D1FCP-53')))
    def test_0041_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D19P+0'), float.fromhex('-0X1.921FB54442D18P+0'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-53'), float.fromhex('0X1.1A62633145C07P-54')))
    def test_0042_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D18P+0'), 0.0)), Interval(float.fromhex('0X1.1A62633145C06P-54'), 1.0))
    def test_0043_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D18P+0'), -0.0)), Interval(float.fromhex('0X1.1A62633145C06P-54'), 1.0))
    def test_0044_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D19P+0'), 0.0)), Interval(float.fromhex('-0X1.72CECE675D1FDP-53'), 1.0))
    def test_0045_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D19P+0'), -0.0)), Interval(float.fromhex('-0X1.72CECE675D1FDP-53'), 1.0))
    def test_0046_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D18P+1'), float.fromhex('-0X1.921FB54442D18P+1'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('-0X1.FFFFFFFFFFFFFP-1')))
    def test_0047_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D19P+1'), float.fromhex('-0X1.921FB54442D19P+1'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('-0X1.FFFFFFFFFFFFFP-1')))
    def test_0048_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D19P+1'), float.fromhex('-0X1.921FB54442D18P+1'))), Interval(float.fromhex('-0X1P+0'), float.fromhex('-0X1.FFFFFFFFFFFFFP-1')))
    def test_0049_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D18P+1'), 0.0)), Interval(-1.0, 1.0))
    def test_0050_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D18P+1'), -0.0)), Interval(-1.0, 1.0))
    def test_0051_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D19P+1'), 0.0)), Interval(-1.0, 1.0))
    def test_0052_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D19P+1'), -0.0)), Interval(-1.0, 1.0))
    def test_0053_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D18P+1'), float.fromhex('-0X1.921FB54442D18P+0'))), Interval(-1.0, float.fromhex('0X1.1A62633145C07P-54')))
    def test_0054_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D19P+1'), float.fromhex('-0X1.921FB54442D18P+0'))), Interval(-1.0, float.fromhex('0X1.1A62633145C07P-54')))
    def test_0055_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D18P+1'), float.fromhex('-0X1.921FB54442D19P+0'))), Interval(-1.0, float.fromhex('-0X1.72CECE675D1FCP-53')))
    def test_0056_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D19P+1'), float.fromhex('-0X1.921FB54442D19P+0'))), Interval(-1.0, float.fromhex('-0X1.72CECE675D1FCP-53')))
    def test_0057_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D18P+0'), float.fromhex('0X1.921FB54442D18P+0'))), Interval(float.fromhex('0X1.1A62633145C06P-54'), 1.0))
    def test_0058_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D18P+0'), float.fromhex('0X1.921FB54442D19P+0'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-53'), 1.0))
    def test_0059_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D19P+0'), float.fromhex('0X1.921FB54442D18P+0'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-53'), 1.0))
    def test_0060_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(float.fromhex('-0X1.921FB54442D19P+0'), float.fromhex('0X1.921FB54442D19P+0'))), Interval(float.fromhex('-0X1.72CECE675D1FDP-53'), 1.0))
    def test_0061_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(-0.7, 0.1)), Interval(float.fromhex('0X1.87996529F9D92P-1'), 1.0))
    def test_0062_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(1.0, 2.0)), Interval(float.fromhex('-0X1.AA22657537205P-2'), float.fromhex('0X1.14A280FB5068CP-1')))
    def test_0063_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(-3.2, -2.9)), Interval(-1.0, float.fromhex('-0X1.F1216DBA340C8P-1')))
    def test_0064_minimal_cos_test(self):
        self.assertEqual(ip.cos(Interval(2.0, 3.0)), Interval(float.fromhex('-0X1.FAE04BE85E5D3P-1'), float.fromhex('-0X1.AA22657537204P-2')))

suite.addTest(TestCase_minimal_cos_test())

class TestCase_minimal_cos_dec_test(unittest.TestCase):
    """minimal_cos_dec_test"""

suite.addTest(TestCase_minimal_cos_dec_test())

class TestCase_minimal_tan_test(unittest.TestCase):
    """minimal_tan_test"""

suite.addTest(TestCase_minimal_tan_test())

class TestCase_minimal_tan_dec_test(unittest.TestCase):
    """minimal_tan_dec_test"""

suite.addTest(TestCase_minimal_tan_dec_test())

class TestCase_minimal_asin_test(unittest.TestCase):
    """minimal_asin_test"""

suite.addTest(TestCase_minimal_asin_test())

class TestCase_minimal_asin_dec_test(unittest.TestCase):
    """minimal_asin_dec_test"""

suite.addTest(TestCase_minimal_asin_dec_test())

class TestCase_minimal_acos_test(unittest.TestCase):
    """minimal_acos_test"""

suite.addTest(TestCase_minimal_acos_test())

class TestCase_minimal_acos_dec_test(unittest.TestCase):
    """minimal_acos_dec_test"""

suite.addTest(TestCase_minimal_acos_dec_test())

class TestCase_minimal_atan_test(unittest.TestCase):
    """minimal_atan_test"""

suite.addTest(TestCase_minimal_atan_test())

class TestCase_minimal_atan_dec_test(unittest.TestCase):
    """minimal_atan_dec_test"""

suite.addTest(TestCase_minimal_atan_dec_test())

class TestCase_minimal_atan2_test(unittest.TestCase):
    """minimal_atan2_test"""

suite.addTest(TestCase_minimal_atan2_test())

class TestCase_minimal_atan2_dec_test(unittest.TestCase):
    """minimal_atan2_dec_test"""

suite.addTest(TestCase_minimal_atan2_dec_test())

class TestCase_minimal_sinh_test(unittest.TestCase):
    """minimal_sinh_test"""

suite.addTest(TestCase_minimal_sinh_test())

class TestCase_minimal_sinh_dec_test(unittest.TestCase):
    """minimal_sinh_dec_test"""

suite.addTest(TestCase_minimal_sinh_dec_test())

class TestCase_minimal_cosh_test(unittest.TestCase):
    """minimal_cosh_test"""

suite.addTest(TestCase_minimal_cosh_test())

class TestCase_minimal_cosh_dec_test(unittest.TestCase):
    """minimal_cosh_dec_test"""

suite.addTest(TestCase_minimal_cosh_dec_test())

class TestCase_minimal_tanh_test(unittest.TestCase):
    """minimal_tanh_test"""

suite.addTest(TestCase_minimal_tanh_test())

class TestCase_minimal_tanh_dec_test(unittest.TestCase):
    """minimal_tanh_dec_test"""

suite.addTest(TestCase_minimal_tanh_dec_test())

class TestCase_minimal_asinh_test(unittest.TestCase):
    """minimal_asinh_test"""

suite.addTest(TestCase_minimal_asinh_test())

class TestCase_minimal_asinh_dec_test(unittest.TestCase):
    """minimal_asinh_dec_test"""

suite.addTest(TestCase_minimal_asinh_dec_test())

class TestCase_minimal_acosh_test(unittest.TestCase):
    """minimal_acosh_test"""

suite.addTest(TestCase_minimal_acosh_test())

class TestCase_minimal_acosh_dec_test(unittest.TestCase):
    """minimal_acosh_dec_test"""

suite.addTest(TestCase_minimal_acosh_dec_test())

class TestCase_minimal_atanh_test(unittest.TestCase):
    """minimal_atanh_test"""

suite.addTest(TestCase_minimal_atanh_test())

class TestCase_minimal_atanh_dec_test(unittest.TestCase):
    """minimal_atanh_dec_test"""

suite.addTest(TestCase_minimal_atanh_dec_test())

class TestCase_minimal_sign_test(unittest.TestCase):
    """minimal_sign_test"""

suite.addTest(TestCase_minimal_sign_test())

class TestCase_minimal_sign_dec_test(unittest.TestCase):
    """minimal_sign_dec_test"""

suite.addTest(TestCase_minimal_sign_dec_test())

class TestCase_minimal_ceil_test(unittest.TestCase):
    """minimal_ceil_test"""

suite.addTest(TestCase_minimal_ceil_test())

class TestCase_minimal_ceil_dec_test(unittest.TestCase):
    """minimal_ceil_dec_test"""

suite.addTest(TestCase_minimal_ceil_dec_test())

class TestCase_minimal_floor_test(unittest.TestCase):
    """minimal_floor_test"""

suite.addTest(TestCase_minimal_floor_test())

class TestCase_minimal_floor_dec_test(unittest.TestCase):
    """minimal_floor_dec_test"""

suite.addTest(TestCase_minimal_floor_dec_test())

class TestCase_minimal_trunc_test(unittest.TestCase):
    """minimal_trunc_test"""

suite.addTest(TestCase_minimal_trunc_test())

class TestCase_minimal_trunc_dec_test(unittest.TestCase):
    """minimal_trunc_dec_test"""

suite.addTest(TestCase_minimal_trunc_dec_test())

class TestCase_minimal_round_ties_to_even_test(unittest.TestCase):
    """minimal_round_ties_to_even_test"""

suite.addTest(TestCase_minimal_round_ties_to_even_test())

class TestCase_minimal_round_ties_to_even_dec_test(unittest.TestCase):
    """minimal_round_ties_to_even_dec_test"""

suite.addTest(TestCase_minimal_round_ties_to_even_dec_test())

class TestCase_minimal_round_ties_to_away_test(unittest.TestCase):
    """minimal_round_ties_to_away_test"""

suite.addTest(TestCase_minimal_round_ties_to_away_test())

class TestCase_minimal_round_ties_to_away_dec_test(unittest.TestCase):
    """minimal_round_ties_to_away_dec_test"""

suite.addTest(TestCase_minimal_round_ties_to_away_dec_test())

class TestCase_minimal_abs_test(unittest.TestCase):
    """minimal_abs_test"""
    def test_0001_minimal_abs_test(self):
        self.assertEqual(abs(Interval(math.nan, math.nan)), Interval(math.nan, math.nan))
    def test_0002_minimal_abs_test(self):
        self.assertEqual(abs(Interval(-math.inf, math.inf)), Interval(0.0, math.inf))
    def test_0003_minimal_abs_test(self):
        self.assertEqual(abs(Interval(1.1, 2.1)), Interval(1.1, 2.1))
    def test_0004_minimal_abs_test(self):
        self.assertEqual(abs(Interval(-1.1, 2.0)), Interval(0.0, 2.0))
    def test_0005_minimal_abs_test(self):
        self.assertEqual(abs(Interval(-1.1, 0.0)), Interval(0.0, 1.1))
    def test_0006_minimal_abs_test(self):
        self.assertEqual(abs(Interval(-1.1, -0.0)), Interval(0.0, 1.1))
    def test_0007_minimal_abs_test(self):
        self.assertEqual(abs(Interval(-1.1, -0.4)), Interval(0.4, 1.1))
    def test_0008_minimal_abs_test(self):
        self.assertEqual(abs(Interval(-1.9, 0.2)), Interval(0.0, 1.9))
    def test_0009_minimal_abs_test(self):
        self.assertEqual(abs(Interval(0.0, 0.2)), Interval(0.0, 0.2))
    def test_0010_minimal_abs_test(self):
        self.assertEqual(abs(Interval(-0.0, 0.2)), Interval(0.0, 0.2))
    def test_0011_minimal_abs_test(self):
        self.assertEqual(abs(Interval(-1.5, math.inf)), Interval(0.0, math.inf))
    def test_0012_minimal_abs_test(self):
        self.assertEqual(abs(Interval(-math.inf, -2.2)), Interval(2.2, math.inf))

suite.addTest(TestCase_minimal_abs_test())

class TestCase_minimal_abs_dec_test(unittest.TestCase):
    """minimal_abs_dec_test"""

suite.addTest(TestCase_minimal_abs_dec_test())

class TestCase_minimal_min_test(unittest.TestCase):
    """minimal_min_test"""

suite.addTest(TestCase_minimal_min_test())

class TestCase_minimal_min_dec_test(unittest.TestCase):
    """minimal_min_dec_test"""

suite.addTest(TestCase_minimal_min_dec_test())

class TestCase_minimal_max_test(unittest.TestCase):
    """minimal_max_test"""

suite.addTest(TestCase_minimal_max_test())

class TestCase_minimal_max_dec_test(unittest.TestCase):
    """minimal_max_dec_test"""

suite.addTest(TestCase_minimal_max_dec_test())

if __name__ == '__main__':
    unittest.main()
