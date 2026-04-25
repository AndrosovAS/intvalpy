#
# 
# Unit tests from FI_LIB version 1.2
# (Original authors: Werner Hofschuster and Walter Kraemer)
# converted into portable ITL format by Oliver Heimlich.
# 
# Copyright 1997-2000 Institut fuer Wissenschaftliches Rechnen
#                     und Mathematische Modellbildung (IWRMM)
#                                      and
#                     Institut fuer Angewandte Mathematik
#                     Universitaet Karlsruhe, Germany
# Copyright 2000-2005 Wiss. Rechnen/Softwaretechnologie
#                     Universitaet Wuppertal, Germany
# Copyright 2015-2016 Oliver Heimlich (oheim@posteo.de)
# 
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
# 
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
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
class TestCase_FI_LIB_addii(unittest.TestCase):
    """FI_LIB_addii"""
    def test_1(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) + Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_2(self):
        self.assertEqual(Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')) + Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')), Interval(float.fromhex('0X3.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')))
    def test_3(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')) + Interval(float.fromhex('-0X2.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')), Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X3.0000000000000P+0')))
    def test_4(self):
        self.assertEqual(Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')) + Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_5(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')) + Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_6(self):
        self.assertEqual(Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')) + Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')), Interval(float.fromhex('0X3.F400000000000P-1064'), float.fromhex('0X3.F400000000000P-1064')))
    def test_7(self):
        self.assertEqual(Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')) + Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000001P+0')))
    def test_8(self):
        self.assertEqual(Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')) + Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')), Interval(float.fromhex('-0X3.F400000000000P-1064'), float.fromhex('-0X3.F400000000000P-1064')))
    def test_9(self):
        self.assertEqual(Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')) + Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('0XF.FFFFFFFFFFFF8P-4'), float.fromhex('0X1.0000000000000P+0')))
    def test_10(self):
        self.assertEqual(Interval(float.fromhex('0XF.FFFFFFFFFFFF8P+1020'), float.fromhex('0XF.FFFFFFFFFFFF8P+1020')) + Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')), Interval(float.fromhex('0XF.FFFFFFFFFFFF0P+1020'), float.fromhex('0XF.FFFFFFFFFFFF8P+1020')))
    def test_11(self):
        self.assertEqual(Interval(float.fromhex('-0XF.FFFFFFFFFFFF8P+1020'), float.fromhex('-0XF.FFFFFFFFFFFF8P+1020')) + Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('-0XF.FFFFFFFFFFFF8P+1020'), float.fromhex('-0XF.FFFFFFFFFFFF0P+1020')))
    def test_12(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')) + Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')), Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')))
    def test_13(self):
        self.assertEqual(Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')) + Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')))
    def test_14(self):
        self.assertEqual(Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')) + Interval(float.fromhex('0X3.0000000000000P+0'), float.fromhex('0X4.0000000000000P+0')), Interval(float.fromhex('0X4.0000000000000P+0'), float.fromhex('0X6.0000000000000P+0')))
    def test_15(self):
        self.assertEqual(Interval(float.fromhex('0X3.0000000000000P+0'), float.fromhex('0X4.0000000000000P+0')) + Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')), Interval(float.fromhex('0X4.0000000000000P+0'), float.fromhex('0X6.0000000000000P+0')))
    def test_16(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) + Interval(float.fromhex('-0X4.0000000000000P+0'), float.fromhex('-0X3.0000000000000P+0')), Interval(float.fromhex('-0X5.0000000000000P+0'), float.fromhex('-0X3.0000000000000P+0')))
    def test_17(self):
        self.assertEqual(Interval(float.fromhex('-0X4.0000000000000P+0'), float.fromhex('-0X3.0000000000000P+0')) + Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('-0X5.0000000000000P+0'), float.fromhex('-0X3.0000000000000P+0')))
    def test_18(self):
        self.assertEqual(Interval(float.fromhex('-0X5.0000000000000P+0'), float.fromhex('-0X4.0000000000000P+0')) + Interval(float.fromhex('0X4.0000000000000P+0'), float.fromhex('0X5.0000000000000P+0')), Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')))
    def test_19(self):
        self.assertEqual(Interval(float.fromhex('0X4.0000000000000P+0'), float.fromhex('0X5.0000000000000P+0')) + Interval(float.fromhex('-0X5.0000000000000P+0'), float.fromhex('-0X4.0000000000000P+0')), Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')))
suite.addTest(TestCase_FI_LIB_addii())

class TestCase_FI_LIB_subii(unittest.TestCase):
    """FI_LIB_subii"""
    def test_20(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) - Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_21(self):
        self.assertEqual(Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')) - Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')), Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')))
    def test_22(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')) - Interval(float.fromhex('-0X2.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')), Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')))
    def test_23(self):
        self.assertEqual(Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')) - Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')), Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')))
    def test_24(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')) - Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('-0X2.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')))
    def test_25(self):
        self.assertEqual(Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')) - Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_26(self):
        self.assertEqual(Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')) - Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0XF.FFFFFFFFFFFF8P-4')))
    def test_27(self):
        self.assertEqual(Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')) - Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_28(self):
        self.assertEqual(Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')) - Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('-0X1.0000000000001P+0'), float.fromhex('-0X1.0000000000000P+0')))
    def test_29(self):
        self.assertEqual(Interval(float.fromhex('0XF.FFFFFFFFFFFF8P+1020'), float.fromhex('0XF.FFFFFFFFFFFF8P+1020')) - Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('0XF.FFFFFFFFFFFF0P+1020'), float.fromhex('0XF.FFFFFFFFFFFF8P+1020')))
    def test_30(self):
        self.assertEqual(Interval(float.fromhex('-0XF.FFFFFFFFFFFF8P+1020'), float.fromhex('-0XF.FFFFFFFFFFFF8P+1020')) - Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')), Interval(float.fromhex('-0XF.FFFFFFFFFFFF8P+1020'), float.fromhex('-0XF.FFFFFFFFFFFF0P+1020')))
    def test_31(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')) - Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')), Interval(float.fromhex('-0X2.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_32(self):
        self.assertEqual(Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')) - Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')))
    def test_33(self):
        self.assertEqual(Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')) - Interval(float.fromhex('0X3.0000000000000P+0'), float.fromhex('0X4.0000000000000P+0')), Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')))
    def test_34(self):
        self.assertEqual(Interval(float.fromhex('0X3.0000000000000P+0'), float.fromhex('0X4.0000000000000P+0')) - Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')), Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')))
    def test_35(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) - Interval(float.fromhex('-0X4.0000000000000P+0'), float.fromhex('-0X3.0000000000000P+0')), Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X4.0000000000000P+0')))
    def test_36(self):
        self.assertEqual(Interval(float.fromhex('-0X4.0000000000000P+0'), float.fromhex('-0X3.0000000000000P+0')) - Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('-0X4.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')))
    def test_37(self):
        self.assertEqual(Interval(float.fromhex('-0X5.0000000000000P+0'), float.fromhex('-0X4.0000000000000P+0')) - Interval(float.fromhex('0X4.0000000000000P+0'), float.fromhex('0X5.0000000000000P+0')), Interval(float.fromhex('-0XA.0000000000000P+0'), float.fromhex('-0X8.0000000000000P+0')))
    def test_38(self):
        self.assertEqual(Interval(float.fromhex('0X4.0000000000000P+0'), float.fromhex('0X5.0000000000000P+0')) - Interval(float.fromhex('-0X5.0000000000000P+0'), float.fromhex('-0X4.0000000000000P+0')), Interval(float.fromhex('0X8.0000000000000P+0'), float.fromhex('0XA.0000000000000P+0')))
suite.addTest(TestCase_FI_LIB_subii())

class TestCase_FI_LIB_mulii(unittest.TestCase):
    """FI_LIB_mulii"""
    def test_39(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) * Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_40(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')) * Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')), Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')))
    def test_41(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')) * Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_42(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')) * Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')))
    def test_43(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) * Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_44(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) * Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_45(self):
        self.assertEqual(Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')) * Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')), Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')))
    def test_46(self):
        self.assertEqual(Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')) * Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_47(self):
        self.assertEqual(Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')) * Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')))
    def test_48(self):
        self.assertEqual(Interval(float.fromhex('0XF.FFFFFFFFFFFF8P+1020'), float.fromhex('0XF.FFFFFFFFFFFF8P+1020')) * Interval(float.fromhex('0X8.0000000000000P-4'), float.fromhex('0X8.0000000000000P-4')), Interval(float.fromhex('0X7.FFFFFFFFFFFFCP+1020'), float.fromhex('0X7.FFFFFFFFFFFFCP+1020')))
    def test_49(self):
        self.assertEqual(Interval(float.fromhex('-0XF.FFFFFFFFFFFF8P+1020'), float.fromhex('-0XF.FFFFFFFFFFFF8P+1020')) * Interval(float.fromhex('0X8.0000000000000P-4'), float.fromhex('0X8.0000000000000P-4')), Interval(float.fromhex('-0X7.FFFFFFFFFFFFCP+1020'), float.fromhex('-0X7.FFFFFFFFFFFFCP+1020')))
    def test_50(self):
        self.assertEqual(Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')) * Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X4.0000000000000P-1076')))
    def test_51(self):
        self.assertEqual(Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')) * Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')))
    def test_52(self):
        self.assertEqual(Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')) * Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')), Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')))
    def test_53(self):
        self.assertEqual(Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')) * Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')), Interval(float.fromhex('0X4.0000000000000P+0'), float.fromhex('0X9.0000000000000P+0')))
    def test_54(self):
        self.assertEqual(Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')) * Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')), Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('+0X3.0000000000000P+0')))
    def test_55(self):
        self.assertEqual(Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')) * Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')), Interval(float.fromhex('-0X9.0000000000000P+0'), float.fromhex('-0X4.0000000000000P+0')))
    def test_56(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')) * Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')), Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('+0X3.0000000000000P+0')))
    def test_57(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')) * Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')), Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')))
    def test_58(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')) * Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')), Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('+0X3.0000000000000P+0')))
    def test_59(self):
        self.assertEqual(Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')) * Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')), Interval(float.fromhex('-0X9.0000000000000P+0'), float.fromhex('-0X4.0000000000000P+0')))
    def test_60(self):
        self.assertEqual(Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')) * Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')), Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('+0X3.0000000000000P+0')))
    def test_61(self):
        self.assertEqual(Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')) * Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')), Interval(float.fromhex('0X4.0000000000000P+0'), float.fromhex('0X9.0000000000000P+0')))
    def test_62(self):
        self.assertEqual(Interval(float.fromhex('-0X5.0000000000000P+0'), float.fromhex('+0X2.0000000000000P+0')) * Interval(float.fromhex('-0X4.0000000000000P+0'), float.fromhex('+0X3.0000000000000P+0')), Interval(float.fromhex('-0XF.0000000000000P+0'), float.fromhex('+0X1.4000000000000P+4')))
    def test_63(self):
        self.assertEqual(Interval(float.fromhex('-0X5.0000000000000P+0'), float.fromhex('+0X2.0000000000000P+0')) * Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('+0X8.0000000000000P+0')), Interval(float.fromhex('-0X2.8000000000000P+4'), float.fromhex('+0X1.0000000000000P+4')))
    def test_64(self):
        self.assertEqual(Interval(float.fromhex('-0X2.0000000000000P+0'), float.fromhex('+0X5.0000000000000P+0')) * Interval(float.fromhex('-0X4.0000000000000P+0'), float.fromhex('+0X3.0000000000000P+0')), Interval(float.fromhex('-0X1.4000000000000P+4'), float.fromhex('+0XF.0000000000000P+0')))
    def test_65(self):
        self.assertEqual(Interval(float.fromhex('-0X4.0000000000000P+0'), float.fromhex('+0X5.0000000000000P+0')) * Interval(float.fromhex('-0X4.0000000000000P+0'), float.fromhex('+0X3.0000000000000P+0')), Interval(float.fromhex('-0X1.4000000000000P+4'), float.fromhex('+0X1.0000000000000P+4')))
    def test_66(self):
        self.assertEqual(Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) * Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X9.0000000000000P+0')))
    def test_67(self):
        self.assertEqual(Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) * Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')), Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('+0X3.0000000000000P+0')))
    def test_68(self):
        self.assertEqual(Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) * Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')), Interval(float.fromhex('-0X9.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_69(self):
        self.assertEqual(Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')) * Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X9.0000000000000P+0')))
    def test_70(self):
        self.assertEqual(Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')) * Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')), Interval(float.fromhex('-0X9.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_71(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')) * Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('+0X3.0000000000000P+0')))
    def test_72(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')) * Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')), Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('+0X3.0000000000000P+0')))
    def test_73(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')) * Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')), Interval(float.fromhex('-0X9.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_74(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')) * Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')), Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('+0X3.0000000000000P+0')))
    def test_75(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')) * Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X9.0000000000000P+0')))
    def test_76(self):
        self.assertEqual(Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')) * Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('-0X9.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_77(self):
        self.assertEqual(Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')) * Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('-0X9.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_78(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')) * Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('-0X6.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_79(self):
        self.assertEqual(Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')) * Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X4.0000000000000P-1076')))
    def test_80(self):
        self.assertEqual(Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')) * Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')))
    def test_81(self):
        self.assertEqual(Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')) * Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')), Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')))
    def test_82(self):
        self.assertEqual(Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')) * Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')), Interval(float.fromhex('-0X4.0000000000000P-1076'), float.fromhex('0X0.0000000000000P+0')))
    def test_83(self):
        self.assertEqual(Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')) * Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_84(self):
        self.assertEqual(Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')) * Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
suite.addTest(TestCase_FI_LIB_mulii())

class TestCase_FI_LIB_divii(unittest.TestCase):
    """FI_LIB_divii"""
    def test_85(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')) / Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')), Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')))
    def test_86(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')) / Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')))
    def test_87(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) / Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_88(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) / Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_89(self):
        self.assertEqual(Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')) / Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')), Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')))
    def test_90(self):
        self.assertEqual(Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')) / Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')))
    def test_91(self):
        self.assertEqual(Interval(float.fromhex('0XF.FFFFFFFFFFFF8P+1020'), float.fromhex('0XF.FFFFFFFFFFFF8P+1020')) / Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')), Interval(float.fromhex('0X7.FFFFFFFFFFFFCP+1020'), float.fromhex('0X7.FFFFFFFFFFFFCP+1020')))
    def test_92(self):
        self.assertEqual(Interval(float.fromhex('-0XF.FFFFFFFFFFFF8P+1020'), float.fromhex('-0XF.FFFFFFFFFFFF8P+1020')) / Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X2.0000000000000P+0')), Interval(float.fromhex('-0X7.FFFFFFFFFFFFCP+1020'), float.fromhex('-0X7.FFFFFFFFFFFFCP+1020')))
    def test_93(self):
        self.assertEqual(Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')) / Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')), Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')))
    def test_94(self):
        self.assertEqual(Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')) / Interval(float.fromhex('0X1.0000000000000P+0'), float.fromhex('0X1.0000000000000P+0')), Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')))
    def test_95(self):
        self.assertEqual(Interval(float.fromhex('0X1.FA00000000000P-1064'), float.fromhex('0X1.FA00000000000P-1064')) / Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('-0X1.0000000000000P+0')), Interval(float.fromhex('-0X1.FA00000000000P-1064'), float.fromhex('-0X1.FA00000000000P-1064')))
    def test_96(self):
        self.assertEqual(Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')) / Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')), Interval(float.fromhex('0XA.AAAAAAAAAAAA8P-4'), float.fromhex('0X1.8000000000000P+0')))
    def test_97(self):
        self.assertEqual(Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')) / Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')), Interval(float.fromhex('-0X1.8000000000000P+0'), float.fromhex('-0XA.AAAAAAAAAAAA8P-4')))
    def test_98(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')) / Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')), Interval(float.fromhex('-0X8.0000000000000P-4'), float.fromhex('+0X8.0000000000000P-4')))
    def test_99(self):
        self.assertEqual(Interval(float.fromhex('-0X1.0000000000000P+0'), float.fromhex('+0X1.0000000000000P+0')) / Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')), Interval(float.fromhex('-0X8.0000000000000P-4'), float.fromhex('+0X8.0000000000000P-4')))
    def test_100(self):
        self.assertEqual(Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')) / Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')), Interval(float.fromhex('-0X1.8000000000000P+0'), float.fromhex('-0XA.AAAAAAAAAAAA8P-4')))
    def test_101(self):
        self.assertEqual(Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')) / Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')), Interval(float.fromhex('0XA.AAAAAAAAAAAA8P-4'), float.fromhex('0X1.8000000000000P+0')))
    def test_102(self):
        self.assertEqual(Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) / Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X1.8000000000000P+0')))
    def test_103(self):
        self.assertEqual(Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('0X0.0000000000000P+0')) / Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')), Interval(float.fromhex('-0X1.8000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_104(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')) / Interval(float.fromhex('-0X3.0000000000000P+0'), float.fromhex('-0X2.0000000000000P+0')), Interval(float.fromhex('-0X1.8000000000000P+0'), float.fromhex('0X0.0000000000000P+0')))
    def test_105(self):
        self.assertEqual(Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')) / Interval(float.fromhex('0X2.0000000000000P+0'), float.fromhex('0X3.0000000000000P+0')), Interval(float.fromhex('0X0.0000000000000P+0'), float.fromhex('0X1.8000000000000P+0')))
suite.addTest(TestCase_FI_LIB_divii())

class TestCase_FI_LIB_unary_functions(unittest.TestCase):
    """FI_LIB_unary_functions"""

suite.addTest(TestCase_FI_LIB_unary_functions())

if __name__ == '__main__':
    unittest.main()
