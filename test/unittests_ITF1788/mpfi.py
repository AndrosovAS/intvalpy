#
# 
# Unit tests from GNU MPFI
# (Original authors: Philippe Theveny and Nathalie Revol )
# converted into portable ITL format by Oliver Heimlich.
# 
# Copyright 2009–2012 Spaces project, Inria Lorraine
#                     and Salsa project, INRIA Rocquencourt,
#                     and Arenaire project, Inria Rhone-Alpes, France
#                     and Lab. ANO, USTL (Univ. of Lille), France
# Copyright 2015-2016 Oliver Heimlich
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
class TestCase_mpfi_abs(unittest.TestCase):
    """mpfi_abs"""

suite.addTest(TestCase_mpfi_abs())

class TestCase_mpfi_acos(unittest.TestCase):
    """mpfi_acos"""

suite.addTest(TestCase_mpfi_acos())

class TestCase_mpfi_acosh(unittest.TestCase):
    """mpfi_acosh"""

suite.addTest(TestCase_mpfi_acosh())

class TestCase_mpfi_add(unittest.TestCase):
    """mpfi_add"""
    # special values
    def test_26(self):
        self.assertEqual(Interval(-inf, -7.0) + Interval(-1.0, +8.0), Interval(-inf, +1.0))
    def test_27(self):
        self.assertEqual(Interval(-inf, 0.0) + Interval(+8.0, inf), Interval(-math.inf, math.inf))
    def test_28(self):
        self.assertEqual(Interval(-inf, +8.0) + Interval(0.0, +8.0), Interval(-inf, +16.0))
    def test_29(self):
        self.assertEqual(Interval(-math.inf, math.inf) + Interval(0.0, +8.0), Interval(-math.inf, math.inf))
    def test_30(self):
        self.assertEqual(Interval(0.0, 0.0) + Interval(-inf, -7.0), Interval(-inf, -7.0))
    def test_31(self):
        self.assertEqual(Interval(0.0, +8.0) + Interval(-7.0, 0.0), Interval(-7.0, +8.0))
    def test_32(self):
        self.assertEqual(Interval(0.0, 0.0) + Interval(0.0, +8.0), Interval(0.0, +8.0))
    def test_33(self):
        self.assertEqual(Interval(0.0, inf) + Interval(0.0, +8.0), Interval(0.0, inf))
    def test_34(self):
        self.assertEqual(Interval(0.0, 0.0) + Interval(+8.0, inf), Interval(+8.0, inf))
    def test_35(self):
        self.assertEqual(Interval(0.0, 0.0) + Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_36(self):
        self.assertEqual(Interval(0.0, +8.0) + Interval(0.0, +8.0), Interval(0.0, +16.0))
    def test_37(self):
        self.assertEqual(Interval(0.0, 0.0) + Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_38(self):
        self.assertEqual(Interval(0.0, inf) + Interval(-7.0, +8.0), Interval(-7.0, inf))
    # regular values
    def test_39(self):
        self.assertEqual(Interval(-0.375, float.fromhex('-0x10187p-256')) + Interval(-0.125, float.fromhex('0x1p-240')), Interval(float.fromhex('-0x1p-1'), float.fromhex('-0x187p-256')))
    def test_40(self):
        self.assertEqual(Interval(float.fromhex('-0x1p-300'), float.fromhex('0x123456p+28')) + Interval(float.fromhex('-0x10000000000000p-93'), float.fromhex('0x789abcdp0')), Interval(float.fromhex('-0x10000000000001p-93'), float.fromhex('0x123456789abcdp0')))
    def test_41(self):
        self.assertEqual(Interval(-4.0, +7.0) + Interval(float.fromhex('-0x123456789abcdp-17'), 3e300), Interval(float.fromhex('-0x123456791abcdp-17'), float.fromhex('0x8f596b3002c1bp+947')))
    def test_42(self):
        self.assertEqual(Interval(float.fromhex('0x1000100010001p+8'), float.fromhex('0x1p+60')) + Interval(float.fromhex('0x1000100010001p0'), 3.0e300), Interval(float.fromhex('+0x1010101010101p+8'), float.fromhex('0x8f596b3002c1bp+947')))
    # signed zeros
    def test_43(self):
        self.assertEqual(Interval(+4.0, +8.0) + Interval(-4.0, -2.0), Interval(0.0, +6.0))
    def test_44(self):
        self.assertEqual(Interval(+4.0, +8.0) + Interval(-9.0, -8.0), Interval(-5.0, 0.0))
suite.addTest(TestCase_mpfi_add())

class TestCase_mpfi_add_d(unittest.TestCase):
    """mpfi_add_d"""
    # special values
    def test_45(self):
        self.assertEqual(Interval(-inf, -7.0) + Interval(float.fromhex('-0x170ef54646d497p-107'), float.fromhex('-0x170ef54646d497p-107')), Interval(-inf, -7.0))
    def test_46(self):
        self.assertEqual(Interval(-inf, -7.0) + Interval(0.0, 0.0), Interval(-inf, -7.0))
    def test_47(self):
        self.assertEqual(Interval(-inf, -7.0) + Interval(float.fromhex('0x170ef54646d497p-107'), float.fromhex('0x170ef54646d497p-107')), Interval(-inf, float.fromhex('-0x1bffffffffffffp-50')))
    def test_48(self):
        self.assertEqual(Interval(-inf, 0.0) + Interval(float.fromhex('-0x170ef54646d497p-106'), float.fromhex('-0x170ef54646d497p-106')), Interval(-inf, -8.0e-17))
    def test_49(self):
        self.assertEqual(Interval(-inf, 0.0) + Interval(0.0, 0.0), Interval(-inf, 0.0))
    def test_50(self):
        self.assertEqual(Interval(-inf, 0.0) + Interval(float.fromhex('0x170ef54646d497p-106'), float.fromhex('0x170ef54646d497p-106')), Interval(-inf, float.fromhex('0x170ef54646d497p-106')))
    def test_51(self):
        self.assertEqual(Interval(-inf, 8.0) + Interval(float.fromhex('-0x16345785d8a00000p0'), float.fromhex('-0x16345785d8a00000p0')), Interval(-inf, float.fromhex('-0x16345785d89fff00p0')))
    def test_52(self):
        self.assertEqual(Interval(-inf, 8.0) + Interval(0.0, 0.0), Interval(-inf, 8.0))
    def test_53(self):
        self.assertEqual(Interval(-inf, 8.0) + Interval(float.fromhex('0x16345785d8a00000p0'), float.fromhex('0x16345785d8a00000p0')), Interval(-inf, float.fromhex('0x16345785d8a00100p0')))
    def test_54(self):
        self.assertEqual(Interval(-math.inf, math.inf) + Interval(float.fromhex('-0x170ef54646d497p-105'), float.fromhex('-0x170ef54646d497p-105')), Interval(-math.inf, math.inf))
    def test_55(self):
        self.assertEqual(Interval(-math.inf, math.inf) + Interval(0.0e-17, 0.0e-17), Interval(-math.inf, math.inf))
    def test_56(self):
        self.assertEqual(Interval(-math.inf, math.inf) + Interval(float.fromhex('+0x170ef54646d497p-105'), float.fromhex('+0x170ef54646d497p-105')), Interval(-math.inf, math.inf))
    def test_57(self):
        self.assertEqual(Interval(0.0, 0.0) + Interval(float.fromhex('-0x170ef54646d497p-109'), float.fromhex('-0x170ef54646d497p-109')), Interval(float.fromhex('-0x170ef54646d497p-109'), float.fromhex('-0x170ef54646d497p-109')))
    def test_58(self):
        self.assertEqual(Interval(0.0, 0.0) + Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_59(self):
        self.assertEqual(Interval(0.0, 0.0) + Interval(float.fromhex('0x170ef54646d497p-109'), float.fromhex('0x170ef54646d497p-109')), Interval(float.fromhex('0x170ef54646d497p-109'), float.fromhex('0x170ef54646d497p-109')))
    def test_60(self):
        self.assertEqual(Interval(0.0, 8.0) + Interval(float.fromhex('-0x114b37f4b51f71p-107'), float.fromhex('-0x114b37f4b51f71p-107')), Interval(float.fromhex('-0x114b37f4b51f71p-107'), 8.0))
    def test_61(self):
        self.assertEqual(Interval(0.0, 8.0) + Interval(0.0, 0.0), Interval(0.0, 8.0))
    def test_62(self):
        self.assertEqual(Interval(0.0, 8.0) + Interval(float.fromhex('0x114b37f4b51f7p-103'), float.fromhex('0x114b37f4b51f7p-103')), Interval(float.fromhex('0x114b37f4b51f7p-103'), float.fromhex('0x10000000000001p-49')))
    def test_63(self):
        self.assertEqual(Interval(0.0, inf) + Interval(float.fromhex('-0x50b45a75f7e81p-104'), float.fromhex('-0x50b45a75f7e81p-104')), Interval(float.fromhex('-0x50b45a75f7e81p-104'), inf))
    def test_64(self):
        self.assertEqual(Interval(0.0, inf) + Interval(0.0, 0.0), Interval(0.0, inf))
    def test_65(self):
        self.assertEqual(Interval(0.0, inf) + Interval(float.fromhex('0x142d169d7dfa03p-106'), float.fromhex('0x142d169d7dfa03p-106')), Interval(float.fromhex('0x142d169d7dfa03p-106'), inf))
    # regular values
    def test_66(self):
        self.assertEqual(Interval(-32.0, -17.0) + Interval(float.fromhex('-0xfb53d14aa9c2fp-47'), float.fromhex('-0xfb53d14aa9c2fp-47')), Interval(float.fromhex('-0x1fb53d14aa9c2fp-47'), float.fromhex('-0x18353d14aa9c2fp-47')))
    def test_67(self):
        self.assertEqual(Interval(float.fromhex('-0xfb53d14aa9c2fp-47'), -17.0) + Interval(float.fromhex('0xfb53d14aa9c2fp-47'), float.fromhex('0xfb53d14aa9c2fp-47')), Interval(0.0, float.fromhex('0x7353d14aa9c2fp-47')))
    def test_68(self):
        self.assertEqual(Interval(-32.0, float.fromhex('-0xfb53d14aa9c2fp-48')) + Interval(float.fromhex('0xfb53d14aa9c2fp-48'), float.fromhex('0xfb53d14aa9c2fp-48')), Interval(float.fromhex('-0x104ac2eb5563d1p-48'), 0.0))
    def test_69(self):
        self.assertEqual(Interval(float.fromhex('0x123456789abcdfp-48'), float.fromhex('0x123456789abcdfp-4')) + Interval(3.5, 3.5), Interval(float.fromhex('0x15b456789abcdfp-48'), float.fromhex('0x123456789abd17p-4')))
    def test_70(self):
        self.assertEqual(Interval(float.fromhex('0x123456789abcdfp-56'), float.fromhex('0x123456789abcdfp-4')) + Interval(3.5, 3.5), Interval(float.fromhex('0x3923456789abcdp-52'), float.fromhex('0x123456789abd17p-4')))
    def test_71(self):
        self.assertEqual(Interval(float.fromhex('-0xffp0'), float.fromhex('0x123456789abcdfp-52')) + Interval(256.5, 256.5), Interval(float.fromhex('0x18p-4'), float.fromhex('0x101a3456789abdp-44')))
    def test_72(self):
        self.assertEqual(Interval(float.fromhex('-0x1fffffffffffffp-52'), float.fromhex('-0x1p-550')) + Interval(4097.5, 4097.5), Interval(float.fromhex('0xfff8p-4'), float.fromhex('0x10018p-4')))
    def test_73(self):
        self.assertEqual(Interval(float.fromhex('0x123456789abcdfp-48'), float.fromhex('0x123456789abcdfp-4')) + Interval(-3.5, -3.5), Interval(float.fromhex('0xeb456789abcdfp-48'), float.fromhex('0x123456789abca7p-4')))
    def test_74(self):
        self.assertEqual(Interval(float.fromhex('0x123456789abcdfp-56'), float.fromhex('0x123456789abcdfp-4')) + Interval(-3.5, -3.5), Interval(float.fromhex('-0x36dcba98765434p-52'), float.fromhex('0x123456789abca7p-4')))
    def test_75(self):
        self.assertEqual(Interval(float.fromhex('-0xffp0'), float.fromhex('0x123456789abcdfp-52')) + Interval(-256.5, -256.5), Interval(float.fromhex('-0x1ff8p-4'), float.fromhex('-0xff5cba9876543p-44')))
    def test_76(self):
        self.assertEqual(Interval(float.fromhex('-0x1fffffffffffffp-52'), float.fromhex('-0x1p-550')) + Interval(-4097.5, -4097.5), Interval(float.fromhex('-0x10038p-4'), float.fromhex('-0x10018p-4')))
suite.addTest(TestCase_mpfi_add_d())

class TestCase_mpfi_asin(unittest.TestCase):
    """mpfi_asin"""

suite.addTest(TestCase_mpfi_asin())

class TestCase_mpfi_asinh(unittest.TestCase):
    """mpfi_asinh"""

suite.addTest(TestCase_mpfi_asinh())

class TestCase_mpfi_atan(unittest.TestCase):
    """mpfi_atan"""

suite.addTest(TestCase_mpfi_atan())

class TestCase_mpfi_atan2(unittest.TestCase):
    """mpfi_atan2"""

suite.addTest(TestCase_mpfi_atan2())

class TestCase_mpfi_atanh(unittest.TestCase):
    """mpfi_atanh"""

suite.addTest(TestCase_mpfi_atanh())

class TestCase_mpfi_bounded_p(unittest.TestCase):
    """mpfi_bounded_p"""

suite.addTest(TestCase_mpfi_bounded_p())

class TestCase_mpfi_cbrt(unittest.TestCase):
    """mpfi_cbrt"""

suite.addTest(TestCase_mpfi_cbrt())

class TestCase_mpfi_cos(unittest.TestCase):
    """mpfi_cos"""

suite.addTest(TestCase_mpfi_cos())

class TestCase_mpfi_cosh(unittest.TestCase):
    """mpfi_cosh"""

suite.addTest(TestCase_mpfi_cosh())

class TestCase_mpfi_cot(unittest.TestCase):
    """mpfi_cot"""

suite.addTest(TestCase_mpfi_cot())

class TestCase_mpfi_coth(unittest.TestCase):
    """mpfi_coth"""

suite.addTest(TestCase_mpfi_coth())

class TestCase_mpfi_csc(unittest.TestCase):
    """mpfi_csc"""

suite.addTest(TestCase_mpfi_csc())

class TestCase_mpfi_csch(unittest.TestCase):
    """mpfi_csch"""

suite.addTest(TestCase_mpfi_csch())

class TestCase_mpfi_d_div(unittest.TestCase):
    """mpfi_d_div"""
    # special values
    def test_396(self):
        self.assertEqual(Interval(float.fromhex('-0x170ef54646d496p-107'), float.fromhex('-0x170ef54646d496p-107')) / Interval(-inf, -7.0), Interval(0.0, float.fromhex('0x1a5a3ce29a1787p-110')))
    def test_397(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-inf, -7.0), Interval(0.0, 0.0))
    def test_398(self):
        self.assertEqual(Interval(float.fromhex('0x170ef54646d496p-107'), float.fromhex('0x170ef54646d496p-107')) / Interval(-inf, -7.0), Interval(float.fromhex('-0x1a5a3ce29a1787p-110'), 0.0))
    def test_399(self):
        self.assertEqual(Interval(float.fromhex('-0x170ef54646d497p-106'), float.fromhex('-0x170ef54646d497p-106')) / Interval(-inf, 0.0), Interval(0.0, inf))
    def test_400(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-inf, 0.0), Interval(0.0, 0.0))
    def test_401(self):
        self.assertEqual(Interval(float.fromhex('0x170ef54646d497p-106'), float.fromhex('0x170ef54646d497p-106')) / Interval(-inf, 0.0), Interval(-inf, 0.0))
    def test_402(self):
        self.assertEqual(Interval(float.fromhex('-0x16345785d8a00000p0'), float.fromhex('-0x16345785d8a00000p0')) / Interval(-inf, 8.0), Interval(-math.inf, math.inf))
    def test_403(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-inf, 8.0), Interval(0.0, 0.0))
    def test_404(self):
        self.assertEqual(Interval(float.fromhex('0x16345785d8a00000p0'), float.fromhex('0x16345785d8a00000p0')) / Interval(-inf, 8.0), Interval(-math.inf, math.inf))
    def test_405(self):
        self.assertEqual(Interval(float.fromhex('-0x170ef54646d497p-105'), float.fromhex('-0x170ef54646d497p-105')) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_406(self):
        self.assertEqual(Interval(0.0e-17, 0.0e-17) / Interval(-math.inf, math.inf), Interval(0.0, 0.0))
    def test_407(self):
        self.assertEqual(Interval(float.fromhex('+0x170ef54646d497p-105'), float.fromhex('+0x170ef54646d497p-105')) / Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_408(self):
        self.assertEqual(Interval(float.fromhex('-0x170ef54646d497p-109'), float.fromhex('-0x170ef54646d497p-109')) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_409(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_410(self):
        self.assertEqual(Interval(float.fromhex('0x170ef54646d497p-109'), float.fromhex('0x170ef54646d497p-109')) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_411(self):
        self.assertEqual(Interval(float.fromhex('-0x114b37f4b51f71p-107'), float.fromhex('-0x114b37f4b51f71p-107')) / Interval(0.0, 7.0), Interval(-inf, float.fromhex('-0x13c3ada9f391a5p-110')))
    def test_412(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(0.0, 7.0), Interval(0.0, 0.0))
    def test_413(self):
        self.assertEqual(Interval(float.fromhex('0x114b37f4b51f71p-107'), float.fromhex('0x114b37f4b51f71p-107')) / Interval(0.0, 7.0), Interval(float.fromhex('0x13c3ada9f391a5p-110'), inf))
    def test_414(self):
        self.assertEqual(Interval(float.fromhex('-0x50b45a75f7e81p-104'), float.fromhex('-0x50b45a75f7e81p-104')) / Interval(0.0, inf), Interval(-inf, 0.0))
    def test_415(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(0.0, inf), Interval(0.0, 0.0))
    def test_416(self):
        self.assertEqual(Interval(float.fromhex('0x142d169d7dfa03p-106'), float.fromhex('0x142d169d7dfa03p-106')) / Interval(0.0, inf), Interval(0.0, inf))
    # regular values
    def test_417(self):
        self.assertEqual(Interval(-2.5, -2.5) / Interval(-8.0, 8.0), Interval(-math.inf, math.inf))
    def test_418(self):
        self.assertEqual(Interval(-2.5, -2.5) / Interval(-8.0, -5.0), Interval(float.fromhex('0x5p-4'), 0.5))
    def test_419(self):
        self.assertEqual(Interval(-2.5, -2.5) / Interval(25.0, 40.0), Interval(float.fromhex('-0x1999999999999ap-56'), float.fromhex('-0x1p-4')))
    def test_420(self):
        self.assertEqual(Interval(-2.5, -2.5) / Interval(-16.0, -7.0), Interval(float.fromhex('0x5p-5'), float.fromhex('0x16db6db6db6db7p-54')))
    def test_421(self):
        self.assertEqual(Interval(-2.5, -2.5) / Interval(11.0, 143.0), Interval(float.fromhex('-0x1d1745d1745d18p-55'), float.fromhex('-0x11e6efe35b4cfap-58')))
    def test_422(self):
        self.assertEqual(Interval(33.125, 33.125) / Interval(8.28125, 530.0), Interval(float.fromhex('0x1p-4'), 4.0))
    def test_423(self):
        self.assertEqual(Interval(33.125, 33.125) / Interval(-530.0, -496.875), Interval(float.fromhex('-0x11111111111112p-56'), float.fromhex('-0x1p-4')))
    def test_424(self):
        self.assertEqual(Interval(33.125, 33.125) / Interval(54.0, 265.0), Interval(0.125, float.fromhex('0x13a12f684bda13p-53')))
    def test_425(self):
        self.assertEqual(Interval(33.125, 33.125) / Interval(52.0, 54.0), Interval(float.fromhex('0x13a12f684bda12p-53'), float.fromhex('0x14627627627628p-53')))
suite.addTest(TestCase_mpfi_d_div())

class TestCase_mpfi_diam_abs(unittest.TestCase):
    """mpfi_diam_abs"""
    # special values
    def test_426(self):
        self.assertEqual(Interval(-inf, -8.0).wid, inf)
    def test_427(self):
        self.assertEqual(Interval(-inf, 0.0).wid, inf)
    def test_428(self):
        self.assertEqual(Interval(-inf, 5.0).wid, inf)
    def test_429(self):
        self.assertEqual(Interval(-math.inf, math.inf).wid, inf)
    def test_430(self):
        self.assertEqual(Interval(-inf, 0.0).wid, inf)
    def test_431(self):
        self.assertEqual(Interval(-8.0, 0.0).wid, +8)
    def test_432(self):
        self.assertEqual(Interval(0.0, 0.0).wid, -0)
    def test_433(self):
        self.assertEqual(Interval(0.0, 5.0).wid, +5)
    def test_434(self):
        self.assertEqual(Interval(0.0, inf).wid, inf)
    # regular values
    def test_435(self):
        self.assertEqual(Interval(-34.0, -17.0).wid, 17)
suite.addTest(TestCase_mpfi_diam_abs())

class TestCase_mpfi_div(unittest.TestCase):
    """mpfi_div"""
    # special values
    def test_436(self):
        self.assertEqual(Interval(-inf, -7.0) / Interval(-1.0, +8.0), Interval(-math.inf, math.inf))
    def test_437(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(+8.0, inf), Interval(-inf, 0.0))
    def test_438(self):
        self.assertEqual(Interval(-inf, +8.0) / Interval(0.0, +8.0), Interval(-math.inf, math.inf))
    def test_439(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(0.0, +8.0), Interval(-math.inf, math.inf))
    def test_440(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-inf, -7.0), Interval(0.0, 0.0))
    def test_441(self):
        self.assertEqual(Interval(0.0, +8.0) / Interval(-7.0, 0.0), Interval(-inf, 0.0))
    def test_442(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(0.0, +8.0), Interval(0.0, 0.0))
    def test_443(self):
        self.assertEqual(Interval(0.0, inf) / Interval(0.0, +8.0), Interval(0.0, inf))
    def test_444(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(+8.0, inf), Interval(0.0, 0.0))
    def test_445(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(-math.inf, math.inf), Interval(0.0, 0.0))
    def test_446(self):
        self.assertEqual(Interval(0.0, +8.0) / Interval(-7.0, +8.0), Interval(-math.inf, math.inf))
    def test_447(self):
        self.assertEqual(Interval(0.0, inf) / Interval(0.0, +8.0), Interval(0.0, inf))
    # regular value
    def test_448(self):
        self.assertEqual(Interval(float.fromhex('-0x75bcd15p0'), float.fromhex('-0x754ep0')) / Interval(float.fromhex('-0x11ep0'), float.fromhex('-0x9p0')), Interval(float.fromhex('0x69p0'), float.fromhex('0xd14fadp0')))
    def test_449(self):
        self.assertEqual(Interval(float.fromhex('-0x75bcd15p0'), float.fromhex('-0x1.489c07caba163p-4')) / Interval(float.fromhex('-0x2.e8e36e560704ap+4'), float.fromhex('-0x9p0')), Interval(float.fromhex('0x7.0ef61537b1704p-12'), float.fromhex('0xd14fadp0')))
    def test_450(self):
        self.assertEqual(Interval(float.fromhex('-0x1.02f0415f9f596p+0'), float.fromhex('-0x754ep-16')) / Interval(float.fromhex('-0x11ep0'), float.fromhex('-0x7.62ce64fbacd2cp-8')), Interval(float.fromhex('0x69p-16'), float.fromhex('0x2.30ee5eef9c36cp+4')))
    def test_451(self):
        self.assertEqual(Interval(float.fromhex('-0x1.02f0415f9f596p+0'), float.fromhex('-0x1.489c07caba163p-4')) / Interval(float.fromhex('-0x2.e8e36e560704ap+0'), float.fromhex('-0x7.62ce64fbacd2cp-8')), Interval(float.fromhex('0x7.0ef61537b1704p-8'), float.fromhex('0x2.30ee5eef9c36cp+4')))
    def test_452(self):
        self.assertEqual(Interval(float.fromhex('-0xacbp+256'), float.fromhex('-0x6f9p0')) / Interval(float.fromhex('-0x7p0'), 0.0), Interval(float.fromhex('0xffp0'), inf))
    def test_453(self):
        self.assertEqual(Interval(float.fromhex('-0x100p0'), float.fromhex('-0xe.bb80d0a0824ep-4')) / Interval(float.fromhex('-0x1.7c6d760a831fap+0'), 0.0), Interval(float.fromhex('0x9.e9f24790445fp-4'), inf))
    def test_454(self):
        self.assertEqual(Interval(float.fromhex('-0x1.25f2d73472753p+0'), float.fromhex('-0x9.9a19fd3c1fc18p-4')) / Interval(float.fromhex('-0x9.3b0c8074ccc18p-4'), float.fromhex('+0x4.788df5d72af78p-4')), Interval(-math.inf, math.inf))
    def test_455(self):
        self.assertEqual(Interval(-100.0, -15.0) / Interval(0.0, +3.0), Interval(-inf, -5.0))
    def test_456(self):
        self.assertEqual(Interval(-2.0, float.fromhex('-0x1.25f2d73472753p+0')) / Interval(0.0, float.fromhex('+0x9.3b0c8074ccc18p-4')), Interval(-inf, float.fromhex('-0x1.fd8457415f917p+0')))
    def test_457(self):
        self.assertEqual(Interval(float.fromhex('-0x123456789p0'), float.fromhex('-0x754ep+4')) / Interval(float.fromhex('0x40bp0'), float.fromhex('0x11ep+4')), Interval(float.fromhex('-0x480b3bp0'), float.fromhex('-0x69p0')))
    def test_458(self):
        self.assertEqual(Interval(float.fromhex('-0xd.67775e4b8588p-4'), float.fromhex('-0x754ep-53')) / Interval(float.fromhex('0x4.887091874ffc8p+0'), float.fromhex('0x11ep+201')), Interval(float.fromhex('-0x2.f5008d2df94ccp-4'), float.fromhex('-0x69p-254')))
    def test_459(self):
        self.assertEqual(Interval(float.fromhex('-0x123456789p0'), float.fromhex('-0x1.b0a62934c76e9p+0')) / Interval(float.fromhex('0x40bp-17'), float.fromhex('0x2.761ec797697a4p-4')), Interval(float.fromhex('-0x480b3bp+17'), float.fromhex('-0xa.fc5e7338f3e4p+0')))
    def test_460(self):
        self.assertEqual(Interval(float.fromhex('-0xd.67775e4b8588p+0'), float.fromhex('-0x1.b0a62934c76e9p+0')) / Interval(float.fromhex('0x4.887091874ffc8p-4'), float.fromhex('0x2.761ec797697a4p+4')), Interval(float.fromhex('-0x2.f5008d2df94ccp+4'), float.fromhex('-0xa.fc5e7338f3e4p-8')))
    def test_461(self):
        self.assertEqual(Interval(float.fromhex('-0x75bcd15p0'), 0.0) / Interval(float.fromhex('-0x90p0'), float.fromhex('-0x9p0')), Interval(0.0, float.fromhex('0xd14fadp0')))
    def test_462(self):
        self.assertEqual(Interval(float.fromhex('-0x1.4298b2138f2a7p-4'), 0.0) / Interval(float.fromhex('-0x1p-8'), float.fromhex('-0xf.5e4900c9c19fp-12')), Interval(0.0, float.fromhex('0x1.4fdb41a33d6cep+4')))
    def test_463(self):
        self.assertEqual(Interval(float.fromhex('-0xeeeeeeeeep0'), 0.0) / Interval(float.fromhex('-0xaaaaaaaaap0'), 0.0), Interval(0.0, inf))
    def test_464(self):
        self.assertEqual(Interval(float.fromhex('-0x1.25f2d73472753p+0'), 0.0) / Interval(float.fromhex('-0x9.3b0c8074ccc18p-4'), float.fromhex('+0x4.788df5d72af78p-4')), Interval(-math.inf, math.inf))
    def test_465(self):
        self.assertEqual(Interval(float.fromhex('-0xeeeeeeeeep0'), 0.0) / Interval(0.0, float.fromhex('+0x3p0')), Interval(-inf, 0.0))
    def test_466(self):
        self.assertEqual(Interval(float.fromhex('-0x75bcd15p0'), 0.0) / Interval(float.fromhex('0x9p0'), float.fromhex('0x90p0')), Interval(float.fromhex('-0xd14fadp0'), 0.0))
    def test_467(self):
        self.assertEqual(Interval(float.fromhex('-0x1.4298b2138f2a7p-4'), 0.0) / Interval(float.fromhex('0xf.5e4900c9c19fp-12'), float.fromhex('0x9p0')), Interval(float.fromhex('-0x1.4fdb41a33d6cep+4'), 0.0))
    def test_468(self):
        self.assertEqual(Interval(float.fromhex('-0x75bcd15p0'), float.fromhex('0xa680p0')) / Interval(float.fromhex('-0xaf6p0'), float.fromhex('-0x9p0')), Interval(float.fromhex('-0x1280p0'), float.fromhex('0xd14fadp0')))
    def test_469(self):
        self.assertEqual(Interval(float.fromhex('-0x12p0'), float.fromhex('0x10p0')) / Interval(float.fromhex('-0xbbbbbbbbbbp0'), float.fromhex('-0x9p0')), Interval(float.fromhex('-0x1.c71c71c71c71dp0'), 2.0))
    def test_470(self):
        self.assertEqual(Interval(float.fromhex('-0x1p0'), float.fromhex('0x754ep-16')) / Interval(float.fromhex('-0xccccccccccp0'), float.fromhex('-0x11ep0')), Interval(float.fromhex('-0x69p-16'), float.fromhex('0xe.525982af70c9p-12')))
    def test_471(self):
        self.assertEqual(Interval(float.fromhex('-0xb.5b90b4d32136p-4'), float.fromhex('0x6.e694ac6767394p+0')) / Interval(float.fromhex('-0xdddddddddddp0'), float.fromhex('-0xc.f459be9e80108p-4')), Interval(float.fromhex('-0x8.85e40b3c3f63p+0'), float.fromhex('0xe.071cbfa1de788p-4')))
    def test_472(self):
        self.assertEqual(Interval(float.fromhex('-0xacbp+256'), float.fromhex('0x6f9p0')) / Interval(float.fromhex('-0x7p0'), 0.0), Interval(-math.inf, math.inf))
    def test_473(self):
        self.assertEqual(Interval(float.fromhex('-0x1.25f2d73472753p+0'), float.fromhex('+0x9.9a19fd3c1fc18p-4')) / Interval(float.fromhex('-0x9.3b0c8074ccc18p-4'), float.fromhex('+0x4.788df5d72af78p-4')), Interval(-math.inf, math.inf))
    def test_474(self):
        self.assertEqual(Interval(0.0, +15.0) / Interval(-3.0, +3.0), Interval(-math.inf, math.inf))
    def test_475(self):
        self.assertEqual(Interval(float.fromhex('-0x754ep0'), float.fromhex('0xd0e9dc4p+12')) / Interval(float.fromhex('0x11ep0'), float.fromhex('0xbbbp0')), Interval(float.fromhex('-0x69p0'), float.fromhex('0xbaffep+12')))
    def test_476(self):
        self.assertEqual(Interval(float.fromhex('-0x10p0'), float.fromhex('0xd0e9dc4p+12')) / Interval(float.fromhex('0x11ep0'), float.fromhex('0xbbbp0')), Interval(float.fromhex('-0xe.525982af70c9p-8'), float.fromhex('0xbaffep+12')))
    def test_477(self):
        self.assertEqual(Interval(float.fromhex('-0x754ep0'), float.fromhex('0x1p+10')) / Interval(float.fromhex('0x11ep0'), float.fromhex('0xbbbp0')), Interval(float.fromhex('-0x69p0'), float.fromhex('0xe.525982af70c9p-2')))
    def test_478(self):
        self.assertEqual(Interval(float.fromhex('-0x1.18333622af827p+0'), float.fromhex('0x2.14b836907297p+0')) / Interval(float.fromhex('0x1.263147d1f4bcbp+0'), float.fromhex('0x111p0')), Interval(float.fromhex('-0xf.3d2f5db8ec728p-4'), float.fromhex('0x1.cf8fa732de129p+0')))
    def test_479(self):
        self.assertEqual(Interval(0.0, float.fromhex('0x75bcd15p0')) / Interval(float.fromhex('-0xap0'), float.fromhex('-0x9p0')), Interval(float.fromhex('-0xd14fadp0'), 0.0))
    def test_480(self):
        self.assertEqual(Interval(0.0, float.fromhex('0x1.acbf1702af6edp+0')) / Interval(float.fromhex('-0x0.fp0'), float.fromhex('-0xe.3d7a59e2bdacp-4')), Interval(float.fromhex('-0x1.e1bb896bfda07p+0'), 0.0))
    def test_481(self):
        self.assertEqual(Interval(0.0, float.fromhex('0xap0')) / Interval(float.fromhex('-0x9p0'), 0.0), Interval(-inf, 0.0))
    def test_482(self):
        self.assertEqual(Interval(0.0, float.fromhex('0xap0')) / Interval(-1.0, +1.0), Interval(-math.inf, math.inf))
    def test_483(self):
        self.assertEqual(Interval(0.0, float.fromhex('0x75bcd15p0')) / Interval(float.fromhex('+0x9p0'), float.fromhex('+0xap0')), Interval(0.0, float.fromhex('0xd14fadp0')))
    def test_484(self):
        self.assertEqual(Interval(0.0, float.fromhex('0x1.5f6b03dc8c66fp+0')) / Interval(float.fromhex('+0x2.39ad24e812dcep+0'), float.fromhex('0xap0')), Interval(0.0, float.fromhex('0x9.deb65b02baep-4')))
    def test_485(self):
        self.assertEqual(Interval(float.fromhex('0x754ep0'), float.fromhex('0x75bcd15p0')) / Interval(float.fromhex('-0x11ep0'), float.fromhex('-0x9p0')), Interval(float.fromhex('-0xd14fadp0'), float.fromhex('-0x69p0')))
    def test_486(self):
        self.assertEqual(Interval(float.fromhex('0x754ep-16'), float.fromhex('0x1.008a3accc766dp+4')) / Interval(float.fromhex('-0x11ep0'), float.fromhex('-0x2.497403b31d32ap+0')), Interval(float.fromhex('-0x7.02d3edfbc8b6p+0'), float.fromhex('-0x69p-16')))
    def test_487(self):
        self.assertEqual(Interval(float.fromhex('0x9.ac412ff1f1478p-4'), float.fromhex('0x75bcd15p0')) / Interval(float.fromhex('-0x1.5232c83a0e726p+4'), float.fromhex('-0x9p0')), Interval(float.fromhex('-0xd14fadp0'), float.fromhex('-0x7.52680a49e5d68p-8')))
    def test_488(self):
        self.assertEqual(Interval(float.fromhex('0xe.1552a314d629p-4'), float.fromhex('0x1.064c5adfd0042p+0')) / Interval(float.fromhex('-0x5.0d4d319a50b04p-4'), float.fromhex('-0x2.d8f51df1e322ep-4')), Interval(float.fromhex('-0x5.c1d97d57d81ccp+0'), float.fromhex('-0x2.c9a600c455f5ap+0')))
    def test_489(self):
        self.assertEqual(Interval(float.fromhex('0x754ep0'), float.fromhex('0xeeeep0')) / Interval(float.fromhex('-0x11ep0'), 0.0), Interval(-inf, float.fromhex('-0x69p0')))
    def test_490(self):
        self.assertEqual(Interval(float.fromhex('0x1.a9016514490e6p-4'), float.fromhex('0xeeeep0')) / Interval(float.fromhex('-0xe.316e87be0b24p-4'), 0.0), Interval(-inf, float.fromhex('-0x1.df1cc82e6a583p-4')))
    def test_491(self):
        self.assertEqual(Interval(5.0, 6.0) / Interval(float.fromhex('-0x5.0d4d319a50b04p-4'), float.fromhex('0x2.d8f51df1e322ep-4')), Interval(-math.inf, math.inf))
    def test_492(self):
        self.assertEqual(Interval(float.fromhex('0x754ep0'), float.fromhex('+0xeeeeep0')) / Interval(0.0, float.fromhex('+0x11ep0')), Interval(float.fromhex('0x69p0'), inf))
    def test_493(self):
        self.assertEqual(Interval(float.fromhex('0x1.7f03f2a978865p+0'), float.fromhex('0xeeeeep0')) / Interval(0.0, float.fromhex('0x1.48b08624606b9p+0')), Interval(float.fromhex('0x1.2a4fcda56843p+0'), inf))
    def test_494(self):
        self.assertEqual(Interval(float.fromhex('0x5efc1492p0'), float.fromhex('0x1ba2dc763p0')) / Interval(float.fromhex('0x2fdd1fp0'), float.fromhex('0x889b71p0')), Interval(float.fromhex('0xb2p0'), float.fromhex('0x93dp0')))
    def test_495(self):
        self.assertEqual(Interval(float.fromhex('0x1.d7c06f9ff0706p-8'), float.fromhex('0x1ba2dc763p0')) / Interval(float.fromhex('0x2fdd1fp-20'), float.fromhex('0xe.3d7a59e2bdacp+0')), Interval(float.fromhex('0x2.120d75be74b54p-12'), float.fromhex('0x93dp+20')))
    def test_496(self):
        self.assertEqual(Interval(float.fromhex('0x5.efc1492p-4'), float.fromhex('0x1.008a3accc766dp+0')) / Interval(float.fromhex('0x2.497403b31d32ap+0'), float.fromhex('0x8.89b71p+0')), Interval(float.fromhex('0xb.2p-8'), float.fromhex('0x7.02d3edfbc8b6p-4')))
    def test_497(self):
        self.assertEqual(Interval(float.fromhex('0x8.440e7d65be6bp-8'), float.fromhex('0x3.99982e9eae09ep+0')) / Interval(float.fromhex('0x8.29fa8d0659e48p-4'), float.fromhex('0xc.13d2fd762e4a8p-4')), Interval(float.fromhex('0xa.f3518768b206p-8'), float.fromhex('0x7.0e2acad54859cp+0')))
suite.addTest(TestCase_mpfi_div())

class TestCase_mpfi_div_d(unittest.TestCase):
    """mpfi_div_d"""
    # special values
    def test_498(self):
        self.assertEqual(Interval(-inf, -7.0) / Interval(-7.0, -7.0), Interval(1.0, inf))
    def test_499(self):
        self.assertEqual(Interval(-inf, -7.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_500(self):
        self.assertEqual(Interval(-inf, -7.0) / Interval(7.0, 7.0), Interval(-inf, -1.0))
    def test_501(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(float.fromhex('-0x170ef54646d497p-106'), float.fromhex('-0x170ef54646d497p-106')), Interval(0.0, inf))
    def test_502(self):
        self.assertEqual(Interval(-inf, 0.0) / Interval(float.fromhex('0x170ef54646d497p-106'), float.fromhex('0x170ef54646d497p-106')), Interval(-inf, 0.0))
    def test_503(self):
        self.assertEqual(Interval(-inf, 8.0) / Interval(-3.0, -3.0), Interval(float.fromhex('-0x15555555555556p-51'), inf))
    def test_504(self):
        self.assertEqual(Interval(-inf, 8.0) / Interval(0.0, 0.0), Interval(ip.nan, ip.nan))
    def test_505(self):
        self.assertEqual(Interval(-inf, 8.0) / Interval(3.0, 3.0), Interval(-inf, float.fromhex('0x15555555555556p-51')))
    def test_506(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(float.fromhex('-0x170ef54646d497p-105'), float.fromhex('-0x170ef54646d497p-105')), Interval(-math.inf, math.inf))
    def test_507(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(0.0e-17, 0.0e-17), Interval(ip.nan, ip.nan))
    def test_508(self):
        self.assertEqual(Interval(-math.inf, math.inf) / Interval(float.fromhex('+0x170ef54646d497p-105'), float.fromhex('+0x170ef54646d497p-105')), Interval(-math.inf, math.inf))
    def test_509(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(float.fromhex('-0x170ef54646d497p-109'), float.fromhex('-0x170ef54646d497p-109')), Interval(0.0, 0.0))
    def test_510(self):
        self.assertEqual(Interval(0.0, 0.0) / Interval(float.fromhex('0x170ef54646d497p-109'), float.fromhex('0x170ef54646d497p-109')), Interval(0.0, 0.0))
    def test_511(self):
        self.assertEqual(Interval(0.0, 8.0) / Interval(float.fromhex('-0x114b37f4b51f71p-107'), float.fromhex('-0x114b37f4b51f71p-107')), Interval(float.fromhex('-0x1d9b1f5d20d556p+5'), 0.0))
    def test_512(self):
        self.assertEqual(Interval(0.0, 8.0) / Interval(float.fromhex('0x114b37f4b51f71p-107'), float.fromhex('0x114b37f4b51f71p-107')), Interval(0.0, float.fromhex('0x1d9b1f5d20d556p+5')))
    def test_513(self):
        self.assertEqual(Interval(0.0, inf) / Interval(float.fromhex('-0x50b45a75f7e81p-104'), float.fromhex('-0x50b45a75f7e81p-104')), Interval(-inf, 0.0))
    def test_514(self):
        self.assertEqual(Interval(0.0, inf) / Interval(float.fromhex('0x142d169d7dfa03p-106'), float.fromhex('0x142d169d7dfa03p-106')), Interval(0.0, inf))
    # regular values
    def test_515(self):
        self.assertEqual(Interval(float.fromhex('-0x10000000000001p-20'), float.fromhex('-0x10000000000001p-53')) / Interval(-1.0, -1.0), Interval(float.fromhex('0x10000000000001p-53'), float.fromhex('0x10000000000001p-20')))
    def test_516(self):
        self.assertEqual(Interval(float.fromhex('-0x10000000000002p-20'), float.fromhex('-0x10000000000001p-53')) / Interval(float.fromhex('0x10000000000001p-53'), float.fromhex('0x10000000000001p-53')), Interval(float.fromhex('-0x10000000000001p-19'), -1.0))
    def test_517(self):
        self.assertEqual(Interval(float.fromhex('-0x10000000000001p-20'), float.fromhex('-0x10000020000001p-53')) / Interval(float.fromhex('0x10000000000001p-53'), float.fromhex('0x10000000000001p-53')), Interval(float.fromhex('-0x1p+33'), float.fromhex('-0x1000001fffffffp-52')))
    def test_518(self):
        self.assertEqual(Interval(float.fromhex('-0x10000000000002p-20'), float.fromhex('-0x10000020000001p-53')) / Interval(float.fromhex('0x10000000000001p-53'), float.fromhex('0x10000000000001p-53')), Interval(float.fromhex('-0x10000000000001p-19'), float.fromhex('-0x1000001fffffffp-52')))
    def test_519(self):
        self.assertEqual(Interval(float.fromhex('-0x123456789abcdfp-53'), float.fromhex('0x123456789abcdfp-7')) / Interval(float.fromhex('-0x123456789abcdfp0'), float.fromhex('-0x123456789abcdfp0')), Interval(float.fromhex('-0x1p-7'), float.fromhex('0x1p-53')))
    def test_520(self):
        self.assertEqual(Interval(float.fromhex('-0x123456789abcdfp-53'), float.fromhex('0x10000000000001p-53')) / Interval(float.fromhex('-0x123456789abcdfp0'), float.fromhex('-0x123456789abcdfp0')), Interval(float.fromhex('-0x1c200000000002p-106'), float.fromhex('0x1p-53')))
    def test_521(self):
        self.assertEqual(Interval(-1.0, float.fromhex('0x123456789abcdfp-7')) / Interval(float.fromhex('-0x123456789abcdfp0'), float.fromhex('-0x123456789abcdfp0')), Interval(float.fromhex('-0x1p-7'), float.fromhex('0x1c200000000001p-105')))
    def test_522(self):
        self.assertEqual(Interval(-1.0, float.fromhex('0x10000000000001p-53')) / Interval(float.fromhex('-0x123456789abcdfp0'), float.fromhex('-0x123456789abcdfp0')), Interval(float.fromhex('-0x1c200000000002p-106'), float.fromhex('0x1c200000000001p-105')))
suite.addTest(TestCase_mpfi_div_d())

class TestCase_mpfi_d_sub(unittest.TestCase):
    """mpfi_d_sub"""
    # special values
    def test_523(self):
        self.assertEqual(Interval(float.fromhex('-0x170ef54646d497p-107'), float.fromhex('-0x170ef54646d497p-107')) - Interval(-inf, -7.0), Interval(float.fromhex('0x1bffffffffffffp-50'), inf))
    def test_524(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(-inf, -7.0), Interval(7.0, inf))
    def test_525(self):
        self.assertEqual(Interval(float.fromhex('0x170ef54646d497p-107'), float.fromhex('0x170ef54646d497p-107')) - Interval(-inf, -7.0), Interval(7.0, inf))
    def test_526(self):
        self.assertEqual(Interval(float.fromhex('-0x170ef54646d497p-96'), float.fromhex('-0x170ef54646d497p-96')) - Interval(-inf, 0.0), Interval(float.fromhex('-0x170ef54646d497p-96'), inf))
    def test_527(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(-inf, 0.0), Interval(0.0, inf))
    def test_528(self):
        self.assertEqual(Interval(float.fromhex('0x170ef54646d497p-96'), float.fromhex('0x170ef54646d497p-96')) - Interval(-inf, 0.0), Interval(float.fromhex('0x170ef54646d497p-96'), inf))
    def test_529(self):
        self.assertEqual(Interval(float.fromhex('-0x16345785d8a00000p0'), float.fromhex('-0x16345785d8a00000p0')) - Interval(-inf, 8.0), Interval(float.fromhex('-0x16345785d8a00100p0'), inf))
    def test_530(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(-inf, 8.0), Interval(-8.0, inf))
    def test_531(self):
        self.assertEqual(Interval(float.fromhex('0x16345785d8a00000p0'), float.fromhex('0x16345785d8a00000p0')) - Interval(-inf, 8.0), Interval(float.fromhex('0x16345785d89fff00p0'), inf))
    def test_532(self):
        self.assertEqual(Interval(float.fromhex('-0x170ef54646d497p-105'), float.fromhex('-0x170ef54646d497p-105')) - Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_533(self):
        self.assertEqual(Interval(0.0e-17, 0.0e-17) - Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_534(self):
        self.assertEqual(Interval(float.fromhex('0x170ef54646d497p-105'), float.fromhex('0x170ef54646d497p-105')) - Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_535(self):
        self.assertEqual(Interval(float.fromhex('-0x170ef54646d497p-109'), float.fromhex('-0x170ef54646d497p-109')) - Interval(0.0, 0.0), Interval(float.fromhex('-0x170ef54646d497p-109'), float.fromhex('-0x170ef54646d497p-109')))
    def test_536(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_537(self):
        self.assertEqual(Interval(float.fromhex('0x170ef54646d497p-109'), float.fromhex('0x170ef54646d497p-109')) - Interval(0.0, 0.0), Interval(float.fromhex('0x170ef54646d497p-109'), float.fromhex('0x170ef54646d497p-109')))
    def test_538(self):
        self.assertEqual(Interval(float.fromhex('-0x114b37f4b51f71p-107'), float.fromhex('-0x114b37f4b51f71p-107')) - Interval(0.0, 8.0), Interval(float.fromhex('-0x10000000000001p-49'), float.fromhex('-0x114b37f4b51f71p-107')))
    def test_539(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(0.0, 8.0), Interval(-8.0, 0.0))
    def test_540(self):
        self.assertEqual(Interval(float.fromhex('0x114b37f4b51f71p-107'), float.fromhex('0x114b37f4b51f71p-107')) - Interval(0.0, 8.0), Interval(-8.0, float.fromhex('0x114b37f4b51f71p-107')))
    def test_541(self):
        self.assertEqual(Interval(float.fromhex('-0x50b45a75f7e81p-104'), float.fromhex('-0x50b45a75f7e81p-104')) - Interval(0.0, inf), Interval(-inf, float.fromhex('-0x50b45a75f7e81p-104')))
    def test_542(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(0.0, inf), Interval(-inf, 0.0))
    def test_543(self):
        self.assertEqual(Interval(float.fromhex('-0x142d169d7dfa03p-106'), float.fromhex('-0x142d169d7dfa03p-106')) - Interval(0.0, inf), Interval(-inf, float.fromhex('-0x142d169d7dfa03p-106')))
    # regular values
    def test_544(self):
        self.assertEqual(Interval(float.fromhex('-0xfb53d14aa9c2fp-47'), float.fromhex('-0xfb53d14aa9c2fp-47')) - Interval(17.0, 32.0), Interval(float.fromhex('-0x1fb53d14aa9c2fp-47'), float.fromhex('-0x18353d14aa9c2fp-47')))
    def test_545(self):
        self.assertEqual(Interval(float.fromhex('0xfb53d14aa9c2fp-47'), float.fromhex('0xfb53d14aa9c2fp-47')) - Interval(17.0, float.fromhex('0xfb53d14aa9c2fp-47')), Interval(0.0, float.fromhex('0x7353d14aa9c2fp-47')))
    def test_546(self):
        self.assertEqual(Interval(float.fromhex('0xfb53d14aa9c2fp-48'), float.fromhex('0xfb53d14aa9c2fp-48')) - Interval(float.fromhex('0xfb53d14aa9c2fp-48'), 32.0), Interval(float.fromhex('-0x104ac2eb5563d1p-48'), 0.0))
    def test_547(self):
        self.assertEqual(Interval(3.5, 3.5) - Interval(float.fromhex('-0x123456789abcdfp-4'), float.fromhex('-0x123456789abcdfp-48')), Interval(float.fromhex('0x15b456789abcdfp-48'), float.fromhex('0x123456789abd17p-4')))
    def test_548(self):
        self.assertEqual(Interval(3.5, 3.5) - Interval(float.fromhex('-0x123456789abcdfp-4'), float.fromhex('-0x123456789abcdfp-56')), Interval(float.fromhex('0x3923456789abcdp-52'), float.fromhex('0x123456789abd17p-4')))
    def test_549(self):
        self.assertEqual(Interval(256.5, 256.5) - Interval(float.fromhex('-0x123456789abcdfp-52'), float.fromhex('0xffp0')), Interval(float.fromhex('0x18p-4'), float.fromhex('0x101a3456789abdp-44')))
    def test_550(self):
        self.assertEqual(Interval(4097.5, 4097.5) - Interval(float.fromhex('0x1p-550'), float.fromhex('0x1fffffffffffffp-52')), Interval(float.fromhex('0xfff8p-4'), float.fromhex('0x10018p-4')))
    def test_551(self):
        self.assertEqual(Interval(-3.5, -3.5) - Interval(float.fromhex('-0x123456789abcdfp-4'), float.fromhex('-0x123456789abcdfp-48')), Interval(float.fromhex('0xeb456789abcdfp-48'), float.fromhex('0x123456789abca7p-4')))
    def test_552(self):
        self.assertEqual(Interval(-3.5, -3.5) - Interval(float.fromhex('-0x123456789abcdfp-4'), float.fromhex('-0x123456789abcdfp-56')), Interval(float.fromhex('-0x36dcba98765434p-52'), float.fromhex('0x123456789abca7p-4')))
    def test_553(self):
        self.assertEqual(Interval(-256.5, -256.5) - Interval(float.fromhex('-0x123456789abcdfp-52'), float.fromhex('0xffp0')), Interval(float.fromhex('-0x1ff8p-4'), float.fromhex('-0xff5cba9876543p-44')))
    def test_554(self):
        self.assertEqual(Interval(-4097.5, -4097.5) - Interval(float.fromhex('0x1p-550'), float.fromhex('0x1fffffffffffffp-52')), Interval(float.fromhex('-0x10038p-4'), float.fromhex('-0x10018p-4')))
suite.addTest(TestCase_mpfi_d_sub())

class TestCase_mpfi_exp(unittest.TestCase):
    """mpfi_exp"""

suite.addTest(TestCase_mpfi_exp())

class TestCase_mpfi_exp2(unittest.TestCase):
    """mpfi_exp2"""

suite.addTest(TestCase_mpfi_exp2())

class TestCase_mpfi_expm1(unittest.TestCase):
    """mpfi_expm1"""

suite.addTest(TestCase_mpfi_expm1())

class TestCase_mpfi_hypot(unittest.TestCase):
    """mpfi_hypot"""

suite.addTest(TestCase_mpfi_hypot())

class TestCase_mpfi_intersect(unittest.TestCase):
    """mpfi_intersect"""

suite.addTest(TestCase_mpfi_intersect())

class TestCase_mpfi_inv(unittest.TestCase):
    """mpfi_inv"""

suite.addTest(TestCase_mpfi_inv())

class TestCase_mpfi_is_neg(unittest.TestCase):
    """mpfi_is_neg"""

suite.addTest(TestCase_mpfi_is_neg())

class TestCase_mpfi_is_nonneg(unittest.TestCase):
    """mpfi_is_nonneg"""

suite.addTest(TestCase_mpfi_is_nonneg())

class TestCase_mpfi_is_nonpos(unittest.TestCase):
    """mpfi_is_nonpos"""

suite.addTest(TestCase_mpfi_is_nonpos())

class TestCase_mpfi_is_pos(unittest.TestCase):
    """mpfi_is_pos"""

suite.addTest(TestCase_mpfi_is_pos())

class TestCase_mpfi_is_strictly_neg(unittest.TestCase):
    """mpfi_is_strictly_neg"""

suite.addTest(TestCase_mpfi_is_strictly_neg())

class TestCase_mpfi_is_strictly_pos(unittest.TestCase):
    """mpfi_is_strictly_pos"""

suite.addTest(TestCase_mpfi_is_strictly_pos())

class TestCase_mpfi_log(unittest.TestCase):
    """mpfi_log"""

suite.addTest(TestCase_mpfi_log())

class TestCase_mpfi_log1p(unittest.TestCase):
    """mpfi_log1p"""

suite.addTest(TestCase_mpfi_log1p())

class TestCase_mpfi_log2(unittest.TestCase):
    """mpfi_log2"""

suite.addTest(TestCase_mpfi_log2())

class TestCase_mpfi_log10(unittest.TestCase):
    """mpfi_log10"""

suite.addTest(TestCase_mpfi_log10())

class TestCase_mpfi_mag(unittest.TestCase):
    """mpfi_mag"""

suite.addTest(TestCase_mpfi_mag())

class TestCase_mpfi_mid(unittest.TestCase):
    """mpfi_mid"""
    # special values
    def test_767(self):
        self.assertEqual(Interval(-8.0, 0.0).mid, -4)
    def test_768(self):
        self.assertEqual(Interval(0.0, 0.0).mid, +0)
    def test_769(self):
        self.assertEqual(Interval(0.0, 5.0).mid, +2.5)
    # regular values
    def test_770(self):
        self.assertEqual(Interval(-34.0, -17.0).mid, float.fromhex('-0x33p-1'))
    def test_771(self):
        self.assertEqual(Interval(-34.0, 17.0).mid, -8.5)
    def test_772(self):
        self.assertEqual(Interval(0.0, float.fromhex('+0x123456789abcdp-2')).mid, float.fromhex('+0x123456789abcdp-3'))
    def test_773(self):
        self.assertEqual(Interval(float.fromhex('0x1921fb54442d18p-51'), float.fromhex('0x1921fb54442d19p-51')).mid, float.fromhex('0x1921fb54442d18p-51'))
    def test_774(self):
        self.assertEqual(Interval(float.fromhex('-0x1921fb54442d19p-51'), float.fromhex('-0x1921fb54442d18p-51')).mid, float.fromhex('-0x1921fb54442d18p-51'))
    def test_775(self):
        self.assertEqual(Interval(-4.0, float.fromhex('-0x7fffffffffffdp-51')).mid, float.fromhex('-0x27fffffffffffbp-52'))
    def test_776(self):
        self.assertEqual(Interval(-8.0, float.fromhex('-0x7fffffffffffbp-51')).mid, float.fromhex('-0x47fffffffffffbp-52'))
    def test_777(self):
        self.assertEqual(Interval(float.fromhex('-0x1fffffffffffffp-53'), 2.0).mid, 0.5)
suite.addTest(TestCase_mpfi_mid())

class TestCase_mpfi_mig(unittest.TestCase):
    """mpfi_mig"""

suite.addTest(TestCase_mpfi_mig())

class TestCase_mpfi_mul(unittest.TestCase):
    """mpfi_mul"""
    # special values
    def test_788(self):
        self.assertEqual(Interval(-inf, -7.0) * Interval(-1.0, +8.0), Interval(-math.inf, math.inf))
    def test_789(self):
        self.assertEqual(Interval(-inf, 0.0) * Interval(+8.0, inf), Interval(-inf, 0.0))
    def test_790(self):
        self.assertEqual(Interval(-inf, +8.0) * Interval(0.0, +8.0), Interval(-inf, +64.0))
    def test_791(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_792(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(0.0, +8.0), Interval(-math.inf, math.inf))
    def test_793(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(-inf, -7.0), Interval(0.0, 0.0))
    def test_794(self):
        self.assertEqual(Interval(0.0, +8.0) * Interval(-7.0, 0.0), Interval(-56.0, 0.0))
    def test_795(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(0.0, +8.0), Interval(0.0, 0.0))
    def test_796(self):
        self.assertEqual(Interval(0.0, inf) * Interval(0.0, +8.0), Interval(0.0, inf))
    def test_797(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(+8.0, inf), Interval(0.0, 0.0))
    def test_798(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(-math.inf, math.inf), Interval(0.0, 0.0))
    def test_799(self):
        self.assertEqual(Interval(0.0, +8.0) * Interval(-7.0, +8.0), Interval(-56.0, +64.0))
    def test_800(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_801(self):
        self.assertEqual(Interval(0.0, inf) * Interval(0.0, +8.0), Interval(0.0, inf))
    def test_802(self):
        self.assertEqual(Interval(-3.0, +7.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    # regular values
    def test_803(self):
        self.assertEqual(Interval(float.fromhex('-0x0dp0'), float.fromhex('-0x09p0')) * Interval(float.fromhex('-0x04p0'), float.fromhex('-0x02p0')), Interval(float.fromhex('0x12p0'), float.fromhex('0x34p0')))
    def test_804(self):
        self.assertEqual(Interval(float.fromhex('-0x0dp0'), float.fromhex('-0xd.f0e7927d247cp-4')) * Interval(float.fromhex('-0x04p0'), float.fromhex('-0xa.41084aff48f8p-8')), Interval(float.fromhex('0x8.ef3aa21dba748p-8'), float.fromhex('0x34p0')))
    def test_805(self):
        self.assertEqual(Interval(float.fromhex('-0xe.26c9e9eb67b48p-4'), float.fromhex('-0x8.237d2eb8b1178p-4')) * Interval(float.fromhex('-0x5.8c899a0706d5p-4'), float.fromhex('-0x3.344e57a37b5e8p-4')), Interval(float.fromhex('0x1.a142a930de328p-4'), float.fromhex('0x4.e86c3434cd924p-4')))
    def test_806(self):
        self.assertEqual(Interval(float.fromhex('-0x37p0'), float.fromhex('-0x07p0')) * Interval(float.fromhex('-0x01p0'), float.fromhex('0x22p0')), Interval(float.fromhex('-0x74ep0'), float.fromhex('0x37p0')))
    def test_807(self):
        self.assertEqual(Interval(float.fromhex('-0xe.063f267ed51ap-4'), float.fromhex('-0x0.33p0')) * Interval(float.fromhex('-0x01p0'), float.fromhex('0x1.777ab178b4a1ep+0')), Interval(float.fromhex('-0x1.491df346a9f15p+0'), float.fromhex('0xe.063f267ed51ap-4')))
    def test_808(self):
        self.assertEqual(Interval(float.fromhex('-0x1.cb540b71699a8p+4'), float.fromhex('-0x0.33p0')) * Interval(float.fromhex('-0x1.64dcaaa101f18p+0'), float.fromhex('0x01p0')), Interval(float.fromhex('-0x1.cb540b71699a8p+4'), float.fromhex('0x2.804cce4a3f42ep+4')))
    def test_809(self):
        self.assertEqual(Interval(float.fromhex('-0x1.cb540b71699a8p+4'), float.fromhex('-0x0.33p0')) * Interval(float.fromhex('-0x1.64dcaaa101f18p+0'), float.fromhex('0x1.eb67a1a6ef725p+4')), Interval(float.fromhex('-0x3.71b422ce817f4p+8'), float.fromhex('0x2.804cce4a3f42ep+4')))
    def test_810(self):
        self.assertEqual(Interval(float.fromhex('-0x123456789ap0'), float.fromhex('-0x01p0')) * Interval(float.fromhex('0x01p0'), float.fromhex('0x10p0')), Interval(float.fromhex('-0x123456789a0p0'), float.fromhex('-0x01p0')))
    def test_811(self):
        self.assertEqual(Interval(float.fromhex('-0xb.6c67d3a37d54p-4'), float.fromhex('-0x0.8p0')) * Interval(float.fromhex('0x02p0'), float.fromhex('0x2.0bee4e8bb3dfp+0')), Interval(float.fromhex('-0x1.7611a672948a5p+0'), float.fromhex('-0x01p0')))
    def test_812(self):
        self.assertEqual(Interval(float.fromhex('-0x04p0'), float.fromhex('-0xa.497d533c3b2ep-8')) * Interval(float.fromhex('0xb.d248df3373e68p-4'), float.fromhex('0x04p0')), Interval(float.fromhex('-0x10p0'), float.fromhex('-0x7.99b990532d434p-8')))
    def test_813(self):
        self.assertEqual(Interval(float.fromhex('-0xb.6c67d3a37d54p-4'), float.fromhex('-0xa.497d533c3b2ep-8')) * Interval(float.fromhex('0xb.d248df3373e68p-4'), float.fromhex('0x2.0bee4e8bb3dfp+0')), Interval(float.fromhex('-0x1.7611a672948a5p+0'), float.fromhex('-0x7.99b990532d434p-8')))
    def test_814(self):
        self.assertEqual(Interval(float.fromhex('-0x01p0'), float.fromhex('0x11p0')) * Interval(float.fromhex('-0x07p0'), float.fromhex('-0x04p0')), Interval(float.fromhex('-0x77p0'), float.fromhex('0x07p0')))
    def test_815(self):
        self.assertEqual(Interval(float.fromhex('-0x01p0'), float.fromhex('0xe.ca7ddfdb8572p-4')) * Interval(float.fromhex('-0x2.3b46226145234p+0'), float.fromhex('-0x0.1p0')), Interval(float.fromhex('-0x2.101b41d3d48b8p+0'), float.fromhex('0x2.3b46226145234p+0')))
    def test_816(self):
        self.assertEqual(Interval(float.fromhex('-0x1.1d069e75e8741p+8'), float.fromhex('0x01p0')) * Interval(float.fromhex('-0x2.3b46226145234p+0'), float.fromhex('-0x0.1p0')), Interval(float.fromhex('-0x2.3b46226145234p+0'), float.fromhex('0x2.7c0bd9877f404p+8')))
    def test_817(self):
        self.assertEqual(Interval(float.fromhex('-0xe.ca7ddfdb8572p-4'), float.fromhex('0x1.1d069e75e8741p+8')) * Interval(float.fromhex('-0x2.3b46226145234p+0'), float.fromhex('-0x0.1p0')), Interval(float.fromhex('-0x2.7c0bd9877f404p+8'), float.fromhex('0x2.101b41d3d48b8p+0')))
    def test_818(self):
        self.assertEqual(Interval(float.fromhex('-0x01p0'), float.fromhex('0x10p0')) * Interval(float.fromhex('-0x02p0'), float.fromhex('0x03p0')), Interval(float.fromhex('-0x20p0'), float.fromhex('0x30p0')))
    def test_819(self):
        self.assertEqual(Interval(float.fromhex('-0x01p0'), float.fromhex('0x2.db091cea593fap-4')) * Interval(float.fromhex('-0x2.6bff2625fb71cp-4'), float.fromhex('0x1p-8')), Interval(float.fromhex('-0x6.ea77a3ee43de8p-8'), float.fromhex('0x2.6bff2625fb71cp-4')))
    def test_820(self):
        self.assertEqual(Interval(float.fromhex('-0x01p0'), float.fromhex('0x6.e211fefc216ap-4')) * Interval(float.fromhex('-0x1p-4'), float.fromhex('0x1.8e3fe93a4ea52p+0')), Interval(float.fromhex('-0x1.8e3fe93a4ea52p+0'), float.fromhex('0xa.b52fe22d72788p-4')))
    def test_821(self):
        self.assertEqual(Interval(float.fromhex('-0x1.15e079e49a0ddp+0'), float.fromhex('0x1p-8')) * Interval(float.fromhex('-0x2.77fc84629a602p+0'), float.fromhex('0x8.3885932f13fp-4')), Interval(float.fromhex('-0x8.ec5de73125be8p-4'), float.fromhex('0x2.adfe651d3b19ap+0')))
    def test_822(self):
        self.assertEqual(Interval(float.fromhex('-0x07p0'), float.fromhex('0x07p0')) * Interval(float.fromhex('0x13p0'), float.fromhex('0x24p0')), Interval(float.fromhex('-0xfcp0'), float.fromhex('0xfcp0')))
    def test_823(self):
        self.assertEqual(Interval(float.fromhex('-0xa.8071f870126cp-4'), float.fromhex('0x10p0')) * Interval(float.fromhex('0x02p0'), float.fromhex('0x2.3381083e7d3b4p+0')), Interval(float.fromhex('-0x1.71dc5b5607781p+0'), float.fromhex('0x2.3381083e7d3b4p+4')))
    def test_824(self):
        self.assertEqual(Interval(float.fromhex('-0x01p0'), float.fromhex('0x1.90aa487ecf153p+0')) * Interval(float.fromhex('0x01p-53'), float.fromhex('0x1.442e2695ac81ap+0')), Interval(float.fromhex('-0x1.442e2695ac81ap+0'), float.fromhex('0x1.fb5fbebd0cbc6p+0')))
    def test_825(self):
        self.assertEqual(Interval(float.fromhex('-0x1.c40db77f2f6fcp+0'), float.fromhex('0x1.8eb70bbd94478p+0')) * Interval(float.fromhex('0x02p0'), float.fromhex('0x3.45118635235c6p+0')), Interval(float.fromhex('-0x5.c61fcad908df4p+0'), float.fromhex('0x5.17b7c49130824p+0')))
    def test_826(self):
        self.assertEqual(Interval(float.fromhex('0xcp0'), float.fromhex('0x2dp0')) * Interval(float.fromhex('-0x679p0'), float.fromhex('-0xe5p0')), Interval(float.fromhex('-0x12345p0'), float.fromhex('-0xabcp0')))
    def test_827(self):
        self.assertEqual(Interval(float.fromhex('0xcp0'), float.fromhex('0x1.1833fdcab4c4ap+10')) * Interval(float.fromhex('-0x2.4c0afc50522ccp+40'), float.fromhex('-0xe5p0')), Interval(float.fromhex('-0x2.83a3712099234p+50'), float.fromhex('-0xabcp0')))
    def test_828(self):
        self.assertEqual(Interval(float.fromhex('0xb.38f1fb0ef4308p+0'), float.fromhex('0x2dp0')) * Interval(float.fromhex('-0x679p0'), float.fromhex('-0xa.4771d7d0c604p+0')), Interval(float.fromhex('-0x12345p0'), float.fromhex('-0x7.35b3c8400ade4p+4')))
    def test_829(self):
        self.assertEqual(Interval(float.fromhex('0xf.08367984ca1cp-4'), float.fromhex('0xa.bcf6c6cbe341p+0')) * Interval(float.fromhex('-0x5.cbc445e9952c4p+0'), float.fromhex('-0x2.8ad05a7b988fep-8')), Interval(float.fromhex('-0x3.e3ce52d4a139cp+4'), float.fromhex('-0x2.637164cf2f346p-8')))
    def test_830(self):
        self.assertEqual(Interval(float.fromhex('0x01p0'), float.fromhex('0xcp0')) * Interval(float.fromhex('-0xe5p0'), float.fromhex('0x01p0')), Interval(float.fromhex('-0xabcp0'), float.fromhex('0xcp0')))
    def test_831(self):
        self.assertEqual(Interval(float.fromhex('0x123p-52'), float.fromhex('0x1.ec24910ac6aecp+0')) * Interval(float.fromhex('-0xa.a97267f56a9b8p-4'), float.fromhex('0x1p+32')), Interval(float.fromhex('-0x1.47f2dbe4ef916p+0'), float.fromhex('0x1.ec24910ac6aecp+32')))
    def test_832(self):
        self.assertEqual(Interval(float.fromhex('0x03p0'), float.fromhex('0x7.2bea531ef4098p+0')) * Interval(float.fromhex('-0x01p0'), float.fromhex('0xa.a97267f56a9b8p-4')), Interval(float.fromhex('-0x7.2bea531ef4098p+0'), float.fromhex('0x4.c765967f9468p+0')))
    def test_833(self):
        self.assertEqual(Interval(float.fromhex('0x0.3p0'), float.fromhex('0xa.a97267f56a9b8p-4')) * Interval(float.fromhex('-0x1.ec24910ac6aecp+0'), float.fromhex('0x7.2bea531ef4098p+0')), Interval(float.fromhex('-0x1.47f2dbe4ef916p+0'), float.fromhex('0x4.c765967f9468p+0')))
    def test_834(self):
        self.assertEqual(Interval(float.fromhex('0x3p0'), float.fromhex('0x7p0')) * Interval(float.fromhex('0x5p0'), float.fromhex('0xbp0')), Interval(float.fromhex('0xfp0'), float.fromhex('0x4dp0')))
    def test_835(self):
        self.assertEqual(Interval(float.fromhex('0x2.48380232f6c16p+0'), float.fromhex('0x7p0')) * Interval(float.fromhex('0x3.71cb6c53e68eep+0'), float.fromhex('0xbp0')), Interval(float.fromhex('0x7.dc58fb323ad78p+0'), float.fromhex('0x4dp0')))
    def test_836(self):
        self.assertEqual(Interval(float.fromhex('0x3p0'), float.fromhex('0x3.71cb6c53e68eep+0')) * Interval(float.fromhex('0x5p-25'), float.fromhex('0x2.48380232f6c16p+0')), Interval(float.fromhex('0xfp-25'), float.fromhex('0x7.dc58fb323ad7cp+0')))
    def test_837(self):
        self.assertEqual(Interval(float.fromhex('0x3.10e8a605572p-4'), float.fromhex('0x2.48380232f6c16p+0')) * Interval(float.fromhex('0xc.3d8e305214ecp-4'), float.fromhex('0x2.9e7db05203c88p+0')), Interval(float.fromhex('0x2.587a32d02bc04p-4'), float.fromhex('0x5.fa216b7c20c6cp+0')))
suite.addTest(TestCase_mpfi_mul())

class TestCase_mpfi_mul_d(unittest.TestCase):
    """mpfi_mul_d"""
    # special values
    def test_838(self):
        self.assertEqual(Interval(-inf, -7.0) * Interval(float.fromhex('-0x17p0'), float.fromhex('-0x17p0')), Interval(float.fromhex('+0xa1p0'), inf))
    def test_839(self):
        self.assertEqual(Interval(-inf, -7.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_840(self):
        self.assertEqual(Interval(-inf, -7.0) * Interval(float.fromhex('0x170ef54646d497p-107'), float.fromhex('0x170ef54646d497p-107')), Interval(-inf, float.fromhex('-0xa168b4ebefd020p-107')))
    def test_841(self):
        self.assertEqual(Interval(-inf, 0.0) * Interval(float.fromhex('-0x170ef54646d497p-106'), float.fromhex('-0x170ef54646d497p-106')), Interval(0.0, inf))
    def test_842(self):
        self.assertEqual(Interval(-inf, 0.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_843(self):
        self.assertEqual(Interval(-inf, 0.0) * Interval(float.fromhex('0x170ef54646d497p-106'), float.fromhex('0x170ef54646d497p-106')), Interval(-inf, 0.0))
    def test_844(self):
        self.assertEqual(Interval(-inf, 8.0) * Interval(float.fromhex('-0x16345785d8a00000p0'), float.fromhex('-0x16345785d8a00000p0')), Interval(float.fromhex('-0xb1a2bc2ec5000000p0'), inf))
    def test_845(self):
        self.assertEqual(Interval(-inf, 8.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_846(self):
        self.assertEqual(Interval(-inf, 8.0) * Interval(float.fromhex('0x16345785d8a00000p0'), float.fromhex('0x16345785d8a00000p0')), Interval(-inf, float.fromhex('0xb1a2bc2ec5000000p0')))
    def test_847(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(float.fromhex('-0x170ef54646d497p-105'), float.fromhex('-0x170ef54646d497p-105')), Interval(-math.inf, math.inf))
    def test_848(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(0.0e-17, 0.0e-17), Interval(0.0, 0.0))
    def test_849(self):
        self.assertEqual(Interval(-math.inf, math.inf) * Interval(float.fromhex('+0x170ef54646d497p-105'), float.fromhex('+0x170ef54646d497p-105')), Interval(-math.inf, math.inf))
    def test_850(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(float.fromhex('-0x170ef54646d497p-109'), float.fromhex('-0x170ef54646d497p-109')), Interval(0.0, 0.0))
    def test_851(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_852(self):
        self.assertEqual(Interval(0.0, 0.0) * Interval(float.fromhex('0x170ef54646d497p-109'), float.fromhex('0x170ef54646d497p-109')), Interval(0.0, 0.0))
    def test_853(self):
        self.assertEqual(Interval(0.0, 7.0) * Interval(float.fromhex('-0x114b37f4b51f71p-107'), float.fromhex('-0x114b37f4b51f71p-107')), Interval(float.fromhex('-0x790e87b0f3dc18p-107'), 0.0))
    def test_854(self):
        self.assertEqual(Interval(0.0, 8.0) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_855(self):
        self.assertEqual(Interval(0.0, 9.0) * Interval(float.fromhex('0x114b37f4b51f71p-103'), float.fromhex('0x114b37f4b51f71p-103')), Interval(0.0, float.fromhex('0x9ba4f79a5e1b00p-103')))
    def test_856(self):
        self.assertEqual(Interval(0.0, inf) * Interval(float.fromhex('-0x50b45a75f7e81p-104'), float.fromhex('-0x50b45a75f7e81p-104')), Interval(-inf, 0.0))
    def test_857(self):
        self.assertEqual(Interval(0.0, inf) * Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_858(self):
        self.assertEqual(Interval(0.0, inf) * Interval(float.fromhex('0x142d169d7dfa03p-106'), float.fromhex('0x142d169d7dfa03p-106')), Interval(0.0, inf))
    # regular values
    def test_859(self):
        self.assertEqual(Interval(float.fromhex('-0x1717170p0'), float.fromhex('-0xaaaaaaaaaaaaap-123')) * Interval(-1.5, -1.5), Interval(float.fromhex('0xfffffffffffffp-123'), float.fromhex('0x22a2a28p0')))
    def test_860(self):
        self.assertEqual(Interval(float.fromhex('-0xaaaaaaaaaaaaap0'), float.fromhex('0x1717170p+401')) * Interval(-1.5, -1.5), Interval(float.fromhex('-0x22a2a28p+401'), float.fromhex('0xfffffffffffffp0')))
    def test_861(self):
        self.assertEqual(Interval(float.fromhex('0x10000000000010p0'), float.fromhex('0x888888888888p+654')) * Interval(-2.125, -2.125), Interval(float.fromhex('-0x1222222222221p+654'), float.fromhex('-0x22000000000022p0')))
    def test_862(self):
        self.assertEqual(Interval(float.fromhex('-0x1717170p0'), float.fromhex('-0xaaaaaaaaaaaaap-123')) * Interval(1.5, 1.5), Interval(float.fromhex('-0x22a2a28p0'), float.fromhex('-0xfffffffffffffp-123')))
    def test_863(self):
        self.assertEqual(Interval(float.fromhex('-0xaaaaaaaaaaaaap0'), float.fromhex('0x1717170p+401')) * Interval(1.5, 1.5), Interval(float.fromhex('-0xfffffffffffffp0'), float.fromhex('0x22a2a28p+401')))
    def test_864(self):
        self.assertEqual(Interval(float.fromhex('0x10000000000010p0'), float.fromhex('0x888888888888p+654')) * Interval(2.125, 2.125), Interval(float.fromhex('0x22000000000022p0'), float.fromhex('0x1222222222221p+654')))
    def test_865(self):
        self.assertEqual(Interval(float.fromhex('-0x1717170p+36'), float.fromhex('-0x10000000000001p0')) * Interval(-1.5, -1.5), Interval(float.fromhex('0x18000000000001p0'), float.fromhex('0x22a2a28p+36')))
    def test_866(self):
        self.assertEqual(Interval(float.fromhex('-0xaaaaaaaaaaaaap0'), float.fromhex('0x10000000000001p0')) * Interval(-1.5, -1.5), Interval(float.fromhex('-0x18000000000002p0'), float.fromhex('0xfffffffffffffp0')))
    def test_867(self):
        self.assertEqual(Interval(float.fromhex('0x10000000000010p0'), float.fromhex('0x11111111111111p0')) * Interval(-2.125, -2.125), Interval(float.fromhex('-0x12222222222223p+1'), float.fromhex('-0x22000000000022p0')))
    def test_868(self):
        self.assertEqual(Interval(float.fromhex('-0x10000000000001p0'), float.fromhex('-0xaaaaaaaaaaaaap-123')) * Interval(1.5, 1.5), Interval(float.fromhex('-0x18000000000002p0'), float.fromhex('-0xfffffffffffffp-123')))
    def test_869(self):
        self.assertEqual(Interval(float.fromhex('-0xaaaaaaaaaaaabp0'), float.fromhex('0x1717170p+401')) * Interval(1.5, 1.5), Interval(float.fromhex('-0x10000000000001p0'), float.fromhex('0x22a2a28p+401')))
    def test_870(self):
        self.assertEqual(Interval(float.fromhex('0x10000000000001p0'), float.fromhex('0x888888888888p+654')) * Interval(2.125, 2.125), Interval(float.fromhex('0x22000000000002p0'), float.fromhex('0x1222222222221p+654')))
    def test_871(self):
        self.assertEqual(Interval(float.fromhex('-0x11717171717171p0'), float.fromhex('-0xaaaaaaaaaaaaap-123')) * Interval(-1.5, -1.5), Interval(float.fromhex('0xfffffffffffffp-123'), float.fromhex('0x1a2a2a2a2a2a2ap0')))
    def test_872(self):
        self.assertEqual(Interval(float.fromhex('-0x10000000000001p0'), float.fromhex('0x1717170p+401')) * Interval(-1.5, -1.5), Interval(float.fromhex('-0x22a2a28p+401'), float.fromhex('0x18000000000002p0')))
    def test_873(self):
        self.assertEqual(Interval(float.fromhex('0x10000000000001p0'), float.fromhex('0x888888888888p+654')) * Interval(-2.125, -2.125), Interval(float.fromhex('-0x1222222222221p+654'), float.fromhex('-0x22000000000002p0')))
    def test_874(self):
        self.assertEqual(Interval(float.fromhex('-0x1717170p0'), float.fromhex('-0x1aaaaaaaaaaaaap-123')) * Interval(1.5, 1.5), Interval(float.fromhex('-0x22a2a28p0'), float.fromhex('-0x27fffffffffffep-123')))
    def test_875(self):
        self.assertEqual(Interval(float.fromhex('-0xaaaaaaaaaaaaap0'), float.fromhex('0x11717171717171p0')) * Interval(1.5, 1.5), Interval(float.fromhex('-0xfffffffffffffp0'), float.fromhex('0x1a2a2a2a2a2a2ap0')))
    def test_876(self):
        self.assertEqual(Interval(float.fromhex('0x10000000000010p0'), float.fromhex('0x18888888888889p0')) * Interval(2.125, 2.125), Interval(float.fromhex('0x22000000000022p0'), float.fromhex('0x34222222222224p0')))
    def test_877(self):
        self.assertEqual(Interval(float.fromhex('-0x11717171717171p0'), float.fromhex('-0x10000000000001p0')) * Interval(-1.5, -1.5), Interval(float.fromhex('0x18000000000001p0'), float.fromhex('0x1a2a2a2a2a2a2ap0')))
    def test_878(self):
        self.assertEqual(Interval(float.fromhex('-0x10000000000001p0'), float.fromhex('0x10000000000001p0')) * Interval(-1.5, -1.5), Interval(float.fromhex('-0x18000000000002p0'), float.fromhex('0x18000000000002p0')))
    def test_879(self):
        self.assertEqual(Interval(float.fromhex('0x10000000000001p0'), float.fromhex('0x11111111111111p0')) * Interval(-2.125, -2.125), Interval(float.fromhex('-0x12222222222223p+1'), float.fromhex('-0x22000000000002p0')))
    def test_880(self):
        self.assertEqual(Interval(float.fromhex('-0x10000000000001p0'), float.fromhex('-0x1aaaaaaaaaaaaap-123')) * Interval(1.5, 1.5), Interval(float.fromhex('-0x18000000000002p0'), float.fromhex('-0x27fffffffffffep-123')))
    def test_881(self):
        self.assertEqual(Interval(float.fromhex('-0xaaaaaaaaaaaabp0'), float.fromhex('0x11717171717171p0')) * Interval(1.5, 1.5), Interval(float.fromhex('-0x10000000000001p0'), float.fromhex('0x1a2a2a2a2a2a2ap0')))
    def test_882(self):
        self.assertEqual(Interval(float.fromhex('0x10000000000001p0'), float.fromhex('0x18888888888889p0')) * Interval(2.125, 2.125), Interval(float.fromhex('0x22000000000002p0'), float.fromhex('0x34222222222224p0')))
suite.addTest(TestCase_mpfi_mul_d())

class TestCase_mpfi_neg(unittest.TestCase):
    """mpfi_neg"""
    # special values
    def test_883(self):
        self.assertEqual(-(Interval(-inf, -7.0)), Interval(+7.0, inf))
    def test_884(self):
        self.assertEqual(-(Interval(-inf, 0.0)), Interval(0.0, inf))
    def test_885(self):
        self.assertEqual(-(Interval(-inf, +8.0)), Interval(-8.0, inf))
    def test_886(self):
        self.assertEqual(-(Interval(-math.inf, math.inf)), Interval(-math.inf, math.inf))
    def test_887(self):
        self.assertEqual(-(Interval(0.0, 0.0)), Interval(0.0, 0.0))
    def test_888(self):
        self.assertEqual(-(Interval(0.0, +8.0)), Interval(-8.0, 0.0))
    def test_889(self):
        self.assertEqual(-(Interval(0.0, inf)), Interval(-inf, 0.0))
    # regular values
    def test_890(self):
        self.assertEqual(-(Interval(float.fromhex('0x123456789p-16'), float.fromhex('0x123456799p-16'))), Interval(float.fromhex('-0x123456799p-16'), float.fromhex('-0x123456789p-16')))
suite.addTest(TestCase_mpfi_neg())

class TestCase_mpfi_put_d(unittest.TestCase):
    """mpfi_put_d"""

suite.addTest(TestCase_mpfi_put_d())

class TestCase_mpfi_sec(unittest.TestCase):
    """mpfi_sec"""

suite.addTest(TestCase_mpfi_sec())

class TestCase_mpfi_sech(unittest.TestCase):
    """mpfi_sech"""

suite.addTest(TestCase_mpfi_sech())

class TestCase_mpfi_sin(unittest.TestCase):
    """mpfi_sin"""

suite.addTest(TestCase_mpfi_sin())

class TestCase_mpfi_sinh(unittest.TestCase):
    """mpfi_sinh"""

suite.addTest(TestCase_mpfi_sinh())

class TestCase_mpfi_sqr(unittest.TestCase):
    """mpfi_sqr"""

suite.addTest(TestCase_mpfi_sqr())

class TestCase_mpfi_sqrt(unittest.TestCase):
    """mpfi_sqrt"""

suite.addTest(TestCase_mpfi_sqrt())

class TestCase_mpfi_sub(unittest.TestCase):
    """mpfi_sub"""
    # special values
    def test_1176(self):
        self.assertEqual(Interval(-inf, -7.0) - Interval(-1.0, +8.0), Interval(-inf, -6.0))
    def test_1177(self):
        self.assertEqual(Interval(-inf, 0.0) - Interval(+8.0, inf), Interval(-inf, -8.0))
    def test_1178(self):
        self.assertEqual(Interval(-inf, +8.0) - Interval(0.0, +8.0), Interval(-inf, +8.0))
    def test_1179(self):
        self.assertEqual(Interval(-math.inf, math.inf) - Interval(0.0, +8.0), Interval(-math.inf, math.inf))
    def test_1180(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(-inf, -7.0), Interval(+7.0, inf))
    def test_1181(self):
        self.assertEqual(Interval(0.0, +8.0) - Interval(-7.0, 0.0), Interval(0.0, +15.0))
    def test_1182(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(0.0, +8.0), Interval(-8.0, 0.0))
    def test_1183(self):
        self.assertEqual(Interval(0.0, inf) - Interval(0.0, +8.0), Interval(-8.0, inf))
    def test_1184(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(+8.0, inf), Interval(-inf, -8.0))
    def test_1185(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(-math.inf, math.inf), Interval(-math.inf, math.inf))
    def test_1186(self):
        self.assertEqual(Interval(0.0, +8.0) - Interval(-7.0, +8.0), Interval(-8.0, +15.0))
    def test_1187(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_1188(self):
        self.assertEqual(Interval(0.0, inf) - Interval(0.0, +8.0), Interval(-8.0, inf))
    # regular values
    def test_1189(self):
        self.assertEqual(Interval(-5.0, 59.0) - Interval(17.0, 81.0), Interval(-86.0, 42.0))
    def test_1190(self):
        self.assertEqual(Interval(float.fromhex('-0x1p-300'), float.fromhex('0x123456p+28')) - Interval(float.fromhex('-0x789abcdp0'), float.fromhex('0x10000000000000p-93')), Interval(float.fromhex('-0x10000000000001p-93'), float.fromhex('0x123456789abcdp0')))
    def test_1191(self):
        self.assertEqual(Interval(-4.0, 7.0) - Interval(-3e300, float.fromhex('0x123456789abcdp-17')), Interval(float.fromhex('-0x123456791abcdp-17'), float.fromhex('0x8f596b3002c1bp+947')))
    def test_1192(self):
        self.assertEqual(Interval(float.fromhex('-0x1000100010001p+8'), float.fromhex('0x1p+60')) - Interval(-3e300, float.fromhex('0x1000100010001p0')), Interval(float.fromhex('-0x10101010101011p+4'), float.fromhex('0x8f596b3002c1bp+947')))
    def test_1193(self):
        self.assertEqual(Interval(-5.0, 1.0) - Interval(1.0, float.fromhex('0x1p+70')), Interval(float.fromhex('-0x10000000000001p+18'), 0.0))
    def test_1194(self):
        self.assertEqual(Interval(5.0, float.fromhex('0x1p+70')) - Interval(3.0, 5.0), Interval(0.0, float.fromhex('0x1p+70')))
suite.addTest(TestCase_mpfi_sub())

class TestCase_mpfi_sub_d(unittest.TestCase):
    """mpfi_sub_d"""
    # special values
    def test_1195(self):
        self.assertEqual(Interval(-inf, -7.0) - Interval(float.fromhex('-0x170ef54646d497p-107'), float.fromhex('-0x170ef54646d497p-107')), Interval(-inf, float.fromhex('-0x1bffffffffffffp-50')))
    def test_1196(self):
        self.assertEqual(Interval(-inf, -7.0) - Interval(0.0, 0.0), Interval(-inf, -7.0))
    def test_1197(self):
        self.assertEqual(Interval(-inf, -7.0) - Interval(float.fromhex('0x170ef54646d497p-107'), float.fromhex('0x170ef54646d497p-107')), Interval(-inf, -7.0))
    def test_1198(self):
        self.assertEqual(Interval(-inf, 0.0) - Interval(float.fromhex('-0x170ef54646d497p-106'), float.fromhex('-0x170ef54646d497p-106')), Interval(-inf, float.fromhex('0x170ef54646d497p-106')))
    def test_1199(self):
        self.assertEqual(Interval(-inf, 0.0) - Interval(0.0, 0.0), Interval(-inf, 0.0))
    def test_1200(self):
        self.assertEqual(Interval(-inf, 0.0) - Interval(float.fromhex('0x170ef54646d497p-106'), float.fromhex('0x170ef54646d497p-106')), Interval(-inf, -8.0e-17))
    def test_1201(self):
        self.assertEqual(Interval(-inf, 8.0) - Interval(float.fromhex('-0x16345785d8a00000p0'), float.fromhex('-0x16345785d8a00000p0')), Interval(-inf, float.fromhex('0x16345785d8a00100p0')))
    def test_1202(self):
        self.assertEqual(Interval(-inf, 8.0) - Interval(0.0, 0.0), Interval(-inf, 8.0))
    def test_1203(self):
        self.assertEqual(Interval(-inf, 8.0) - Interval(float.fromhex('0x16345785d8a00000p0'), float.fromhex('0x16345785d8a00000p0')), Interval(-inf, float.fromhex('-0x16345785d89fff00p0')))
    def test_1204(self):
        self.assertEqual(Interval(-math.inf, math.inf) - Interval(float.fromhex('-0x170ef54646d497p-105'), float.fromhex('-0x170ef54646d497p-105')), Interval(-math.inf, math.inf))
    def test_1205(self):
        self.assertEqual(Interval(-math.inf, math.inf) - Interval(0.0e-17, 0.0e-17), Interval(-math.inf, math.inf))
    def test_1206(self):
        self.assertEqual(Interval(-math.inf, math.inf) - Interval(float.fromhex('+0x170ef54646d497p-105'), float.fromhex('+0x170ef54646d497p-105')), Interval(-math.inf, math.inf))
    def test_1207(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(float.fromhex('-0x170ef54646d497p-109'), float.fromhex('-0x170ef54646d497p-109')), Interval(float.fromhex('+0x170ef54646d497p-109'), float.fromhex('+0x170ef54646d497p-109')))
    def test_1208(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(0.0, 0.0), Interval(0.0, 0.0))
    def test_1209(self):
        self.assertEqual(Interval(0.0, 0.0) - Interval(float.fromhex('0x170ef54646d497p-109'), float.fromhex('0x170ef54646d497p-109')), Interval(float.fromhex('-0x170ef54646d497p-109'), float.fromhex('-0x170ef54646d497p-109')))
    def test_1210(self):
        self.assertEqual(Interval(0.0, 8.0) - Interval(float.fromhex('-0x114b37f4b51f71p-107'), float.fromhex('-0x114b37f4b51f71p-107')), Interval(float.fromhex('0x114b37f4b51f71p-107'), float.fromhex('0x10000000000001p-49')))
    def test_1211(self):
        self.assertEqual(Interval(0.0, 8.0) - Interval(0.0, 0.0), Interval(0.0, 8.0))
    def test_1212(self):
        self.assertEqual(Interval(0.0, 8.0) - Interval(float.fromhex('0x114b37f4b51f71p-107'), float.fromhex('0x114b37f4b51f71p-107')), Interval(float.fromhex('-0x114b37f4b51f71p-107'), 8.0))
    def test_1213(self):
        self.assertEqual(Interval(0.0, inf) - Interval(float.fromhex('-0x50b45a75f7e81p-104'), float.fromhex('-0x50b45a75f7e81p-104')), Interval(float.fromhex('0x50b45a75f7e81p-104'), inf))
    def test_1214(self):
        self.assertEqual(Interval(0.0, inf) - Interval(0.0, 0.0), Interval(0.0, inf))
    def test_1215(self):
        self.assertEqual(Interval(0.0, inf) - Interval(float.fromhex('0x142d169d7dfa03p-106'), float.fromhex('0x142d169d7dfa03p-106')), Interval(float.fromhex('-0x142d169d7dfa03p-106'), inf))
    # regular values
    def test_1216(self):
        self.assertEqual(Interval(-32.0, -17.0) - Interval(float.fromhex('0xfb53d14aa9c2fp-47'), float.fromhex('0xfb53d14aa9c2fp-47')), Interval(float.fromhex('-0x1fb53d14aa9c2fp-47'), float.fromhex('-0x18353d14aa9c2fp-47')))
    def test_1217(self):
        self.assertEqual(Interval(float.fromhex('-0xfb53d14aa9c2fp-47'), -17.0) - Interval(float.fromhex('-0xfb53d14aa9c2fp-47'), float.fromhex('-0xfb53d14aa9c2fp-47')), Interval(0.0, float.fromhex('0x7353d14aa9c2fp-47')))
    def test_1218(self):
        self.assertEqual(Interval(-32.0, float.fromhex('-0xfb53d14aa9c2fp-48')) - Interval(float.fromhex('-0xfb53d14aa9c2fp-48'), float.fromhex('-0xfb53d14aa9c2fp-48')), Interval(float.fromhex('-0x104ac2eb5563d1p-48'), 0.0))
    def test_1219(self):
        self.assertEqual(Interval(float.fromhex('0x123456789abcdfp-48'), float.fromhex('0x123456789abcdfp-4')) - Interval(-3.5, -3.5), Interval(float.fromhex('0x15b456789abcdfp-48'), float.fromhex('0x123456789abd17p-4')))
    def test_1220(self):
        self.assertEqual(Interval(float.fromhex('0x123456789abcdfp-56'), float.fromhex('0x123456789abcdfp-4')) - Interval(-3.5, -3.5), Interval(float.fromhex('0x3923456789abcdp-52'), float.fromhex('0x123456789abd17p-4')))
    def test_1221(self):
        self.assertEqual(Interval(float.fromhex('-0xffp0'), float.fromhex('0x123456789abcdfp-52')) - Interval(-256.5, -256.5), Interval(float.fromhex('0x18p-4'), float.fromhex('0x101a3456789abdp-44')))
    def test_1222(self):
        self.assertEqual(Interval(float.fromhex('-0x1fffffffffffffp-52'), float.fromhex('-0x1p-550')) - Interval(-4097.5, -4097.5), Interval(float.fromhex('0xfff8p-4'), float.fromhex('0x10018p-4')))
    def test_1223(self):
        self.assertEqual(Interval(float.fromhex('0x123456789abcdfp-48'), float.fromhex('0x123456789abcdfp-4')) - Interval(3.5, 3.5), Interval(float.fromhex('0xeb456789abcdfp-48'), float.fromhex('0x123456789abca7p-4')))
    def test_1224(self):
        self.assertEqual(Interval(float.fromhex('0x123456789abcdfp-56'), float.fromhex('0x123456789abcdfp-4')) - Interval(3.5, 3.5), Interval(float.fromhex('-0x36dcba98765434p-52'), float.fromhex('0x123456789abca7p-4')))
    def test_1225(self):
        self.assertEqual(Interval(float.fromhex('-0xffp0'), float.fromhex('0x123456789abcdfp-52')) - Interval(256.5, 256.5), Interval(float.fromhex('-0x1ff8p-4'), float.fromhex('-0xff5cba9876543p-44')))
    def test_1226(self):
        self.assertEqual(Interval(float.fromhex('-0x1fffffffffffffp-52'), float.fromhex('-0x1p-550')) - Interval(4097.5, 4097.5), Interval(float.fromhex('-0x10038p-4'), float.fromhex('-0x10018p-4')))
suite.addTest(TestCase_mpfi_sub_d())

class TestCase_mpfi_tan(unittest.TestCase):
    """mpfi_tan"""

suite.addTest(TestCase_mpfi_tan())

class TestCase_mpfi_tanh(unittest.TestCase):
    """mpfi_tanh"""

suite.addTest(TestCase_mpfi_tanh())

class TestCase_mpfi_union(unittest.TestCase):
    """mpfi_union"""

suite.addTest(TestCase_mpfi_union())

if __name__ == '__main__':
    unittest.main()
