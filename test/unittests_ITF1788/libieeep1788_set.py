#
# 
# Unit tests from libieeep1788 for interval set operations
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


def _itf_is_empty_interval(x):
    try:
        return math.isnan(float(x.a)) and math.isnan(float(x.b))
    except Exception:
        return False

def _itf_convex_hull(x, y):
    if _itf_is_empty_interval(x):
        return y
    if _itf_is_empty_interval(y):
        return x
    return Interval(min(x.a, y.a), max(x.b, y.b))
class NotTightest(UserWarning):
    """Happens when a test does not produce the most accurate result"""

suite = unittest.TestSuite()
class TestCase_minimal_intersection_test(unittest.TestCase):
    """minimal_intersection_test"""
    def test_0205_minimal_intersection_test(self):
        self.assertEqual(ip.intersection(Interval(1.0, 3.0), Interval(2.1, 4.0)), Interval(2.1, 3.0))
    def test_0206_minimal_intersection_test(self):
        self.assertEqual(ip.intersection(Interval(1.0, 3.0), Interval(3.0, 4.0)), Interval(3.0, 3.0))
    def test_0207_minimal_intersection_test(self):
        self.assertEqual(ip.intersection(Interval(1.0, 3.0), Interval(math.nan, math.nan)), Interval(math.nan, math.nan))
    def test_0208_minimal_intersection_test(self):
        self.assertEqual(ip.intersection(Interval(-math.inf, math.inf), Interval(math.nan, math.nan)), Interval(math.nan, math.nan))
    def test_0209_minimal_intersection_test(self):
        self.assertEqual(ip.intersection(Interval(1.0, 3.0), Interval(-math.inf, math.inf)), Interval(1.0, 3.0))

suite.addTest(TestCase_minimal_intersection_test())

class TestCase_minimal_intersection_dec_test(unittest.TestCase):
    """minimal_intersection_dec_test"""

suite.addTest(TestCase_minimal_intersection_dec_test())

class TestCase_minimal_convex_hull_test(unittest.TestCase):
    """minimal_convex_hull_test"""
    def test_0200_minimal_convex_hull_test(self):
        self.assertEqual(_itf_convex_hull(Interval(1.0, 3.0), Interval(2.1, 4.0)), Interval(1.0, 4.0))
    def test_0201_minimal_convex_hull_test(self):
        self.assertEqual(_itf_convex_hull(Interval(1.0, 1.0), Interval(2.1, 4.0)), Interval(1.0, 4.0))
    def test_0202_minimal_convex_hull_test(self):
        self.assertEqual(_itf_convex_hull(Interval(1.0, 3.0), Interval(math.nan, math.nan)), Interval(1.0, 3.0))
    def test_0203_minimal_convex_hull_test(self):
        self.assertEqual(_itf_convex_hull(Interval(math.nan, math.nan), Interval(math.nan, math.nan)), Interval(math.nan, math.nan))
    def test_0204_minimal_convex_hull_test(self):
        self.assertEqual(_itf_convex_hull(Interval(1.0, 3.0), Interval(-math.inf, math.inf)), Interval(-math.inf, math.inf))

suite.addTest(TestCase_minimal_convex_hull_test())

class TestCase_minimal_convex_hull_dec_test(unittest.TestCase):
    """minimal_convex_hull_dec_test"""

suite.addTest(TestCase_minimal_convex_hull_dec_test())

if __name__ == '__main__':
    unittest.main()
