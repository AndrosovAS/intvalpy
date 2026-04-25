#
# 
# Unit tests from libieeep1788 for interval reverse multiplication
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
class TestCase_minimal_mulRevToPair_test(unittest.TestCase):
    """minimal_mulRevToPair_test"""

suite.addTest(TestCase_minimal_mulRevToPair_test())

class TestCase_minimal_mulRevToPair_dec_test(unittest.TestCase):
    """minimal_mulRevToPair_dec_test"""

suite.addTest(TestCase_minimal_mulRevToPair_dec_test())

if __name__ == '__main__':
    unittest.main()
