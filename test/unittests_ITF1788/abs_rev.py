#
# 
# Test cases for reverse interval absolute value function.
# 
# Copyright 2015-2016 Oliver Heimlich (oheim@posteo.de)
# 
# Copying and distribution of this file, with or without modification,
# are permitted in any medium without royalty provided the copyright
# notice and this notice are preserved.  This file is offered as-is,
# without any warranty.
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
class TestCase_minimal_absRevBin_test(unittest.TestCase):
    """minimal_absRevBin_test"""

suite.addTest(TestCase_minimal_absRevBin_test())

if __name__ == '__main__':
    unittest.main()
