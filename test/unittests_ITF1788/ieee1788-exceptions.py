#
# 
# Test cases for interval exceptions from IEEE Std 1788-2015
# 
# Copyright 2016 Oliver Heimlich (oheim@posteo.de)
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
class TestCase_exceptions(unittest.TestCase):
    """exceptions"""

suite.addTest(TestCase_exceptions())

if __name__ == '__main__':
    unittest.main()
