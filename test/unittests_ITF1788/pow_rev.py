#
# 
# Test cases for reverse interval power operations.
# 
# Copyright 2015-2016 Oliver Heimlich
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
#
# The following tests use boundaries for the first parameter whose reciprocal
# can be computed without round-off error in a binary floating-point context.
# Thus, an implementation should be able to compute tight results with the
# formula x = z ^ (1 / y) for the intervals used here.
# 
# The test values are structured according to table B.1 in
# Heimlich, Oliver. 2011. “The General Interval Power Function.”
# Diplomarbeit, Institute for Computer Science, University of Würzburg.
# http://exp.ln0.de/heimlich-power-2011.htm.
#
class TestCase_minimal_powRev1_test(unittest.TestCase):
    """minimal_powRev1_test"""

suite.addTest(TestCase_minimal_powRev1_test())

#
# The following tests use boundaries for the first and second parameter
# whose binary logarithm can be computed without round-off error in a
# binary floating-point context.
# Thus, an implementation should be able to compute tight results with the
# formula y = log2 z / log2 x for the intervals used here.
# Implementations which use natural logarithm would introduce additional
# errors.
# 
# The test values are structured according to table B.2 in
# Heimlich, Oliver. 2011. “The General Interval Power Function.”
# Diplomarbeit, Institute for Computer Science, University of Würzburg.
# http://exp.ln0.de/heimlich-power-2011.htm.
#
class TestCase_minimal_powRev2_test(unittest.TestCase):
    """minimal_powRev2_test"""

suite.addTest(TestCase_minimal_powRev2_test())

if __name__ == '__main__':
    unittest.main()
