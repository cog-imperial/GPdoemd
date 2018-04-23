
import pytest
import numpy as np 
from GPdoemd.utils import binary_dimensions

"""
TESTS
"""

class TestUtils:

	def test_1_binary_dimension(self):
		bs = [2]
		Z  = np.array( [1, -1, 1] )
		N  = len(Z)
		Z  = np.c_[ np.arange(N), np.ones(N), Z ]

		L  = np.arange( 2 )
		I  = np.array([ 0, 1 ])
		J  = np.array([ 1, 0, 1 ])

		l, i, j = binary_dimensions(Z, bs)

		assert np.all( l == L )
		assert np.all( i == I )
		assert np.all( j == J )
		

	def test_2_binary_dimensions(self):
		bs = [1, 2]
		Z  = np.array([ [-1, -1],
						[ 1, -1],
						[ 1,  1],
						[-1,  1],
						[-1,  1],
						[ 1, -1],
						[-1, -1],
						[ 1,  1],
						[ 1,  1],
						[-1,  1],
						[-1, -1] ])
		N  = len(Z)
		Z  = np.c_[ np.arange(N), Z, np.ones(N) ]

		L  = np.arange( 4 )
		I  = np.array([ 0, 3 ])
		J  = np.array([ 0, 1, 3, 2, 2, 1, 0, 3, 3, 2, 0 ])

		l, i, j = binary_dimensions(Z, bs)

		assert np.all( l == L )
		assert np.all( i == I )
		assert np.all( j == J )
