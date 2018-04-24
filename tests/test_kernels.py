
import pytest
import numpy as np 

from GPy.models import GPRegression
import GPdoemd.kernels


"""
SET-UP
"""

N  = 10
X  = np.random.rand(N,3)
Y  = np.random.rand(N,1)
d  = X.shape[1]
dr = range(d)

R  = np.random.rand(5, 7)

"""
TESTS
"""

class TestStationaryKernels:

	def _get_gp (self, kern):
		return GPRegression(X, Y, kern(d, dr, 'name'))

	def test_stationary(self):
		k   = GPdoemd.kernels.RBF
		gp  = self._get_gp(k)

		n,m = 5,7
		X1  = np.random.rand(n,d)
		X2  = np.random.rand(m,d)
		
		dx = gp.kern.d_r_d_x(X1,X2)
		assert dx.shape == (n,m,d)
		
		dx = gp.kern.d2_r_d_x2(X1,X2)
		assert dx.shape == (n,m,d,d)
		
		dx = gp.kern.d_k_d_x(X1,X2)
		assert dx.shape == (n,m,d)
		
		dx = gp.kern.d2_k_d_x2(X1,X2)
		assert dx.shape == (n,m,d,d)

	def test_rbf(self):
		k   = GPdoemd.kernels.RBF
		gp  = self._get_gp(k)
		ddK = gp.kern.dK2_drdr(R)
		assert ddK.shape == R.shape

	def test_exponential(self):
		k   = GPdoemd.kernels.Exponential
		gp  = self._get_gp(k)
		ddK = gp.kern.dK2_drdr(R)
		assert ddK.shape == R.shape

	def test_matern32(self):
		k   = GPdoemd.kernels.Matern32
		gp  = self._get_gp(k)
		ddK = gp.kern.dK2_drdr(R)
		assert ddK.shape == R.shape

	def test_matern52(self):
		k   = GPdoemd.kernels.Matern52
		gp  = self._get_gp(k)
		ddK = gp.kern.dK2_drdr(R)
		assert ddK.shape == R.shape

	def test_cosine(self):
		k   = GPdoemd.kernels.Cosine
		gp  = self._get_gp(k)
		ddK = gp.kern.dK2_drdr(R)
		assert ddK.shape == R.shape

	def test_ratquad(self):
		k   = GPdoemd.kernels.RatQuad
		gp  = self._get_gp(k)
		ddK = gp.kern.dK2_drdr(R)
		assert ddK.shape == R.shape
