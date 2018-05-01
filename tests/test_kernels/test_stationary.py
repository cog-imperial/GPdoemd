
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
R[2,2] = 0.

"""
TESTS
"""

class TestStationaryKernels:

	def _get_gp (self, kern):
		return GPRegression(X, Y, kern(d, dr, 'name'))

	def _dr2 (self, gp):
		return gp.kern.dK2_drdr(R), R.shape

	def _dX (self, gp):
		xx = gp._predictive_variable
		return gp.kern.gradients_X(1,X[:5],xx), (5, X.shape[1])

	def _dXX (self, gp):
		xx = gp._predictive_variable
		return gp.kern.gradients_XX(1,X[:5],xx), (5, N, X.shape[1], X.shape[1])

	def test_rbf(self):
		gp  = self._get_gp( GPdoemd.kernels.RBF )
		D,r = self._dr2( gp )
		assert D.shape == r
		D,r = self._dX( gp )
		assert D.shape == r
		D,r = self._dXX( gp )
		assert D.shape == r

	def test_exponential(self):
		gp  = self._get_gp( GPdoemd.kernels.Exponential )
		D,r = self._dr2( gp )
		assert D.shape == r
		D,r = self._dX( gp )
		assert D.shape == r
		D,r = self._dXX( gp )
		assert D.shape == r

	def test_matern32(self):
		gp  = self._get_gp( GPdoemd.kernels.Matern32 )
		D,r = self._dr2( gp )
		assert D.shape == r
		D,r = self._dX( gp )
		assert D.shape == r
		D,r = self._dXX( gp )
		assert D.shape == r

	def test_matern52(self):
		gp  = self._get_gp( GPdoemd.kernels.Matern52 )
		D,r = self._dr2( gp )
		assert D.shape == r
		D,r = self._dX( gp )
		assert D.shape == r
		D,r = self._dXX( gp )
		assert D.shape == r

	def test_cosine(self):
		gp  = self._get_gp( GPdoemd.kernels.Cosine )
		D,r = self._dr2( gp )
		assert D.shape == r
		D,r = self._dX( gp )
		assert D.shape == r
		D,r = self._dXX( gp )
		assert D.shape == r

	def test_ratquad(self):
		gp  = self._get_gp( GPdoemd.kernels.RatQuad )
		D,r = self._dr2( gp )
		assert D.shape == r
		D,r = self._dX( gp )
		assert D.shape == r
		D,r = self._dXX( gp )
		assert D.shape == r
