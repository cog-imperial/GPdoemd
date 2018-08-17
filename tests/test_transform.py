
import pytest
import numpy as np 
from numpy.testing import assert_almost_equal
import warnings

from GPdoemd.transform import BoxTransform, MeanTransform

"""
SET UP MODEL ONCE
"""

xb = np.array([[10., 20.], [5., 8.], [-10, -2], [-1, 10]])
N  = 20
D  = len(xb)
X  = np.random.uniform(xb[:,0], xb[:,1], size=(N, D))
xmin = np.min(X, axis=0)
xmax = np.max(X, axis=0)
BT   = BoxTransform(xmin, xmax)

Y     = X[:,::-1]**2
ymean = np.mean( Y, axis=0 )
ystd  = np.std(  Y, axis=0 )
MT    = MeanTransform(ymean, ystd)


"""
TESTS
"""

class TestBoxTransform:

	def test_trans (self):
		Z = BT(X)
		assert np.all( np.min(Z, axis=0) == np.zeros(D) )
		assert np.all( np.max(Z, axis=0) == np.ones(D) )

	def test_backtrans_min (self):
		Z = BT( np.zeros(D), back=True )
		assert np.all(Z == xmin)

	def test_backtrans_max (self):
		Z = BT( np.ones(D), back=True )
		assert np.all(Z == xmax)

	def _trans_var (self, C, m, M):
		return C / (M - m)**2
	def test_trans_var (self):
		Z  = np.random.rand(D)
		Zt = self._trans_var(Z, xmin, xmax)
		Zp = BT.var( Z )
		assert_almost_equal(Zt, Zp)
		Zp = BT.var( Zp, back=True )
		assert_almost_equal(Z, Zp)

	def _trans_cov (self, C, m, M):
		mt = M - m
		return C / (mt[:,None] * mt[None,:])
	def test_trans_cov(self):
		Z  = np.random.rand(D,D)
		Z  = np.matmul(Z, Z.T)
		Zt = self._trans_cov(Z, xmin, xmax)
		Zp = BT.cov( Z )
		assert_almost_equal(Zt, Zp)
		Zp = BT.cov( Zp, back=True )
		assert_almost_equal(Z, Zp)


class TestMeanTransform:

	def test_trans (self):
		Z = MT(Y)
		assert np.all( np.abs(np.mean(Z, axis=0)) <= 1e-10 )
		assert np.all( np.abs(np.std(Z, axis=0) - 1) <= 1e-10 )

	def test_backtrans_mean (self):
		Z = MT(np.zeros(D), back=True)
		assert np.all(Z == ymean)

	def test_backtrans_std (self):
		Z = np.std( MT( MT(Y), back=True ), axis=0 )
		assert np.all( np.abs(Z - ystd) <= 1e-10 )

	def _trans_var (self, C, std):
		return C / std**2
	def test_trans_var (self):
		Z  = np.random.rand(D)
		Zt = self._trans_var(Z, ystd)
		Zp = MT.var( Z )
		assert_almost_equal(Zt, Zp)
		Zp = MT.var( Zp, back=True )
		assert_almost_equal(Z, Zp)

	def _trans_cov (self, C, std):
		return C / (std[:,None] * std[None,:])
	def test_trans_cov(self):
		Z  = np.random.rand(D,D)
		Z  = np.matmul(Z, Z.T)
		Zt = self._trans_cov(Z, ystd)
		Zp = MT.cov( Z )
		assert_almost_equal(Zt, Zp)
		Zp = MT.cov( Zp, back=True )
		assert_almost_equal(Z, Zp)
