
import random 
random.seed(12345)

import pytest
import numpy as np 

from GPdoemd.param_covar import laplace_approximation

"""
TEST MODEL
"""
E    = 2
D    = E
N    = 25
Xs   = np.random.rand(N, D)
mvar = 0.05

class TestModel:
	def __init__ (self, meas_noise_var):
		self.num_outputs    = E
		self.dim_p          = E
		self.meas_noise_var = meas_noise_var

	def d_mu_d_p (self, e, x):
		dmu      = np.zeros((len(x),self.dim_p))
		dmu[:,e] = x[:,e]**(e+1)
		return dmu

"""
TESTS
"""
class TestLaplaceApproximation:

	def test_laplace_approximation (self):

		def diff (S1,S2):
			return np.abs(S1 - S2) / np.abs(S1 + S2 + 1e-300)

		M = TestModel(mvar)
		Sigma = laplace_approximation(M, Xs)
		assert Sigma.shape == (D,D)
		assert not np.any( np.isnan(Sigma) )
		S = Sigma.copy()

		M = TestModel(mvar * np.ones(E))
		Sigma = laplace_approximation(M, Xs)
		assert Sigma.shape == (D,D)
		assert not np.any( np.isnan(Sigma) )
		d = diff(S, Sigma)
		assert np.all( d < 1e-6 )

		M = TestModel(mvar * np.eye(E))
		Sigma = laplace_approximation(M, Xs)
		assert Sigma.shape == (D,D)
		assert not np.any( np.isnan(Sigma) )
		d = diff(S, Sigma)
		assert np.all( d < 1e-6 )

		M = TestModel(mvar * np.eye(E) + 1e-10 * np.ones((E,E)))
		Sigma = laplace_approximation(M, Xs)
		assert Sigma.shape == (D,D)
		assert not np.any( np.isnan(Sigma) )
		d = diff( np.diag(S), np.diag(Sigma) )
		assert np.all( d < 1e-6 )