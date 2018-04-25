
import pytest
import numpy as np 

from GPdoemd.models import Model
from GPdoemd.marginal import Analytic as marg

"""
SET UP MODEL ONCE
"""
E = 3
D = E

def f (x, p, grad=False):
	if not grad: 
		return x * p
	dx = np.zeros((E,D))
	for i in range(E):
		dx[i,i] = x[i]
	return x * p, dx

X = np.random.randn( 10, D )
p = np.random.rand( D ) 
mvar = 0.05

class Model:
	def __init__ (self):
		self.num_outputs = E
		self.call        = f
M = Model()


"""
TESTS
"""

class TestMarginal:

	def test_set_up (self):
		m = marg( M, p )
		assert m.num_outputs == E
		assert np.all( m.param_mean == p )
		assert m.Sigma is None

	def test_compute_param_covar (self):
		m = marg( M, p )

		def diff (S1,S2):
			return np.abs(S1-S2) / np.abs(S1+S2+1e-300)

		def sigma (order):
			s = [
				mvar,
				mvar * np.ones(E),
				mvar * np.eye(E),
				mvar * np.eye(E) + 1e-8 * np.ones((E,E))
				]
			return s[order]

		m.compute_param_covar(X, sigma(0))
		assert m.Sigma.shape == (D,D)
		assert not np.any( np.isnan(m.Sigma) )
		S = m.Sigma.copy()

		m.compute_param_covar(X, sigma(1))
		assert m.Sigma.shape == (D,D)
		assert not np.any( np.isnan(m.Sigma) )
		d = diff(S, m.Sigma)
		assert np.all( d < 1e-6 )

		m.compute_param_covar(X, sigma(2))
		assert m.Sigma.shape == (D,D)
		assert not np.any( np.isnan(m.Sigma) )
		d = diff(S, m.Sigma)
		assert np.all( d < 1e-6 )

		m.compute_param_covar(X, sigma(3))
		assert m.Sigma.shape == (D,D)
		assert not np.any( np.isnan(m.Sigma) )
		d = diff( np.diag(S), np.diag(m.Sigma) )
		assert np.all( d < 1e-6 )

	def test_call (self):
		m  = marg( M, p )
		m.compute_param_covar(X, mvar)

		N  = 12
		Xs = np.random.randn(N, D)
		mu, s2 = m(Xs)
		assert mu.shape == (N,E)
		assert s2.shape == (N,E,E)