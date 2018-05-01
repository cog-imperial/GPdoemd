
import pytest
import numpy as np 
import random 
random.seed(12345)

from GPdoemd.models import GPModel
from GPdoemd.kernels import RBF
from GPdoemd.marginal import GPMarginal as marg
from GPdoemd.marginal import TaylorFirstOrder
from GPdoemd.marginal import TaylorSecondOrder

"""
SET UP MODEL ONCE
"""

E = 2
D = E

xb = np.array([[0., 1.]] * E)
pb = np.array([[0., 1.]] * E)

ptrue = 0.5 * np.ones( D ) 
mvar  = 0.05

def f (x, p):
	return x * p

d = {
	'name':        'testmodel',
	'call':        f,
	'x_bounds':    xb,
	'p_bounds':    pb,
	'num_outputs': E
}
M = GPModel(d)

N  = 50
Xs = np.random.rand(N, D)
Ps = ptrue + 0.001 * np.random.randn(N, D)
Zs = np.c_[Xs, Ps]
Ys = f(Xs, Ps)
M.gp_surrogate(Zs, Ys, RBF, RBF)

h_len = len( M.gps[0][0][:] )
hyp   = np.array([10] + [1e-1]*E + [10] + [1e-1]*E + [1e-5])
hyps  = [[ hyp for _ in gps] for gps in M.gps]
M.hyp = hyps
M.gp_load_hyp()


"""
TESTS
"""

class TestMarginal:

	def test_set_up (self):
		m = marg( M, ptrue )
		assert np.all( m.param_mean == ptrue )
		assert m.Sigma is None
		assert m.bin_var == []
		assert len(m.gps) == E

	def test_compute_param_covar (self):
		m = marg( M, ptrue )

		def diff (S1,S2):
			return np.abs(S1 - S2) / np.abs(S1 + S2 + 1e-300)

		def sigma (order):
			s = [
				mvar,
				mvar * np.ones(E),
				mvar * np.eye(E),
				mvar * np.eye(E) + 1e-10 * np.ones((E,E))
				]
			return s[order]

		m.compute_param_covar(Xs, sigma(0))
		assert m.Sigma.shape == (D,D)
		assert not np.any( np.isnan(m.Sigma) )
		S = m.Sigma.copy()

		m.compute_param_covar(Xs, sigma(1))
		assert m.Sigma.shape == (D,D)
		assert not np.any( np.isnan(m.Sigma) )
		d = diff(S, m.Sigma)
		assert np.all( d < 1e-6 )

		m.compute_param_covar(Xs, sigma(2))
		assert m.Sigma.shape == (D,D)
		assert not np.any( np.isnan(m.Sigma) )
		d = diff(S, m.Sigma)
		assert np.all( d < 1e-6 )

		m.compute_param_covar(Xs, sigma(3))
		assert m.Sigma.shape == (D,D)
		assert not np.any( np.isnan(m.Sigma) )
		d = diff( np.diag(S), np.diag(m.Sigma) )
		assert np.all( d < 1e-6 )

	def test_Taylor_first_order (self):
		m = TaylorFirstOrder( M, ptrue )
		m.compute_param_covar(Xs, mvar)

		mu, s2 = m(Xs)
		assert mu.shape == (N,E)
		assert s2.shape == (N,E,E)

	def test_Taylor_second_order (self):
		m = TaylorSecondOrder( M, ptrue )
		m.compute_param_covar(Xs, mvar)

		mu, s2 = m(Xs)
		assert mu.shape == (N,E)
		assert s2.shape == (N,E,E)


T = TestMarginal()
T.test_compute_param_covar()