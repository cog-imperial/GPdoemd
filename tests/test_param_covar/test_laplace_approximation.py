
import pytest
import numpy as np 
import random 
random.seed(12345)

from GPdoemd.models import VanillaGPModel
from GPdoemd.kernels import RBF
from GPdoemd.param_covar import laplace_approximation

"""
SET UP MODEL ONCE
"""
E = 2
D = E

xb = np.array([[0., 1.]] * E)
pb = np.array([[0., 1.]] * D)

ptrue = 0.5 * np.ones( D ) 
mvar  = 0.05

def f (x, p):
	return x * p

d = {
	'name':        'testmodel',
	'call':        f,
	'dim_x':       len(xb),
	'dim_p':       len(pb),
	'num_outputs': E
}
M = VanillaGPModel(d)
M.pmean = ptrue

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

class TestLaplaceApproximation:

	def test_laplace_approximation (self):

		def diff (S1,S2):
			return np.abs(S1 - S2) / np.abs(S1 + S2 + 1e-300)

		M.meas_noise_var = mvar
		Sigma = laplace_approximation(M, Xs)
		assert Sigma.shape == (D,D)
		assert not np.any( np.isnan(Sigma) )
		S = Sigma.copy()

		M.meas_noise_var = mvar * np.ones(E)
		Sigma = laplace_approximation(M, Xs)
		assert Sigma.shape == (D,D)
		assert not np.any( np.isnan(Sigma) )
		d = diff(S, Sigma)
		assert np.all( d < 1e-6 )

		M.meas_noise_var = mvar * np.eye(E)
		Sigma = laplace_approximation(M, Xs)
		assert Sigma.shape == (D,D)
		assert not np.any( np.isnan(Sigma) )
		d = diff(S, Sigma)
		assert np.all( d < 1e-6 )

		M.meas_noise_var = mvar * np.eye(E) + 1e-10 * np.ones((E,E))
		Sigma = laplace_approximation(M, Xs)
		assert Sigma.shape == (D,D)
		assert not np.any( np.isnan(Sigma) )
		d = diff( np.diag(S), np.diag(Sigma) )
		assert np.all( d < 1e-6 )