
import pytest
import numpy as np 
import random 
random.seed(12345)

from GPdoemd.models import VanillaGPModel
from GPdoemd.kernels import RBF
from GPdoemd.marginal import taylor_first_order, taylor_second_order
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

M.pmean = ptrue
M.Sigma = laplace_approximation(M, Xs)


"""
TESTS
"""

class TestTaylorApproximation:

	def test_taylor_first_order (self):
		mu, s2 = taylor_first_order(M, Xs)
		assert mu.shape == (N,E)
		assert s2.shape == (N,E,E)

	def test_taylor_second_order (self):
		mu, s2 = taylor_second_order(M, Xs)
		assert mu.shape == (N,E)
		assert s2.shape == (N,E,E)