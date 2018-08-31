
import pytest
import numpy as np 

from GPdoemd.param_estim.param_estim import _residuals

def f (x, p):
	return x * p

N = 10
E = 2
X = np.random.rand(N,E)
P = np.random.rand(E)
Y = f(X,P)

p_bounds = np.array([[0,1]] * E)

class M:
	def __init__ (self):
		self.pmean = None
		self.meas_noise_var = None
		self.call = f


class TestParamEstim:

	def test_residual_noisevar(self):
		model = M()
		model.meas_noise_var = 0.5 * np.ones( E )
		p = np.random.rand(E)
		r = _residuals(p, model, X, Y)
		assert r.shape == (N,E)

	def test_residual_noisecov(self):
		model = M()
		model.meas_noise_var = 0.5 * np.eye( E )
		p = np.random.rand(E)
		r = _residuals(p, model, X, Y)
		assert r.shape == (N,E)