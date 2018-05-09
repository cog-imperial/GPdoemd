
import pytest
import numpy as np 

from GPdoemd.param_estim import residuals, diff_evol, least_squares


"""
SET-UP
"""
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


"""
TESTS
"""

class TestParamEstim:

	def _p_close_enough (self, p):
		diff = np.sum( (np.round(P,2) - np.round(p,2))**2 )
		print(diff)
		return diff < 1e-4

	def test_residual_noisevar(self):
		model = M()
		model.meas_noise_var = 0.5 * np.ones( E )
		p = np.random.rand(E)
		r = residuals(p, model, X, Y)
		assert r.shape == (N,E)

	def test_residual_noisecov(self):
		model = M()
		model.meas_noise_var = 0.5 * np.eye( E )
		p = np.random.rand(E)
		r = residuals(p, model, X, Y)
		assert r.shape == (N,E)

	def test_diff_evol_noisevar(self):
		model = M()
		model.meas_noise_var = 0.5 * np.ones( E )
		p = diff_evol(model, X, Y, p_bounds)
		assert self._p_close_enough(p)

	def test_diff_evol_noisecov(self):
		model = M()
		model.meas_noise_var = 0.5 * np.eye( E )
		p = diff_evol(model, X, Y, p_bounds)
		assert self._p_close_enough(p)

	def test_least_squares_noisevar(self):
		model = M()
		model.meas_noise_var = 0.5 * np.ones( E )
		p = least_squares(model, X, Y, p_bounds)
		assert self._p_close_enough(p)

	def test_least_squares_noisecov(self):
		model = M()
		model.meas_noise_var = 0.5 * np.eye( E )
		p = least_squares(model, X, Y, p_bounds)
		assert self._p_close_enough(p)