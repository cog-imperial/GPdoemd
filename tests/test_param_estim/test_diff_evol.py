
import pytest
import numpy as np 

from GPdoemd.param_estim import diff_evol

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

	def _p_close_enough (self, p):
		diff = np.sum( (np.round(P,2) - np.round(p,2))**2 )
		print(diff)
		return diff < 1e-4

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