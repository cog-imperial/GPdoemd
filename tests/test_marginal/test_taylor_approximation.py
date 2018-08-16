
import random 
random.seed(12345)

import pytest
import numpy as np 

from GPdoemd.marginal import taylor_first_order, taylor_second_order

"""
TEST MODEL
"""
E  = 2
N  = 25
Xs = np.random.rand(N, E)

class SimpleModel:
	def __init__ (self):
		self.num_outputs = E
		self.dim_p       = E
		self.Sigma       = 0.1 * np.eye(self.dim_p)
		self.p           = 0.5 + np.random.rand(self.dim_p)

	def predict (self, x):
		mu = self.p**2 * self.p[::-1] * x**2
		s2 = self.p**1.5 / self.p[::-1] * x
		return mu, s2

	def d_mu_d_p (self, e, x):
		ne        = 1 if e == 0 else 0
		dmu       = np.zeros((len(x), self.dim_p))
		dmu[:,e]  = 2 * self.p[e] * self.p[ne] * x[:,e]**2
		dmu[:,ne] = self.p[e]**2 * x[:,e]**2
		return dmu

	def d2_mu_d_p2 (self, e, x):
		ne          = 1 if e == 0 else 0
		dmu         = np.zeros((len(x), self.dim_p, self.dim_p))
		dmu[:,e,e]  = 2 * self.p[ne] * x[:,e]**2
		dmu[:,e,ne] = 2 * self.p[e]  * x[:,e]**2
		dmu[:,ne,e] = dmu[:,e,ne]
		return dmu

	def d2_s2_d_p2 (self, e, x):
		ne           = 1 if e == 0 else 0
		ds2          = np.zeros((len(x), self.dim_p, self.dim_p))
		ds2[:,e,e]   = 0.75 / (self.p[e]**0.5 * self.p[ne]) * x[:,e]
		ds2[:,e,ne]  = -1.5 * self.p[e]**0.5 / ( self.p[ne]**2 ) * x[:,e]
		ds2[:,ne,e]  = ds2[:,e,ne]
		ds2[:,ne,ne] = 3 * self.p[e]**1.5 / ( self.p[ne]**3 ) * x[:,e]
		return ds2

"""
TESTS
"""
class TestTaylorApproximation:

	def test_taylor_first_order (self):
		M      = SimpleModel()
		mu, s2 = taylor_first_order(M, Xs)
		assert mu.shape == (N,E)
		assert s2.shape == (N,E,E)

	def test_taylor_second_order (self):
		M      = SimpleModel()
		mu, s2 = taylor_second_order(M, Xs)
		assert mu.shape == (N,E)
		assert s2.shape == (N,E,E)