
import pytest
import numpy as np 

from GPdoemd.models import AnalyticModel

"""
SET UP MODEL ONCE
"""
def f (x, p, grad=False):
	if not grad: 
		return x * p
	return x * p, np.array([[x[0],0],[0,x[1]]])

E = 2
d = {
	'name':           'testmodel',
	'call':           f,
	'dim_x':          E,
	'dim_p':          E,
	'num_outputs':    E,
	'meas_noise_var': np.array([1,2])
	}
Mt       = AnalyticModel(d)
Mt.pmean = np.array([2.5, 4.])
Xs       = np.random.uniform([10, 5], [20, 8], size=(10,2))


"""
TESTS
"""
class TestAnalyticModel:

	def test_d_mu_d_p (self):
		for e in range( E ):
			der = Mt.d_mu_d_p(e, Xs)
			assert der.shape == (len(Xs), Mt.dim_p)
			dtrue      = np.zeros((len(Xs), Mt.dim_p))
			dtrue[:,e] = Xs[:,e]
			assert np.all( der == dtrue )

			