
import pytest
import numpy as np 
from numpy.testing import assert_array_almost_equal

from GPdoemd.models import NumericalModel

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
Mt       = NumericalModel(d)
Mt.pmean = np.array([2.5, 4.])
Xs       = np.random.uniform([10, 5], [20, 8], size=(10,2))


class TestNumericalModel:

	def test_eps (self):
		eps = Mt.eps 
		for i,r in enumerate([1, 2., 3*np.ones(E), [4.]*E, (5,)*E ]):
			Mt.eps = r
			assert isinstance(Mt.eps, np.ndarray)
			assert Mt.eps.shape == (E,) 
			assert np.all( Mt.eps == i+1 ), str(Mt.eps) + ' != ' + str(i+1)
		Mt.eps = 1e-6

	def test_invalid_eps (self):
		for r in [ 'hej', np.ones(E+1) ]:
			with pytest.raises(ValueError):
				Mt.eps = r
		Mt.eps = 1e-6

	def test_d_mu_d_p (self):
		for e in range( Mt.num_outputs ):
			der = Mt.d_mu_d_p(e, Xs)
			assert der.shape == (len(Xs), Mt.dim_p)
			dtrue      = np.zeros((len(Xs), Mt.dim_p))
			dtrue[:,e] = Xs[:,e]
			assert_array_almost_equal( der, dtrue, decimal=4 )