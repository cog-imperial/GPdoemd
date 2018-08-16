
import pytest
import numpy as np 
from numpy.testing import assert_array_almost_equal

from GPdoemd.models import NumericalModel


"""
SET UP MODEL ONCE
"""
x_bounds = np.array([[10., 20.], [5., 8.]])
p_bounds = np.array([[ 2.,  4.], [3., 5.]])

def f (x, p, grad=False):
	if not grad: 
		return x * p
	return x * p, np.array([[x[0],0],[0,x[1]]])

d = {
	'name':           'testmodel',
	'call':           f,
	'dim_x':          len(x_bounds),
	'dim_p':          len(p_bounds),
	'num_outputs':    2,
	'meas_noise_var': np.array([1,2])
}
Mt       = NumericalModel(d)
Mt.pmean = np.array([2.5, 4.])
Xs       = np.random.uniform([10, 5], [20, 8], size=(10,2))


"""
TESTS
"""
class TestNumericalModel:

	"""
	Test surrogate model
	"""
	def test_predict (self):
		M,S = Mt.predict(Xs)
		assert M.shape == (len(Xs),2)
		assert S.shape == (len(Xs),2)
		assert np.all(S == 0)

	"""
	Test derivatives
	"""
	def test_d_mu_d_p (self):
		for e in range( Mt.num_outputs ):
			der = Mt.d_mu_d_p(e, Xs)
			assert der.shape == (len(Xs), Mt.dim_p)
			dtrue      = np.zeros((len(Xs), Mt.dim_p))
			dtrue[:,e] = Xs[:,e]
			#st()
			assert_array_almost_equal( der, dtrue, decimal=4 )

	def test_d2_mu_d_p2 (self):
		for e in range( Mt.num_outputs ):
			der = Mt.d2_mu_d_p2(e, Xs)
			assert der is NotImplementedError

	def test_d_s2_d_p (self):
		for e in range( Mt.num_outputs ):
			der = Mt.d_s2_d_p(e, Xs)
			assert der is NotImplementedError

	def test_d2_s2_d_p2 (self):
		for e in range( Mt.num_outputs ):
			der = Mt.d2_s2_d_p2(e, Xs)
			assert der is NotImplementedError