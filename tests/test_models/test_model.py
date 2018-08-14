
import pytest
import numpy as np 

from GPdoemd.models import Model


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
M  = Model(d)
Xs = np.random.uniform([10, 5], [20, 8], size=(10,2))


"""
TESTS
"""

class TestModel:

	def test_call(self):
		p = np.array([3., 4.])
		F = M.call(Xs, p)
		assert np.all(F == f(Xs, p))
		#_,df = f(Xs, p, grad=True)
		#assert np.all( df == Xs )

	"""
	Dimensions
	"""
	def test_dim_x (self):
		assert M.dim_x == 2
		assert M.dim_p == 2

	"""
	Test surrogate model
	"""
	def test_pmean (self):
		p = np.array([3., 4.])

		Mt = Model(d)
		Mt.pmean = p
		assert Mt._old_pmean is None
		del Mt.pmean
		assert Mt.pmean is None
		assert np.all(Mt._old_pmean == p)

	"""
	Test surrogate model
	"""
	def test_prediction (self):
		Mt = Model(d)
		Mt.pmean = np.array([3., 4.])
		M,S = Mt.predict(Xs)
		assert M.shape == (len(Xs),2)
		assert S.shape == (len(Xs),2)

	"""
	Test dictionary
	"""
	def test_dict (self):
		p = np.array([3., 4.])
		M.pmean = p
		d = M._get_save_dict()
		assert isinstance(d,dict)
		assert np.all(d['pmean'] == p)

	"""
	Test measurement noise variance
	"""
	def test_meas_noise_var (self):
		assert M.meas_noise_var.ndim == 1
		C = M.meas_noise_covar
		assert C.ndim == 2
		assert C[0,0] == 1 and C[0,1] == 0 and C[1,0] == 0 and C[1,1] == 2 



