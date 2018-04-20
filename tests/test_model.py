
import pytest
import numpy as np 

from GPdoemd.models import Model
from GPdoemd.marginal import Analytic


"""
SET UP MODEL ONCE
"""

x_bounds = np.array([[10., 20.], [5., 8.]])
p_bounds = np.array([[ 2.,  4.], [3., 5.]])

def f (x, p, grad=False):
	if not grad: 
		return x * p
	return x * p, x 

d = {
	'name':        'testmodel',
	'call':        f,
	'x_bounds':    x_bounds,
	'p_bounds':    p_bounds,
	'num_outputs': 2
}
M  = Model(d)
Xs = np.random.uniform([10, 5], [20, 8], size=(10,2))


"""
TESTS
"""

class TestGPModel:

	def test_call(self):
		p = np.array([3., 4.])
		F = M.call(Xs, p)
		assert np.all(F == f(Xs, p))
		_,df = f(Xs, p, grad=True)
		assert np.all( df == Xs )

	"""
	Dimensions
	"""
	def test_dim_x (self):
		assert M.dim_x == 2
		assert M.dim_p == 2

	"""
	Test surrogate model
	"""
	def test_surrogate (self):
		p = np.array([3., 4.])

		Mt = Model(d)
		Mt.pmean = np.array([3., 4.])
		assert Mt._old_pmean is None

		M,S = Mt.predict(Xs)
		assert M.shape == (len(Xs),2)
		assert S.shape == (len(Xs),2)

		assert Mt.gprm is None
		#Mt.marginal_init_and_compute_covar(Analytic, Xs)
		Mt.marginal_init(Analytic)
		Mt.gprm.Sigma = np.eye(2)
		M,S = Mt.marginal_predict(Xs)
		assert M.shape == (len(Xs),2)
		assert S.shape == (len(Xs),2,2)

	"""
	Test dictionary
	"""
	def test_dict (self):
		d = M._get_save_dict()
		assert isinstance(d,dict)



