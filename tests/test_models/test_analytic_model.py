
import random
random.seed(125)

import pytest
import numpy as np 

from GPdoemd.models import AnalyticModel
from GPdoemd.marginal import Analytic


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
Mt       = AnalyticModel(d)
Mt.pmean = np.array([2.5, 4.])
Xs       = np.random.uniform([10, 5], [20, 8], size=(10,2))


"""
TESTS
"""
class TestAnalyticModel:

	"""
	Test surrogate model
	"""
	def test_predict (self):
		M,S = Mt.predict(Xs)
		assert M.shape == (len(Xs),2)
		assert S.shape == (len(Xs),2)

	def test_uninitialised_gprm (self):
		Mtt = AnalyticModel(d)
		res = Mtt.marginal_compute_covar(Xs)
		assert res is None
		res = Mtt.marginal_predict(Xs)
		assert res is None

	def test_marginal (self):
		Mt.marginal_init_and_compute_covar(Analytic, Xs)
		M,S = Mt.marginal_predict(Xs)
		assert M.shape == (len(Xs),2)
		assert S.shape == (len(Xs),2,2)
		del Mt.gprm
		assert Mt.gprm is None

T = TestAnalyticModel()
T.test_marginal()