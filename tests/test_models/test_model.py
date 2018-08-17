
import pytest
import numpy as np 

from GPdoemd.models import Model

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
M  = Model(d)
Xs = np.random.uniform([10, 5], [20, 8], size=(10,2))
pt = np.array([3., 4.])


"""
TESTS
"""

class TestModel:

	def test_name (self):
		assert M.name == 'testmodel'

	def test_invalid_name (self):
		with pytest.raises(AssertionError):
			M.name = 3
		
	def test_call(self):
		p = np.array([3., 4.])
		F = M.call(Xs, p)
		assert np.all(F == f(Xs, p))
		
	def test_invalid_call(self):
		with pytest.raises(AssertionError):
			M.call = 3

	def test_num_outputs (self):
		assert M.num_outputs == E

	def test_invalid_num_outputs (self):
		for r in ['hej', -2]:
			with pytest.raises(AssertionError):
				M.num_outputs = r

	def test_meas_noise_var (self):
		assert M.meas_noise_var.ndim == 1
		C = M.meas_noise_covar
		assert C.ndim == 2
		assert C[0,0] == 1 and C[0,1] == 0 and C[1,0] == 0 and C[1,1] == 2

	def test_invalid_meas_noise_var (self):
		R = ['hej', np.array([1.,-1]), np.random.rand(E+1), np.random.rand(E+1,E+1)]
		for r in R:
			with pytest.raises(AssertionError):
				M.meas_noise_var = r


	def test_dim_x_and_p (self):
		assert M.dim_x == 2
		assert M.dim_p == 2

	def test_invalid_dim_x_and_p (self):
		for r in ['hej', -2]:
			with pytest.raises(AssertionError):
				M.dim_x = r
			with pytest.raises(AssertionError):
				M.dim_p = r

	def test_pmean (self):
		assert M.pmean is None
		M.pmean = pt
		assert np.all(M.pmean == pt)
		assert M._old_pmean is None
		del M.pmean
		assert M.pmean is None
		assert np.all(M._old_pmean == pt)

	def test_invalid_pmean (self):
		for r in [1, pt[:1]]
			with pytest.raises(AssertionError):
				M.pmean = r

	def test_Sigma (self):
		assert M.Sigma is None
		M.Sigma = np.eye(E)
		assert isinstance(M.Sigma, np.ndarray)
		del M.Sigma
		assert M.Sigma is None

	def test_invalid_Sigma (self):
		for r in [1, np.random.randn(2*E, 2*E)]
			with pytest.raises(AssertionError):
				M.Sigma = r 

	def test_prediction (self):
		Mt       = Model(d)
		Mt.pmean = pt
		M, S     = Mt.predict(Xs)
		assert M.shape == (len(Xs), E)
		assert S.shape == (len(Xs), E)
		assert np.all(S == 0)

	def test_invalid_prediction (self):
		Mt = Model(d)
		with pytest.raises(AssertionError) as errinfo:
			M, S = Mt.predict(Xs)
		assert 'pmean not set' in str(errinfo.value)

	def test_d_mu_d_p (self):
		for e in range( M.num_outputs ):
			with pytest.raises(NotImplementedError):
				der = M.d_mu_d_p(e, Xs)

	def test_d2_mu_d_p2 (self):
		for e in range( M.num_outputs ):
			with pytest.raises(NotImplementedError):
				der = M.d2_mu_d_p2(e, Xs)

	def test_d_s2_d_p (self):
		for e in range( M.num_outputs ):
			with pytest.raises(NotImplementedError):
				der = M.d_s2_d_p(e, Xs)

	def test_d2_s2_d_p2 (self):
		for e in range( M.num_outputs ):
			with pytest.raises(NotImplementedError):
				der = M.d2_s2_d_p2(e, Xs)

	def test_clear_model (self):
		Mt = Model(d)
		assert Mt.pmean is None and Mt.Sigma is None
		Mt.pmean = pt 
		Mt.Sigma = np.eye(E)
		assert Mt.pmean is not None and Mt.Sigma is not None
		Mt.clear_model()
		assert Mt.pmean is None and Mt.Sigma is None

	def test_dict (self):
		Mt       = Model(d)
		Mt.pmean = pt
		Mt.Sigma = np.eye(E)
		dt       = Mt._get_save_dict()
		assert isinstance(dt, dict)
		assert dt['old_pmean'] is None
		assert np.all(dt['pmean'] == pt)
		assert np.all(dt['Sigma'] == np.eye(E))