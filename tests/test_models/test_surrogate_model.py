
import random
random.seed(12345)

import pytest
import numpy as np 
import warnings

from GPdoemd.models import SurrogateModel

from pdb import set_trace as st

"""
SET UP MODEL ONCE
"""
x_bounds = np.array([[10., 20.], [5., 8.], [0, 1]])
p_bounds = np.array([[ 2.,  4.], [3., 5.]])

def _f (x, p):
	return x[:2] * p * (0.5 + 1.5 * x[2])

def f (X, P):
	if X.ndim == 1:
		return _f(X, P)
	return np.array([ _f(x,p) for x,p in zip(X,P) ])

Dx, Dp, E = 3, 2, 2

d = {
	'name':             'testmodel',
	'call':             f,
	'dim_x':            Dx,
	'dim_p':            Dp,
	'num_outputs':      E,
	'binary_variables': [2]
}
M = SurrogateModel(d)

X  = np.array([[10., 5.], [10., 8.], [20., 5.], [20., 8.]])
Xs = np.random.uniform([10, 5], [20, 8], size=(25,2))
X  = np.vstack(( X, Xs ))
X  = np.c_[ X, np.random.randint(0,1,len(X)) ]

P  = np.array([[2., 3.], [2., 5.], [4., 3.], [4., 5.]])
Ps = np.random.uniform([2., 3.], [4., 5.], size=(len(X)-4,2))
P = np.vstack(( P, Ps ))

Z = np.c_[X, P]
Y = f(X, P)
M.set_training_data(Z, Y)

ymean = np.mean( Y, axis=0 )
ystd  = np.std(  Y, axis=0 )




class TestSurrogateModel:

	"""
	Binary variables
	"""
	def test_binary_variables_integer (self):
		d2 = d.copy()
		d2['binary_variables'] = 1
		Mt = SurrogateModel(d2)
		assert isinstance(Mt.binary_variables, list)
		assert len( Mt.binary_variables ) == 1
		assert Mt.binary_variables[0] == 1

	def test_binary_variables_illegal_integer (self):
		d2 = d.copy()
		for t in [ -1, 5 ]:
			d2['binary_variables'] = t
			with pytest.raises(AssertionError) as errinfo:
				Mt = SurrogateModel(d2)
			assert 'Value outside range' in str(errinfo.value)

	def test_binary_variables_list (self):
		d2 = d.copy()
		d2['binary_variables'] = [1]
		Mt = SurrogateModel(d2)
		assert isinstance(Mt.binary_variables, list)
		assert len( Mt.binary_variables ) == 1
		assert Mt.binary_variables[0] == 1

	def test_binary_variables_illegal_list (self):
		d2 = d.copy()
		for t in [ [-1], [5], [0,-1], [-1,0], [0,10], [[-1]], [np.array([0])] ]:
			d2['binary_variables'] = t
			with pytest.raises(AssertionError):
				Mt = SurrogateModel(d2)

	def test_binary_variables_else (self):
		d2 = d.copy()
		for t in [ 0.1, None, np.array([1]) ]:
			d2['binary_variables'] = t
			with pytest.raises(ValueError):
				Mt = SurrogateModel(d2)
	
	def test_dim_b (self):
		assert M.dim_b == 1

	def test_get_Z (self):
		Mt = SurrogateModel(d)
		Mt.trans_p = lambda p: p
		z = (3. + np.arange(Dx+Dp))[None,:]
		x = z[:,:Dx]
		p = z[:,Dx:]
		with pytest.raises(AssertionError):
			Z = Mt.get_Z(x)
		Z = Mt.get_Z(x,p)
		assert np.all(Z == z)
		Mt.pmean = p[0]
		Z = Mt.get_Z(x)
		assert np.all(Z == z)

	def _is_none (self, M):
		for v in ['X', 'P', 'Z', 'Y']:
			assert eval('M.'+v.upper()) is None
		for v in ['X', 'P', 'Z']:
			assert not hasattr(M,v.lower()+'min')
			assert not hasattr(M,v.lower()+'max')
		assert not hasattr(M,'ymean')
		assert not hasattr(M,'ystd')

	def _correct_shape (self, M):
		assert M.Y.shape     == P.shape
		assert M.ymean.shape == (E,)
		assert M.ystd.shape  == (E,)
		assert M.Z.shape     == Z.shape
		assert M.zmin.shape  == (Dx+Dp,)
		assert M.zmax.shape  == (Dx+Dp,)
		assert M.X.shape     == X.shape
		assert M.xmin.shape  == (Dx,)
		assert M.xmax.shape  == (Dx,)
		assert M.P.shape     == P.shape
		assert M.pmin.shape  == (Dp,)
		assert M.pmax.shape  == (Dp,)

	def test_training_data (self):
		self._correct_shape(M)
		Mt = SurrogateModel(d)
		self._is_none(Mt)

		Mt.Z = Z
		Mt.Y = Y
		self._correct_shape(Mt)
		Mt.clear_model()
		self._is_none(Mt)

		Mt.set_training_data(Z, Y)
		self._correct_shape(Mt)
		Mt.clear_model()
		self._is_none(Mt)

		Mt.set_training_data(X, P, Y)
		self._correct_shape(Mt)
		Mt.clear_model()
		self._is_none(Mt)

	def test_prediction (self):
		# Nothing given
		Mt       = SurrogateModel(d)
		with pytest.raises(AssertionError):
			M, S = Mt.predict(X)
		# Give mean 
		Mt.pmean = np.random.uniform(*p_bounds.T)
		with pytest.raises(AssertionError):
			M, S = Mt.predict(X)
		# Give training data
		Mt.set_training_data(Z, Y)
		with pytest.raises(AssertionError):
			M, S = Mt.predict(X)
		# Give hyperparameters
		Mt.hyp = [ 0 ] * Mt.num_outputs
		with pytest.raises(NotImplementedError):
			M, S = Mt.predict(X)

	def test_d_mu_d_p (self):
		for e in range( M.num_outputs ):
			with pytest.raises(NotImplementedError):
				der = M.d_mu_d_p(e, X)

	def test_d2_mu_d_p2 (self):
		for e in range( M.num_outputs ):
			with pytest.raises(NotImplementedError):
				der = M.d2_mu_d_p2(e, X)

	def test_d_s2_d_p (self):
		for e in range( M.num_outputs ):
			with pytest.raises(NotImplementedError):
				der = M.d_s2_d_p(e, X)

	def test_d2_s2_d_p2 (self):
		for e in range( M.num_outputs ):
			with pytest.raises(NotImplementedError):
				der = M.d2_s2_d_p2(e, X)

	def test_save_dict (self):
		d = M._get_save_dict()
		assert isinstance(d, dict)
		assert np.all( np.abs(d['Y'] - Y) <= 1e-10 )
		assert np.all( np.abs(d['Z'] - Z) <= 1e-10 )

	def test_load_dict (self):
		Mt = SurrogateModel(d)
		Y  = np.random.randn(25,Mt.num_outputs)
		Mt.Y = Y
		d2 = Mt._get_save_dict()
		pt = np.random.randn(Mt.dim_p)
		d2['pmean'] = pt.copy()
		Mt._load_save_dict(d2)
		assert np.all(Mt.pmean == pt)
		assert np.all( np.abs(d2['Y'] - Y) <= 1e-10 )
		assert Mt.Z is None
